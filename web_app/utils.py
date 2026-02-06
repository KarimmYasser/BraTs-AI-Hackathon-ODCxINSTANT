"""
Utility functions for brain tumor segmentation web app
"""

import os
import uuid
import numpy as np
import nibabel as nib
from typing import Dict, Tuple, Optional, List
from PIL import Image
import io
import base64
import tempfile
import logging
import matplotlib
matplotlib.use('Agg') # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Tumor class colors (RGBA)
TUMOR_COLORS = {
    0: (0, 0, 0, 0),        # Background - transparent
    1: (255, 0, 0, 180),    # NCR (Necrotic Core) - Red
    2: (0, 255, 0, 180),    # ED (Edema) - Green
    4: (255, 255, 0, 180),  # ET (Enhancing Tumor) - Yellow
}

# Session storage
sessions: Dict[str, dict] = {}


def generate_session_id() -> str:
    """Generate a unique session ID"""
    return str(uuid.uuid4())


def normalize_volume(volume: np.ndarray) -> np.ndarray:
    """
    Standardize normalization using Z-score of brain region.
    MATCHES NOTEBOOK: (Clip 0.5-99.5%) -> (volume - mean) / std.
    """
    mask = volume > 0
    if mask.sum() == 0:
        return volume

    pixels = volume[mask]
    
    # 1. Robust Clipping (from Notebook)
    p_low, p_high = np.percentile(pixels, 0.5), np.percentile(pixels, 99.5)
    volume = np.clip(volume, p_low, p_high)
    
    # Reload pixels after clipping to calculate robust stats
    pixels = volume[mask]
    mean = pixels.mean()
    std = pixels.std()

    volume = (volume - mean) / (std + 1e-8)
    volume[~mask] = 0
    return volume


def load_nifti_file(file_bytes: bytes) -> np.ndarray:
    """Load a NIfTI file from bytes"""
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        nifti = nib.load(temp_path)
        data = nifti.get_fdata().astype(np.float32)
        return data
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def get_volume_dimensions(volume: np.ndarray) -> Tuple[int, int, int]:
    """Get dimensions (D, H, W) from the stacked volume (C, H, W, D)"""
    depth = volume.shape[3]
    height = volume.shape[1]
    width = volume.shape[2]
    return depth, height, width


def _format_slice(slice_data: np.ndarray) -> np.ndarray:
    """
    Pass-through function. 
    Matplotlib handles normalization/contrast, so we just return the raw float data.
    """
    return slice_data

def create_slice_image(slice_data: np.ndarray, segmentation: Optional[np.ndarray] = None) -> str:
    """
    Generate image using Matplotlib to exactly match notebook visualization.
    Uses BoundaryNorm to ensure correct integer-to-color mapping.
    """
    # Create figure with high DPI for clarity, no frame
    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    ax.axis('off')
    
    # 1. Plot MRI Image (Gray)
    # Matplotlib automatically handles Min-Max scaling for floats
    try:
        # Check for NaN/Inf
        if not np.isfinite(slice_data).all():
             logger.warning(f"Slice data contains NaN/Inf. Replacing with 0.")
             slice_data = np.nan_to_num(slice_data)
             
        ax.imshow(slice_data, cmap='gray', aspect='equal', interpolation='nearest')
    except Exception as e:
        logger.error(f"Error plotting slice data: {e}")
        # Fallback to black image
        ax.imshow(np.zeros((128,128)), cmap='gray')
    
    # 2. Overlay Segmentation (if present)
    if segmentation is not None:
        try:
            # Map values: 1->1 (Red), 2->2 (Green), 4->3 (Yellow)
            seg_plot = segmentation.astype(int).copy()
            seg_plot[seg_plot == 4] = 3
            unique_vals = np.unique(seg_plot)
            if len(unique_vals) > 1: # Only log if interesting
                 logger.info(f"Overlaying segmentation with values: {unique_vals}")

            # Use Masked Array to hide 0s completely (Much safer than alpha mapping)
            masked_seg = np.ma.masked_where(seg_plot == 0, seg_plot)

            # Colors for 1, 2, 3
            colors = ['red', 'green', 'yellow']
            cmap = mcolors.ListedColormap(colors)
            
            # Use BoundaryNorm
            # Bins: [0.5, 1.5] -> Red
            #       [1.5, 2.5] -> Green
            #       [2.5, 3.5] -> Yellow
            bounds = [0.5, 1.5, 2.5, 3.5]
            norm = mcolors.BoundaryNorm(bounds, cmap.N)
            
            ax.imshow(masked_seg, cmap=cmap, norm=norm, alpha=0.6, interpolation='nearest')
        except Exception as e:
            logger.error(f"Error plotting segmentation: {e}")

    # Save to memory buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', transparent=True, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def create_modality_grid(modalities: np.ndarray,
                         segmentation: Optional[np.ndarray],
                         slice_idx: int) -> Dict[str, str]:
    """
    Create visualization for all 4 modalities + segmentation at a given slice.
    Volume shape: (C, H, W, D) -- loaded by nibabel
    We slice along D (index 3) to get Axial view by default.
    """
    names = ['t1', 't1ce', 't2', 'flair']
    result = {}

    for i, name in enumerate(names):
        # CORRECT SLICING: Slice the LAST dimension (Depth)
        # modalities is (C, H, W, D)
        mod_slice = np.rot90(modalities[i, :, :, slice_idx])
        
        # segmentation is (H, W, D)
        seg_slice = np.rot90(segmentation[:, :, slice_idx]) if segmentation is not None else None
        
        result[name] = create_slice_image(mod_slice, seg_slice)
        result[f"{name}_no_overlay"] = create_slice_image(mod_slice, None)

    if segmentation is not None:
        seg_slice = np.rot90(segmentation[:, :, slice_idx])
        
        # Prepare pure segmentation view
        fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        ax.axis('off')
        
        # Background: White (1s)
        ax.imshow(np.ones_like(seg_slice), cmap='gray', vmin=0, vmax=1)
        
        # Overlay Mask
        # Use Masked Array to hide 0s completely (Robust!)
        seg_plot = seg_slice.astype(int).copy()
        seg_plot[seg_plot == 4] = 3
        masked_seg = np.ma.masked_where(seg_plot == 0, seg_plot)

        colors = ['red', 'green', 'yellow'] # 1, 2, 3
        cmap = mcolors.ListedColormap(colors)
        # Bounds for 1, 2, 3
        # 0 is masked, so we don't care about bin 0
        # Bins: [0.5, 1.5] -> Red
        #       [1.5, 2.5] -> Green
        #       [2.5, 3.5] -> Yellow
        bounds = [0.5, 1.5, 2.5, 3.5]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)
        
        ax.imshow(masked_seg, cmap=cmap, norm=norm, alpha=1.0, interpolation='nearest')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', transparent=False, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        
        result['segmentation'] = base64.b64encode(buf.getvalue()).decode('utf-8')

    return result


def create_multiview_slices(volume: np.ndarray, 
                          segmentation: Optional[np.ndarray],
                          axial_idx: int,
                          coronal_idx: int,
                          sagittal_idx: int,
                          modality_channel: int = 3) -> Dict[str, str]:
    
    # Select the requested channel (e.g., FLAIR)
    vol_data = volume[modality_channel]
    result = {}

    h, w, d = vol_data.shape
    
    # Ensure indices are within bounds
    ax_idx = min(axial_idx, d - 1)
    co_idx = min(coronal_idx, h - 1)
    sa_idx = min(sagittal_idx, w - 1)

    # This function expects RAW float data for volume (not uint8)
    # create_slice_image handles the display logic with matplotlib
    
    # --- AXIAL VIEW (slice along D) ---
    sl_ax = np.rot90(vol_data[:, :, ax_idx])
    if segmentation is not None:
        seg_ax = np.rot90(segmentation[:, :, ax_idx])
    else:
        seg_ax = None
        
    result['axial'] = create_slice_image(sl_ax, seg_ax)

    # --- CORONAL VIEW (slice along H) ---
    sl_co = np.rot90(vol_data[co_idx, :, :])
    if segmentation is not None:
        seg_co = np.rot90(segmentation[co_idx, :, :])
    else:
        seg_co = None

    result['coronal'] = create_slice_image(sl_co, seg_co)

    # --- SAGITTAL VIEW (slice along W) ---
    sl_sa = np.rot90(vol_data[:, sa_idx, :])
    if segmentation is not None:
        seg_sa = np.rot90(segmentation[:, sa_idx, :])
    else:
        seg_sa = None

    result['sagittal'] = create_slice_image(sl_sa, seg_sa)

    return result


def store_session_data(session_id: str, data: dict):
    sessions[session_id] = data


def get_session_data(session_id: str) -> Optional[dict]:
    return sessions.get(session_id)


def clear_session(session_id: str):
    if session_id in sessions:
        del sessions[session_id]
