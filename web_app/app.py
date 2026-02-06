"""
Brain Tumor Segmentation Web Application
FastAPI backend for processing MRI scans and running segmentation models
"""

import os
import logging
import torch
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import asyncio

from models import load_unet_model, load_segresnet_model, AVAILABLE_MODELS
from mednext_model import (
    load_mednext_model,
    sliding_window_inference,
    robust_zscore_normalize,
    MedNeXtConfig
)
from utils import (
    generate_session_id,
    load_nifti_file,
    normalize_volume,
    create_modality_grid,
    create_multiview_slices,
    get_volume_dimensions,
    store_session_data,
    get_session_data,
    clear_session
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("brain-tumor-webapp")

app = FastAPI(
    title="Brain Tumor Segmentation",
    description="Web application for brain tumor segmentation using U-Net and SegResNet models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Model paths - all models stored in checkpoints directory
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
models_DIR = os.path.join(BASE_DIR, "models")
UNET_MODEL_PATH = os.path.join(models_DIR, "unet_best.pth")
SEGRESNET_MODEL_PATH = os.path.join(models_DIR, "segresnet_best.pth")
MEDNEXT_MODEL_PATH = os.path.join(models_DIR, "MedNeXt.pt")

# Device configuration
current_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global model cache
models_cache = {}


def get_device():
    """Get the current device"""
    return current_device


def set_device(device_name: str):
    """Set the current device and clear model cache"""
    global current_device, models_cache
    
    if device_name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available on this system")
    
    new_device = torch.device(device_name)
    if new_device != current_device:
        # Clear model cache when device changes
        models_cache.clear()
        current_device = new_device
        logger.info("Device changed to: %s", current_device)
    
    return current_device


def crop_or_pad(volume: np.ndarray, target_shape=(128, 128, 128)) -> np.ndarray:
    """Crop or pad volume to target shape (center crop/pad)."""
    current_shape = volume.shape
    result = np.zeros(target_shape, dtype=volume.dtype)

    starts_src = []
    ends_src = []
    starts_dst = []
    ends_dst = []

    for i in range(3):
        if current_shape[i] > target_shape[i]:
            start = (current_shape[i] - target_shape[i]) // 2
            starts_src.append(start)
            ends_src.append(start + target_shape[i])
            starts_dst.append(0)
            ends_dst.append(target_shape[i])
        else:
            start = (target_shape[i] - current_shape[i]) // 2
            starts_src.append(0)
            ends_src.append(current_shape[i])
            starts_dst.append(start)
            ends_dst.append(start + current_shape[i])

    result[
        starts_dst[0]:ends_dst[0],
        starts_dst[1]:ends_dst[1],
        starts_dst[2]:ends_dst[2]
    ] = volume[
        starts_src[0]:ends_src[0],
        starts_src[1]:ends_src[1],
        starts_src[2]:ends_src[2]
    ]

    return result


def get_model(model_name: str):
    """Load model from cache or disk"""
    device = get_device()
    if model_name not in models_cache:
        try:
            if model_name == "unet":
                models_cache[model_name] = load_unet_model(UNET_MODEL_PATH, device)
            elif model_name == "segresnet":
                models_cache[model_name] = load_segresnet_model(SEGRESNET_MODEL_PATH, device)
            elif model_name == "mednext":
                config = MedNeXtConfig()
                config.DEVICE = device
                models_cache[model_name] = load_mednext_model(MEDNEXT_MODEL_PATH, config)
            else:
                raise ValueError(f"Unknown model: {model_name}")
            logger.info("Model loaded and ready: %s on %s", model_name, device)
        except Exception:
            logger.exception("Failed to load model: %s", model_name)
            raise
    return models_cache[model_name]


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main HTML page"""
    with open(os.path.join(STATIC_DIR, "index.html"), "r") as f:
        return f.read()


@app.get("/styles.css")
async def styles():
    """Serve styles for index when using root-relative paths"""
    return FileResponse(os.path.join(STATIC_DIR, "styles.css"))


@app.get("/app.js")
async def app_js():
    """Serve app.js for index when using root-relative paths"""
    return FileResponse(os.path.join(STATIC_DIR, "app.js"))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "device": str(get_device()),
        "cuda_available": torch.cuda.is_available(),
        "models_available": {
            "unet": os.path.exists(UNET_MODEL_PATH),
            "segresnet": os.path.exists(SEGRESNET_MODEL_PATH),
            "mednext": os.path.exists(MEDNEXT_MODEL_PATH)
        }
    }


@app.get("/devices")
async def list_devices():
    """List available compute devices"""
    devices = [{"id": "cpu", "name": "CPU", "available": True}]
    
    if torch.cuda.is_available():
        cuda_name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "CUDA"
        devices.append({"id": "cuda", "name": cuda_name, "available": True})
    else:
        devices.append({"id": "cuda", "name": "CUDA (not available)", "available": False})
    
    return {
        "devices": devices,
        "current_device": str(get_device())
    }


@app.post("/device")
async def change_device(device_name: str = Form(...)):
    """Change the compute device (cpu or cuda)"""
    try:
        if device_name not in ["cpu", "cuda"]:
            raise HTTPException(status_code=400, detail="Device must be 'cpu' or 'cuda'")
        
        new_device = set_device(device_name)
        return {
            "message": f"Device changed to {new_device}",
            "device": str(new_device)
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/upload")
async def upload_files(
    t1: UploadFile = File(...),
    t1ce: UploadFile = File(...),
    t2: UploadFile = File(...),
    flair: UploadFile = File(...)
):
    """Upload 4 MRI modality files"""
    try:
        session_id = generate_session_id()
        
        # Read all files
        modalities = {}
        for name, file in [("t1", t1), ("t1ce", t1ce), ("t2", t2), ("flair", flair)]:
            content = await file.read()
            volume = load_nifti_file(content)
            modalities[name] = volume
        
        # Preprocess modalities for U-Net / SegResNet (crop/pad + normalize)
        processed = np.stack(
            [
                normalize_volume(crop_or_pad(modalities[name]))
                for name in ["t1", "t1ce", "t2", "flair"]
            ],
            axis=0,
        )

        # Full-size modalities for MedNeXt (no resize) and display
        modalities_full_raw = np.stack(
            [modalities[name] for name in ["t1", "t1ce", "t2", "flair"]],
            axis=0,
        )
        modalities_full = np.stack(
            [normalize_volume(modalities[name].copy()) for name in ["t1", "t1ce", "t2", "flair"]],
            axis=0,
        )
        
        # Store in session
        store_session_data(session_id, {
            "modalities": processed,
            "modalities_full": modalities_full,
            "modalities_mednext": modalities_full_raw,
            "original_shape": modalities["t1"].shape,
            "segmentation": None
        })
        
        return {
            "session_id": session_id,
            "message": "Files uploaded successfully",
            "shape": processed.shape,
            "num_slices": processed.shape[-1]
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/segment/{session_id}")
async def run_segmentation(session_id: str, model_name: str = Form(...)):
    """Run segmentation on uploaded data"""
    session = get_session_data(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        # Load model
        model = get_model(model_name)
        device = get_device()
        
        # Prepare input
        modalities = session["modalities"]
        logger.info(
            "Stacking order: Channel 0 (T1), Channel 1 (T1ce), Channel 2 (T2), Channel 3 (FLAIR)"
        )
        
        # MedNeXt uses sliding window inference with special preprocessing
        if model_name == "mednext":
            # Apply robust z-score normalization for MedNeXt
            mednext_modalities = session.get("modalities_mednext", modalities)
            normalized = np.stack([
                robust_zscore_normalize(mednext_modalities[i].copy())
                for i in range(4)
            ])
            volume_tensor = torch.from_numpy(normalized).float()
            
            # Run sliding window inference
            config = MedNeXtConfig()
            config.DEVICE = device
            prediction = sliding_window_inference(
                model, volume_tensor, config,
                patch_size=(128, 128, 128),
                overlap=0.5
            )
        else:
            # Standard inference for U-Net and SegResNet
            input_tensor = torch.from_numpy(modalities).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(input_tensor)
                prediction = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
        
        # Map class 3 back to class 4 (ET) if needed
        # The model outputs classes 0, 1, 2, 3 but ground truth uses 0, 1, 2, 4
        prediction = np.where(prediction == 3, 4, prediction)
        
        # Store segmentation result
        session["segmentation"] = prediction
        session["model_used"] = model_name
        store_session_data(session_id, session)
        
        # Calculate tumor statistics
        total_voxels = prediction.size
        tumor_voxels = {
            "ncr": int(np.sum(prediction == 1)),
            "ed": int(np.sum(prediction == 2)),
            "et": int(np.sum(prediction == 4))
        }
        
        return {
            "session_id": session_id,
            "model_used": model_name,
            "message": "Segmentation completed",
            "tumor_statistics": tumor_voxels,
            "total_voxels": total_voxels,
            "num_slices": prediction.shape[-1]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/slice/{session_id}/{slice_idx}")
async def get_slice(session_id: str, slice_idx: int, overlay: bool = True):
    """Get visualization for a specific slice"""
    session = get_session_data(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    model_used = session.get("model_used")
    if model_used == "mednext" and session.get("modalities_full") is not None:
        modalities = session["modalities_full"]
    else:
        modalities = session["modalities"]
    segmentation = session.get("segmentation")
    
    # Validate slice index
    num_slices = modalities.shape[-1]
    if slice_idx < 0 or slice_idx >= num_slices:
        raise HTTPException(status_code=400, detail=f"Slice index must be between 0 and {num_slices - 1}")
    
    # Create visualization
    images = create_modality_grid(modalities, segmentation, slice_idx)
    
    return {
        "session_id": session_id,
        "slice_idx": slice_idx,
        "num_slices": num_slices,
        "images": images,
        "has_segmentation": segmentation is not None
    }


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Clear session data"""
    clear_session(session_id)
    return {"message": "Session cleared"}


@app.get("/multiview/{session_id}")
async def get_multiview(
    session_id: str,
    axial: int = 64,
    coronal: int = 64,
    sagittal: int = 64,
    modality: str = "flair",
    overlay: bool = True
):
    """
    Get 3D multi-view visualization (axial, coronal, sagittal)
    Returns images for all three anatomical planes
    """
    session = get_session_data(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    model_used = session.get("model_used")
    if model_used == "mednext" and session.get("modalities_full") is not None:
        modalities = session["modalities_full"]
    else:
        modalities = session["modalities"]
    segmentation = session.get("segmentation")
    
    # Get volume dimensions
    dims_raw = get_volume_dimensions(modalities)
    if isinstance(dims_raw, dict):
        dims = dims_raw
    else:
        depth, height, width = dims_raw
        dims = {"depth": depth, "height": height, "width": width}
    
    # Validate slice indices
    if axial < 0 or axial >= dims['width']:
        raise HTTPException(status_code=400, detail=f"Axial index must be between 0 and {dims['width'] - 1}")
    if coronal < 0 or coronal >= dims['height']:
        raise HTTPException(status_code=400, detail=f"Coronal index must be between 0 and {dims['height'] - 1}")
    if sagittal < 0 or sagittal >= dims['depth']:
        raise HTTPException(status_code=400, detail=f"Sagittal index must be between 0 and {dims['depth'] - 1}")
    
    # Create multi-view images
    modality_map = {"t1": 0, "t1ce": 1, "t2": 2, "flair": 3}
    modality_channel = modality_map.get(modality.lower(), 3)

    images = create_multiview_slices(
        modalities,
        segmentation if overlay else None,
        axial_idx=axial,
        coronal_idx=coronal,
        sagittal_idx=sagittal,
        modality_channel=modality_channel,
    )

    no_overlay_images = create_multiview_slices(
        modalities,
        None,
        axial_idx=axial,
        coronal_idx=coronal,
        sagittal_idx=sagittal,
        modality_channel=modality_channel,
    )

    images.update(
        {
            "axial_no_overlay": no_overlay_images.get("axial"),
            "coronal_no_overlay": no_overlay_images.get("coronal"),
            "sagittal_no_overlay": no_overlay_images.get("sagittal"),
        }
    )
    
    return {
        "session_id": session_id,
        "dimensions": dims,
        "current_slices": {
            "axial": axial,
            "coronal": coronal,
            "sagittal": sagittal
        },
        "modality": modality,
        "images": images,
        "has_segmentation": segmentation is not None
    }


@app.get("/models")
async def list_models():
    """List available models"""
    return {
        "models": [
            {
                "id": "unet",
                "name": "U-Net 3D",
                "description": "3D U-Net architecture for volumetric segmentation",
                "icon": "🔬",
                "available": os.path.exists(UNET_MODEL_PATH)
            },
            {
                "id": "segresnet",
                "name": "SegResNet",
                "description": "Residual encoder-decoder network for segmentation",
                "icon": "🧬",
                "available": os.path.exists(SEGRESNET_MODEL_PATH)
            },
            {
                "id": "mednext",
                "name": "MedNeXt",
                "description": "Transformer-inspired ConvNet with sliding window inference",
                "icon": "🧠",
                "available": os.path.exists(MEDNEXT_MODEL_PATH)
            }
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
