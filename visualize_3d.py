"""
3D Visualization Script for Brain Tumor Segmentation
Renders GT and Prediction as interactive 3D volumes.
"""
import os
import json
import torch
import numpy as np
import nibabel as nib
from monai import transforms
from monai.inferers import sliding_window_inference
from monai.transforms import Activations
from nnunet_mednext import create_mednext_v1
from monai.transforms import AsDiscrete
import argparse

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: plotly not installed. Install with: pip install plotly")

def load_and_preprocess(sample_dir, sample_id, patch_size=[128, 128, 128]):
    """Load raw NIfTI files and preprocess them."""
    modalities = ["t2f", "t1n", "t1c", "t2w"]
    
    images = []
    for mod in modalities:
        path = os.path.join(sample_dir, f"{sample_id}-{mod}.nii")
        if not os.path.exists(path):
            path = os.path.join(sample_dir, f"{sample_id}-{mod}.nii.gz")
        nifti = nib.load(path)
        data = np.abs(nifti.get_fdata().astype(np.float32))
        images.append(data)
    
    image = np.stack(images, axis=0)
    
    seg_path = os.path.join(sample_dir, f"{sample_id}-seg.nii")
    if not os.path.exists(seg_path):
        seg_path = os.path.join(sample_dir, f"{sample_id}-seg.nii.gz")
    seg_nifti = nib.load(seg_path)
    seg = seg_nifti.get_fdata().astype(np.uint8)
    
    bbox = transforms.utils.generate_spatial_bounding_box(image)
    crop = transforms.SpatialCrop(roi_start=bbox[0], roi_end=bbox[1])
    image = crop(image)
    seg = crop(np.expand_dims(seg, 0))[0]
    
    normalize = transforms.NormalizeIntensity(nonzero=True, channel_wise=True)
    image = normalize(image)
    
    image_shape = image.shape[1:]
    pad_shape = [max(ps, ish) for ps, ish in zip(patch_size, image_shape)]
    if pad_shape != list(image_shape):
        paddings = [(ps - ish) / 2 for ps, ish in zip(pad_shape, image_shape)]
        image = np.pad(image, (
            (0, 0),
            (int(np.floor(paddings[0])), int(np.ceil(paddings[0]))),
            (int(np.floor(paddings[1])), int(np.ceil(paddings[1]))),
            (int(np.floor(paddings[2])), int(np.ceil(paddings[2]))),
        ))
        seg = np.pad(seg, (
            (int(np.floor(paddings[0])), int(np.ceil(paddings[0]))),
            (int(np.floor(paddings[1])), int(np.ceil(paddings[1]))),
            (int(np.floor(paddings[2])), int(np.ceil(paddings[2]))),
        ))
    
    return image.astype(np.float32), seg

def convert_labels_to_classes(seg):
    tc = ((seg == 1) | (seg == 4)).astype(np.uint8)
    wt = ((seg == 1) | (seg == 2) | (seg == 4)).astype(np.uint8)
    et = (seg == 4).astype(np.uint8)
    return np.stack([tc, wt, et], axis=0)

def load_model(args, ckpt_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = create_mednext_v1(
        num_input_channels=4,
        num_classes=3,
        model_id=args['mednext_size'],
        kernel_size=args['mednext_ksize'],
        deep_supervision=args['deep_sup'],
        checkpoint_style='outside_block',
    )
    
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, device

def downsample_mask(mask, factor=4):
    """Downsample mask for faster 3D rendering."""
    return mask[::factor, ::factor, ::factor]

def create_3d_mesh(mask, color, name, opacity=0.5, downsample=4):
    """Create a 3D isosurface from a binary mask."""
    mask_ds = downsample_mask(mask, downsample)
    
    # Get coordinates of all positive voxels
    z, y, x = np.where(mask_ds > 0)
    
    if len(x) == 0:
        return go.Scatter3d(x=[], y=[], z=[], mode='markers', name=name)
    
    return go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(size=2, color=color, opacity=opacity),
        name=name
    )

def visualize_3d(gt_mask, pred_mask, save_path=None):
    """Create interactive 3D visualization with plotly."""
    if not PLOTLY_AVAILABLE:
        print("Plotly not available. Skipping 3D visualization.")
        return
    
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
        subplot_titles=('Ground Truth', 'Prediction')
    )
    
    colors = {'TC': 'red', 'WT': 'green', 'ET': 'blue'}
    class_names = ['TC', 'WT', 'ET']
    
    # Ground Truth
    for i, name in enumerate(class_names):
        trace = create_3d_mesh(gt_mask[i], colors[name], f'GT {name}', opacity=0.6)
        fig.add_trace(trace, row=1, col=1)
    
    # Prediction
    for i, name in enumerate(class_names):
        trace = create_3d_mesh(pred_mask[i], colors[name], f'Pred {name}', opacity=0.6)
        fig.add_trace(trace, row=1, col=2)
    
    fig.update_layout(
        title='3D Tumor Segmentation: Ground Truth vs Prediction',
        showlegend=True,
        height=700,
        scene=dict(aspectmode='data'),
        scene2=dict(aspectmode='data')
    )
    
    if save_path:
        fig.write_html(save_path)
        print(f"Saved 3D visualization to {save_path}")
    
    fig.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_dir", type=str, default="one sample")
    parser.add_argument("--sample_id", type=str, default="BraTS2021_00016")
    parser.add_argument("--ckpt", type=str, default="outputs/last.ckpt")
    parser.add_argument("--save", type=str, default="visualization_3d.html")
    parser.add_argument("--downsample", type=int, default=2, help="Downsample factor for faster rendering")
    args = parser.parse_args()
    
    with open("train_args.json", "r") as f:
        train_args = json.load(f)
    
    print(f"Loading sample: {args.sample_id}")
    image, seg = load_and_preprocess(args.sample_dir, args.sample_id)
    gt_mask = convert_labels_to_classes(seg)
    
    print(f"GT class sums - TC: {gt_mask[0].sum()}, WT: {gt_mask[1].sum()}, ET: {gt_mask[2].sum()}")
    
    print(f"Loading model from {args.ckpt}")
    model, device = load_model(train_args, args.ckpt)
    
    roi_size = (train_args['roi_x'], train_args['roi_y'], train_args['roi_z'])
    post_sigmoid = Activations(sigmoid=True)
    post_discrete = AsDiscrete(threshold=0.5)  # Raw predictions without min_et filtering
    
    print("Running inference...")
    image_tensor = torch.from_numpy(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = sliding_window_inference(
            inputs=image_tensor,
            roi_size=roi_size,
            sw_batch_size=4,
            predictor=model,
            overlap=0.5,
        )
        probs = post_sigmoid(logits)[0]
        pred_mask = post_discrete(probs).cpu().numpy().astype(np.uint8)
    
    print(f"Pred class sums - TC: {pred_mask[0].sum()}, WT: {pred_mask[1].sum()}, ET: {pred_mask[2].sum()}")
    
    print("Creating 3D visualization...")
    visualize_3d(gt_mask, pred_mask, save_path=args.save)

if __name__ == "__main__":
    main()
