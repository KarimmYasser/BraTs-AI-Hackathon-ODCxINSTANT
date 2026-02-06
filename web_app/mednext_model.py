"""
MedNeXt Model for Brain Tumor Segmentation
Based on: https://github.com/MIC-DKFZ/MedNeXt
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional


class MedNeXtBlock(nn.Module):
    """
    MedNeXt Block with depthwise separable convolutions and expansion.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        exp_r: int = 4,
        kernel_size: int = 7,
        do_res: bool = True,
        norm_type: str = 'group',
        n_groups: int = None,
    ):
        super().__init__()
        
        self.do_res = do_res
        
        # Depthwise convolution
        self.conv1 = nn.Conv3d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=in_channels,
        )
        
        # Normalization
        if norm_type == 'group':
            groups = n_groups if n_groups else in_channels // 4
            groups = max(1, groups)
            self.norm = nn.GroupNorm(num_groups=groups, num_channels=in_channels)
        else:
            self.norm = nn.BatchNorm3d(in_channels)
        
        # Expansion
        self.conv2 = nn.Conv3d(in_channels, in_channels * exp_r, kernel_size=1)
        self.act = nn.GELU()
        self.conv3 = nn.Conv3d(in_channels * exp_r, out_channels, kernel_size=1)
        
    def forward(self, x):
        residual = x
        
        x = self.conv1(x)
        x = self.norm(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)
        
        if self.do_res:
            x = x + residual
            
        return x


class MedNeXtDownBlock(nn.Module):
    """Downsampling block for MedNeXt encoder."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        exp_r: int = 4,
        kernel_size: int = 7,
        do_res: bool = True,
        norm_type: str = 'group',
    ):
        super().__init__()
        
        self.conv_res = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None
        
        self.conv1 = nn.Conv3d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size // 2,
            groups=in_channels,
        )
        
        groups = max(1, in_channels // 4)
        self.norm = nn.GroupNorm(num_groups=groups, num_channels=in_channels)
        self.conv2 = nn.Conv3d(in_channels, in_channels * exp_r, kernel_size=1)
        self.act = nn.GELU()
        self.conv3 = nn.Conv3d(in_channels * exp_r, out_channels, kernel_size=1)
        
    def forward(self, x):
        res = F.avg_pool3d(x, 2)
        if self.conv_res:
            res = self.conv_res(res)
            
        x = self.conv1(x)
        x = self.norm(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)
        
        return x + res


class MedNeXtUpBlock(nn.Module):
    """Upsampling block for MedNeXt decoder."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        exp_r: int = 4,
        kernel_size: int = 7,
        do_res: bool = True,
        norm_type: str = 'group',
    ):
        super().__init__()
        
        self.conv_res = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None
        
        self.conv1 = nn.ConvTranspose3d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size // 2,
            output_padding=1,
            groups=in_channels,
        )
        
        groups = max(1, in_channels // 4)
        self.norm = nn.GroupNorm(num_groups=groups, num_channels=in_channels)
        self.conv2 = nn.Conv3d(in_channels, in_channels * exp_r, kernel_size=1)
        self.act = nn.GELU()
        self.conv3 = nn.Conv3d(in_channels * exp_r, out_channels, kernel_size=1)
        
    def forward(self, x):
        res = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)
        if self.conv_res:
            res = self.conv_res(res)
            
        x = self.conv1(x)
        x = self.norm(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)
        
        return x + res


class MedNeXt(nn.Module):
    """
    MedNeXt: Transformer-inspired architecture for 3D medical segmentation.
    Simplified version implementing core features.
    """
    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 4,
        model_id: str = 'B',  # S, B, M, L
        kernel_size: int = 3,
        deep_supervision: bool = False,
    ):
        super().__init__()
        
        self.deep_supervision = deep_supervision
        
        # Model configurations
        configs = {
            'S': {'channels': [32, 64, 128, 256, 512], 'exp_r': [2, 3, 4, 4, 4]},
            'B': {'channels': [32, 64, 128, 256, 512], 'exp_r': [2, 3, 4, 4, 4]},
            'M': {'channels': [32, 64, 128, 256, 512], 'exp_r': [2, 3, 4, 4, 4]},
            'L': {'channels': [32, 64, 128, 256, 512], 'exp_r': [3, 4, 8, 8, 8]},
        }
        
        config = configs.get(model_id, configs['B'])
        channels = config['channels']
        exp_r = config['exp_r']
        
        # Stem
        self.stem = nn.Conv3d(in_channels, channels[0], kernel_size=1)
        
        # Encoder
        self.enc_block_0 = MedNeXtBlock(channels[0], channels[0], exp_r[0], kernel_size)
        self.down_0 = MedNeXtDownBlock(channels[0], channels[1], exp_r[1], kernel_size)
        
        self.enc_block_1 = MedNeXtBlock(channels[1], channels[1], exp_r[1], kernel_size)
        self.down_1 = MedNeXtDownBlock(channels[1], channels[2], exp_r[2], kernel_size)
        
        self.enc_block_2 = MedNeXtBlock(channels[2], channels[2], exp_r[2], kernel_size)
        self.down_2 = MedNeXtDownBlock(channels[2], channels[3], exp_r[3], kernel_size)
        
        self.enc_block_3 = MedNeXtBlock(channels[3], channels[3], exp_r[3], kernel_size)
        self.down_3 = MedNeXtDownBlock(channels[3], channels[4], exp_r[4], kernel_size)
        
        # Bottleneck
        self.bottleneck = MedNeXtBlock(channels[4], channels[4], exp_r[4], kernel_size)
        
        # Decoder
        self.up_3 = MedNeXtUpBlock(channels[4], channels[3], exp_r[3], kernel_size)
        self.dec_block_3 = MedNeXtBlock(channels[3] * 2, channels[3], exp_r[3], kernel_size, do_res=False)
        
        self.up_2 = MedNeXtUpBlock(channels[3], channels[2], exp_r[2], kernel_size)
        self.dec_block_2 = MedNeXtBlock(channels[2] * 2, channels[2], exp_r[2], kernel_size, do_res=False)
        
        self.up_1 = MedNeXtUpBlock(channels[2], channels[1], exp_r[1], kernel_size)
        self.dec_block_1 = MedNeXtBlock(channels[1] * 2, channels[1], exp_r[1], kernel_size, do_res=False)
        
        self.up_0 = MedNeXtUpBlock(channels[1], channels[0], exp_r[0], kernel_size)
        self.dec_block_0 = MedNeXtBlock(channels[0] * 2, channels[0], exp_r[0], kernel_size, do_res=False)
        
        # Output
        self.out = nn.Conv3d(channels[0], num_classes, kernel_size=1)
        
        # Deep supervision outputs
        if deep_supervision:
            self.out_1 = nn.Conv3d(channels[1], num_classes, kernel_size=1)
            self.out_2 = nn.Conv3d(channels[2], num_classes, kernel_size=1)
            self.out_3 = nn.Conv3d(channels[3], num_classes, kernel_size=1)
        
    def forward(self, x):
        # Stem
        x = self.stem(x)
        
        # Encoder
        e0 = self.enc_block_0(x)
        e1 = self.enc_block_1(self.down_0(e0))
        e2 = self.enc_block_2(self.down_1(e1))
        e3 = self.enc_block_3(self.down_2(e2))
        
        # Bottleneck
        b = self.bottleneck(self.down_3(e3))
        
        # Decoder with skip connections
        d3 = self.up_3(b)
        d3 = self.dec_block_3(torch.cat([d3, e3], dim=1))
        
        d2 = self.up_2(d3)
        d2 = self.dec_block_2(torch.cat([d2, e2], dim=1))
        
        d1 = self.up_1(d2)
        d1 = self.dec_block_1(torch.cat([d1, e1], dim=1))
        
        d0 = self.up_0(d1)
        d0 = self.dec_block_0(torch.cat([d0, e0], dim=1))
        
        out = self.out(d0)
        
        if self.deep_supervision and self.training:
            out_1 = self.out_1(d1)
            out_2 = self.out_2(d2)
            out_3 = self.out_3(d3)
            return [out, out_1, out_2, out_3]
        
        return out


def create_mednext_v1(
    num_input_channels: int = 4,
    num_classes: int = 4,
    model_id: str = 'B',
    kernel_size: int = 3,
    deep_supervision: bool = False,
) -> MedNeXt:
    """Create MedNeXt model."""
    return MedNeXt(
        in_channels=num_input_channels,
        num_classes=num_classes,
        model_id=model_id,
        kernel_size=kernel_size,
        deep_supervision=deep_supervision,
    )


class MedNeXtConfig:
    """Configuration for MedNeXt inference."""
    MODEL_SIZE = 'B'
    KERNEL_SIZE = 3
    IN_CHANNELS = 4
    NUM_CLASSES = 4
    PATCH_SIZE = (128, 128, 128)
    USE_AMP = True
    DEVICE = None


def resolve_device(preferred: Optional[str] = None) -> torch.device:
    """Resolve a safe device (handles unusable CUDA on some RTX setups)."""
    if preferred:
        return torch.device(preferred)

    if torch.cuda.is_available():
        try:
            _ = torch.empty(1, device='cuda')
            return torch.device('cuda')
        except Exception as e:
            print(f"⚠️ CUDA available but unusable, falling back to CPU: {e}")

    return torch.device('cpu')


MedNeXtConfig.DEVICE = resolve_device()


def robust_zscore_normalize(data: np.ndarray) -> np.ndarray:
    """
    Robust Z-Score Normalization matching MedNeXt training preprocessing:
    1. Clip outliers (0.5 - 99.5 percentile)
    2. Normalize only non-zero (brain) region
    """
    mask = data > 0
    if mask.sum() == 0:
        return data
    
    # Get brain pixels only
    pixels = data[mask]
    
    # Clip outliers using percentiles
    p_low, p_high = np.percentile(pixels, 0.5), np.percentile(pixels, 99.5)
    data = np.clip(data, p_low, p_high)
    
    # Recalculate stats after clipping
    pixels = data[mask]
    mean, std = pixels.mean(), pixels.std()
    
    # Normalize
    data = (data - mean) / (std + 1e-8)
    data[~mask] = 0
    
    return data


def sliding_window_inference(
    model: nn.Module,
    volume: torch.Tensor,
    config: MedNeXtConfig,
    patch_size: Tuple[int, int, int] = (128, 128, 128),
    overlap: float = 0.5,
) -> np.ndarray:
    """
    Perform sliding window inference for large 3D volumes.
    
    Args:
        model: Trained MedNeXt model
        volume: Input volume tensor (C, D, H, W)
        config: MedNeXtConfig object
        patch_size: Size of patches for inference
        overlap: Overlap ratio between patches
    
    Returns:
        Predicted segmentation mask
    """
    model.eval()
    device = config.DEVICE
    
    C, D, H, W = volume.shape
    pd, ph, pw = patch_size
    
    # Calculate stride
    stride_d = int(pd * (1 - overlap))
    stride_h = int(ph * (1 - overlap))
    stride_w = int(pw * (1 - overlap))
    
    # Pad volume if necessary
    pad_d = max(0, pd - D)
    pad_h = max(0, ph - H)
    pad_w = max(0, pw - W)
    
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        volume = F.pad(volume, (0, pad_w, 0, pad_h, 0, pad_d))
        D, H, W = volume.shape[1:]
    
    # Initialize output and count tensors
    output = torch.zeros((config.NUM_CLASSES, D, H, W), device=device)
    count = torch.zeros((D, H, W), device=device)
    
    # Generate patch positions
    d_positions = list(range(0, max(1, D - pd + 1), stride_d))
    h_positions = list(range(0, max(1, H - ph + 1), stride_h))
    w_positions = list(range(0, max(1, W - pw + 1), stride_w))
    
    # Ensure we cover the entire volume
    if D > pd and D - pd not in d_positions:
        d_positions.append(D - pd)
    if H > ph and H - ph not in h_positions:
        h_positions.append(H - ph)
    if W > pw and W - pw not in w_positions:
        w_positions.append(W - pw)
    
    with torch.no_grad():
        for d_start in d_positions:
            for h_start in h_positions:
                for w_start in w_positions:
                    # Extract patch
                    patch = volume[:, d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw]
                    patch = patch.unsqueeze(0).to(device)
                    
                    # Forward pass
                    device_type = device.type
                    use_amp = config.USE_AMP and device_type == 'cuda'
                    with torch.amp.autocast(device_type, enabled=use_amp):
                        pred = model(patch)
                    
                    # Handle deep supervision output
                    if isinstance(pred, (list, tuple)):
                        pred = pred[0]
                    
                    pred = F.softmax(pred, dim=1).squeeze(0)
                    
                    # Accumulate predictions
                    output[:, d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw] += pred
                    count[d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw] += 1
    
    # Average predictions
    output = output / count.unsqueeze(0).clamp(min=1)
    
    # Remove padding
    original_d = D - pad_d if pad_d > 0 else D
    original_h = H - pad_h if pad_h > 0 else H
    original_w = W - pad_w if pad_w > 0 else W
    output = output[:, :original_d, :original_h, :original_w]
    
    # Get final prediction
    prediction = torch.argmax(output, dim=0).cpu().numpy()
    
    return prediction


def load_mednext_model(checkpoint_path: str, config: MedNeXtConfig = None) -> nn.Module:
    """
    Load MedNeXt model from checkpoint.
    
    Args:
        checkpoint_path: Path to the model checkpoint
        config: MedNeXtConfig object
    
    Returns:
        Loaded model in evaluation mode
    """
    if config is None:
        config = MedNeXtConfig()
    
    print(f"Loading MedNeXt model from {checkpoint_path}...")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE, weights_only=False)
    
    # Get state dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Check if deep supervision was used during training
    has_deep_supervision = any('out_1' in key or 'out_2' in key or 'out_3' in key for key in state_dict.keys())
    
    # Prefer the official nnunet_mednext implementation if available
    model = None
    try:
        import os
        import sys
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'mednext'))
        if os.path.exists(repo_root) and repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from nnunet_mednext import create_mednext_v1 as create_mednext_v1_ref
        model = create_mednext_v1_ref(
            num_input_channels=config.IN_CHANNELS,
            num_classes=config.NUM_CLASSES,
            model_id=config.MODEL_SIZE,
            kernel_size=config.KERNEL_SIZE,
            deep_supervision=has_deep_supervision,
        )
        print("✅ Using nnunet_mednext model implementation")
    except Exception as e:
        print(f"⚠️ Falling back to local MedNeXt implementation: {e}")
        model = create_mednext_v1(
            num_input_channels=config.IN_CHANNELS,
            num_classes=config.NUM_CLASSES,
            model_id=config.MODEL_SIZE,
            kernel_size=config.KERNEL_SIZE,
            deep_supervision=has_deep_supervision,
        )
    
    # Load weights (prefer strict to catch mismatches)
    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception as e:
        print(f"⚠️ Strict load failed, trying strict=False: {e}")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"Missing keys: {len(missing)}")
            print("  " + "\n  ".join(missing[:20]))
        if unexpected:
            print(f"Unexpected keys: {len(unexpected)}")
            print("  " + "\n  ".join(unexpected[:20]))
    
    model.to(config.DEVICE)
    model.eval()
    
    print(f"✅ MedNeXt model loaded successfully!")
    
    if isinstance(checkpoint, dict) and 'metrics' in checkpoint:
        mean_dice = checkpoint['metrics'].get('Mean', 0)
        print(f"   Best Validation Dice: {mean_dice:.4f}")
    
    return model
