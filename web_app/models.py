"""
Brain Tumor Segmentation Models - U-Net, SegResNet, and MedNeXt architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

# Available models for selection
AVAILABLE_MODELS = {
    "unet": {
        "name": "U-Net",
        "description": "Classic 3D U-Net architecture",
        "icon": "🔬"
    },
    "segresnet": {
        "name": "SegResNet",
        "description": "Residual encoder-decoder network",
        "icon": "🧬"
    },
    "mednext": {
        "name": "MedNeXt",
        "description": "Transformer-inspired ConvNet (sliding window)",
        "icon": "🧠"
    }
}


class DoubleConv(nn.Module):
    """Double convolution block: (Conv3D -> InstanceNorm -> LeakyReLU) x 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downsampling with MaxPool followed by DoubleConv."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upsampling followed by DoubleConv with skip connection."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet3D(nn.Module):
    """3D U-Net for volumetric segmentation."""
    def __init__(self, in_channels=4, out_channels=4, features=32):
        super().__init__()
        
        # Encoder
        self.inc = DoubleConv(in_channels, features)
        self.down1 = Down(features, features * 2)
        self.down2 = Down(features * 2, features * 4)
        self.down3 = Down(features * 4, features * 8)
        self.down4 = Down(features * 8, features * 16)
        
        # Decoder
        self.up1 = Up(features * 16, features * 8)
        self.up2 = Up(features * 8, features * 4)
        self.up3 = Up(features * 4, features * 2)
        self.up4 = Up(features * 2, features)
        
        # Output
        self.outc = nn.Conv3d(features, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        return self.outc(x)


class ResidualBlock(nn.Module):
    """Residual block that learns the difference instead of raw mapping."""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += residual
        x = self.relu(x)
        return x


class SegResNet(nn.Module):
    """SegResNet for volumetric segmentation."""
    def __init__(self, in_channels=4, out_channels=4, init_filters=32):
        super().__init__()
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv3d(in_channels, init_filters, kernel_size=3, padding=1),
            nn.BatchNorm3d(init_filters),
            nn.ReLU(inplace=True)
        )
        self.res_block1 = ResidualBlock(init_filters, init_filters)
        self.res_block2 = ResidualBlock(init_filters, init_filters * 2, stride=2)
        self.res_block3 = ResidualBlock(init_filters * 2, init_filters * 4, stride=2)
        self.res_block4 = ResidualBlock(init_filters * 4, init_filters * 8, stride=2)
        
        # Bottleneck
        self.bottleneck = ResidualBlock(init_filters * 8, init_filters * 16, stride=2)
        
        # Decoder
        self.up4 = nn.ConvTranspose3d(init_filters * 16, init_filters * 8, kernel_size=2, stride=2)
        self.dec4 = ResidualBlock(init_filters * 16, init_filters * 8)
        
        self.up3 = nn.ConvTranspose3d(init_filters * 8, init_filters * 4, kernel_size=2, stride=2)
        self.dec3 = ResidualBlock(init_filters * 8, init_filters * 4)
        
        self.up2 = nn.ConvTranspose3d(init_filters * 4, init_filters * 2, kernel_size=2, stride=2)
        self.dec2 = ResidualBlock(init_filters * 4, init_filters * 2)
        
        self.up1 = nn.ConvTranspose3d(init_filters * 2, init_filters, kernel_size=2, stride=2)
        self.dec1 = ResidualBlock(init_filters * 2, init_filters)
        
        self.final = nn.Conv3d(init_filters, out_channels, kernel_size=1)
        
    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)
        x1 = self.res_block1(x1)
        x2 = self.res_block2(x1)
        x3 = self.res_block3(x2)
        x4 = self.res_block4(x3)
        
        # Bottleneck
        b = self.bottleneck(x4)
        
        # Decoder with skip connections
        u4 = self.up4(b)
        if u4.shape != x4.shape:
            u4 = F.interpolate(u4, size=x4.shape[2:])
        d4 = torch.cat((u4, x4), dim=1)
        d4 = self.dec4(d4)
        
        u3 = self.up3(d4)
        if u3.shape != x3.shape:
            u3 = F.interpolate(u3, size=x3.shape[2:])
        d3 = torch.cat((u3, x3), dim=1)
        d3 = self.dec3(d3)
        
        u2 = self.up2(d3)
        if u2.shape != x2.shape:
            u2 = F.interpolate(u2, size=x2.shape[2:])
        d2 = torch.cat((u2, x2), dim=1)
        d2 = self.dec2(d2)
        
        u1 = self.up1(d2)
        if u1.shape != x1.shape:
            u1 = F.interpolate(u1, size=x1.shape[2:])
        d1 = torch.cat((u1, x1), dim=1)
        d1 = self.dec1(d1)
        
        return self.final(d1)


def load_unet_model(model_path: str, device: torch.device) -> UNet3D:
    """Load the U-Net model from checkpoint"""
    model = UNet3D(in_channels=4, out_channels=4)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
  # Get state dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_segresnet_model(model_path: str, device: torch.device) -> SegResNet:
    """Load the SegResNet model from checkpoint"""
    model = SegResNet(in_channels=4, out_channels=4)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
   # Get state dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
