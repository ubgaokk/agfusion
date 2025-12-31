"""
Satellite Feature Extractor with U-Net
Extracts features from satellite imagery for fusion with BEV features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule, auto_fp16
from mmdet.models import BACKBONES

from .encoder import UNet


@BACKBONES.register_module()
class SatelliteFeatureExtractor(BaseModule):
    """
    Satellite feature extractor using U-Net architecture.
    
    Processes satellite imagery and extracts multi-scale features
    that can be fused with BEV features from onboard sensors.
    
    Args:
        in_channels (int): Number of input channels (3 for RGB satellite images)
        out_channels (int): Number of output feature channels (should match BEV features)
        base_channels (int): Base number of channels in U-Net
        num_layers (int): Number of encoder/decoder layers
        bilinear (bool): Use bilinear upsampling instead of transposed conv
        target_size (tuple): Target output size (H, W) to match BEV grid
        interpolate_mode (str): Interpolation mode for resizing ('bilinear' or 'nearest')
        init_cfg (dict): Initialization config
    """
    
    def __init__(
        self,
        in_channels=3,
        out_channels=256,
        base_channels=64,
        num_layers=4,
        bilinear=True,
        target_size=None,
        interpolate_mode='bilinear',
        init_cfg=None
    ):
        super(SatelliteFeatureExtractor, self).__init__(init_cfg=init_cfg)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.target_size = target_size
        self.interpolate_mode = interpolate_mode
        self.fp16_enabled = False
        
        # U-Net encoder-decoder
        self.unet = UNet(
            in_channels=in_channels,
            base_channels=base_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            bilinear=bilinear
        )
        
        # Optional: Additional refinement layers
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1)
        )
    
    def init_weights(self):
        """Initialize weights of the feature extractor."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    @auto_fp16()
    def forward(self, satellite_images):
        """
        Extract features from satellite images.
        
        Args:
            satellite_images (Tensor): Satellite images
                Shape: (B, C, H, W) or (B, N, C, H, W) for multi-view
        
        Returns:
            Tensor: Extracted satellite features
                Shape: (B, out_channels, H', W') where (H', W') matches target_size or BEV grid
        """
        # Handle multi-view input
        if satellite_images.dim() == 5:
            B, N, C, H, W = satellite_images.shape
            # For satellite, we typically use a single overhead view
            # Take the first view or average multiple views
            satellite_images = satellite_images[:, 0]  # (B, C, H, W)
        
        # Extract features using U-Net
        features = self.unet(satellite_images)
        
        # Refine features
        features = self.refine(features)
        
        # Resize to match target BEV grid size if specified
        if self.target_size is not None:
            features = F.interpolate(
                features,
                size=self.target_size,
                mode=self.interpolate_mode,
                align_corners=False if self.interpolate_mode == 'bilinear' else None
            )
        
        return features


@BACKBONES.register_module()
class MultiScaleSatelliteEncoder(BaseModule):
    """
    Multi-scale satellite feature encoder with feature pyramid.
    
    Extracts features at multiple scales from satellite imagery
    for better alignment with BEV features at different resolutions.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (list[int]): Output channels for each scale
        base_channels (int): Base number of channels in U-Net
        num_layers (int): Number of encoder/decoder layers
        scales (list[float]): Output scales relative to input size
    """
    
    def __init__(
        self,
        in_channels=3,
        out_channels=[64, 128, 256],
        base_channels=64,
        num_layers=4,
        scales=[0.25, 0.5, 1.0],
        init_cfg=None
    ):
        super(MultiScaleSatelliteEncoder, self).__init__(init_cfg=init_cfg)
        
        self.in_channels = in_channels
        self.out_channels_list = out_channels
        self.scales = scales
        self.fp16_enabled = False
        
        # Shared U-Net backbone
        self.backbone = UNet(
            in_channels=in_channels,
            base_channels=base_channels,
            out_channels=base_channels * 4,
            num_layers=num_layers,
            bilinear=True
        )
        
        # Feature projection for each scale
        self.projections = nn.ModuleList()
        for out_ch in out_channels:
            self.projections.append(
                nn.Sequential(
                    nn.Conv2d(base_channels * 4, out_ch, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, kernel_size=1)
                )
            )
    
    @auto_fp16()
    def forward(self, satellite_images):
        """
        Extract multi-scale features from satellite images.
        
        Args:
            satellite_images (Tensor): Satellite images (B, C, H, W)
        
        Returns:
            list[Tensor]: Multi-scale features
        """
        # Extract base features
        base_features = self.backbone(satellite_images)
        
        # Generate multi-scale features
        multi_scale_features = []
        for scale, projection in zip(self.scales, self.projections):
            if scale != 1.0:
                # Resize base features
                scaled_size = (
                    int(base_features.shape[2] * scale),
                    int(base_features.shape[3] * scale)
                )
                scaled_features = F.interpolate(
                    base_features,
                    size=scaled_size,
                    mode='bilinear',
                    align_corners=False
                )
            else:
                scaled_features = base_features
            
            # Project to desired output channels
            projected_features = projection(scaled_features)
            multi_scale_features.append(projected_features)
        
        return multi_scale_features


@BACKBONES.register_module()
class AlignedSatelliteEncoder(BaseModule):
    """
    Satellite encoder with automatic alignment to BEV coordinate system.
    
    Handles coordinate transformation and alignment between satellite
    imagery and BEV grid from onboard sensors.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output feature channels
        bev_h (int): Height of BEV grid
        bev_w (int): Width of BEV grid
        base_channels (int): Base channels in U-Net
        use_geo_transform (bool): Whether to apply geometric transformation
    """
    
    def __init__(
        self,
        in_channels=3,
        out_channels=256,
        bev_h=200,
        bev_w=100,
        base_channels=64,
        use_geo_transform=False,
        init_cfg=None
    ):
        super(AlignedSatelliteEncoder, self).__init__(init_cfg=init_cfg)
        
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.use_geo_transform = use_geo_transform
        self.fp16_enabled = False
        
        # Feature extractor
        self.feature_extractor = SatelliteFeatureExtractor(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            target_size=(bev_h, bev_w),
            interpolate_mode='bilinear'
        )
        
        # Optional: Learnable alignment module
        if use_geo_transform:
            self.alignment = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            )
    
    @auto_fp16()
    def forward(self, satellite_images, ego_pose=None):
        """
        Extract and align satellite features to BEV grid.
        
        Args:
            satellite_images (Tensor): Satellite images (B, C, H, W)
            ego_pose (Tensor, optional): Ego vehicle pose for alignment
        
        Returns:
            Tensor: Aligned satellite features (B, out_channels, bev_h, bev_w)
        """
        # Extract features
        features = self.feature_extractor(satellite_images)
        
        # Apply geometric transformation if enabled
        if self.use_geo_transform and ego_pose is not None:
            features = self.alignment(features)
        
        return features
