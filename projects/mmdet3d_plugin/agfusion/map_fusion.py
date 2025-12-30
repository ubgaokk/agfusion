"""
MapFusion Module - Fuses BEV features with satellite map features
Based on: "Complementing Onboard Sensors with Satellite Map: A New Perspective for HD Map Construction"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.runner import BaseModule, auto_fp16
from mmdet.models import HEADS


@HEADS.register_module()
class MapFusion(BaseModule):
    """
    Fusion module that combines BEV features from onboard sensors with satellite map features.
    
    This module implements the fusion strategy from the paper:
    "Complementing Onboard Sensors with Satellite Map: A New Perspective for HD Map Construction"
    
    Args:
        in_channels (list[int]): Input channels for each feature stream [bev_channels, satellite_channels]
        out_channels (int): Output channels after fusion
        num_levels (int): Number of feature pyramid levels
        fusion_type (str): Fusion strategy - 'concat', 'add', 'attention', or 'adaptive'
        norm_cfg (dict): Config for normalization layer
        act_cfg (dict): Config for activation layer
    """
    
    def __init__(self,
                 in_channels=[256, 256],
                 out_channels=256,
                 num_levels=1,
                 fusion_type='concat',
                 norm_cfg=dict(type='BN2d'),
                 act_cfg=dict(type='ReLU', inplace=True),
                 init_cfg=None):
        super(MapFusion, self).__init__(init_cfg=init_cfg)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_levels = num_levels
        self.fusion_type = fusion_type
        
        # Build fusion layers based on fusion type
        if fusion_type == 'concat':
            # Simple concatenation followed by 1x1 conv
            self.fusion_conv = nn.Sequential(
                nn.Conv2d(sum(in_channels), out_channels, kernel_size=1, bias=False),
                build_norm_layer(norm_cfg, out_channels)[1],
                nn.ReLU(inplace=True)
            )
            
        elif fusion_type == 'add':
            # Element-wise addition (requires same channels)
            assert in_channels[0] == in_channels[1], \
                "For 'add' fusion, both inputs must have same channels"
            if in_channels[0] != out_channels:
                self.projection = nn.Conv2d(in_channels[0], out_channels, 
                                          kernel_size=1, bias=False)
            else:
                self.projection = nn.Identity()
                
        elif fusion_type == 'attention':
            # Cross-attention based fusion
            self.bev_proj = nn.Conv2d(in_channels[0], out_channels, 1)
            self.satellite_proj = nn.Conv2d(in_channels[1], out_channels, 1)
            
            # Attention weights
            self.attention_conv = nn.Sequential(
                nn.Conv2d(out_channels * 2, out_channels, 1),
                nn.Sigmoid()
            )
            
            self.fusion_conv = nn.Sequential(
                nn.Conv2d(out_channels * 2, out_channels, 1, bias=False),
                build_norm_layer(norm_cfg, out_channels)[1],
                nn.ReLU(inplace=True)
            )
            
        elif fusion_type == 'adaptive':
            # Adaptive fusion with learnable weights
            self.bev_conv = nn.Sequential(
                nn.Conv2d(in_channels[0], out_channels, 1, bias=False),
                build_norm_layer(norm_cfg, out_channels)[1],
                nn.ReLU(inplace=True)
            )
            
            self.satellite_conv = nn.Sequential(
                nn.Conv2d(in_channels[1], out_channels, 1, bias=False),
                build_norm_layer(norm_cfg, out_channels)[1],
                nn.ReLU(inplace=True)
            )
            
            # Learnable fusion weights
            self.weight_conv = nn.Sequential(
                nn.Conv2d(out_channels * 2, 2, 1),
                nn.Softmax(dim=1)
            )
            
        else:
            raise ValueError(f"Unsupported fusion_type: {fusion_type}")
    
    @auto_fp16()
    def forward(self, bev_features, satellite_features=None):
        """
        Forward function for MapFusion.
        
        Args:
            bev_features (Tensor): BEV features from onboard sensors
                Shape: (B, C1, H, W)
            satellite_features (Tensor, optional): Satellite map features
                Shape: (B, C2, H, W)
                If None, returns processed bev_features only
        
        Returns:
            Tensor: Fused features with shape (B, out_channels, H, W)
        """
        # If no satellite features provided, just process BEV features
        if satellite_features is None:
            if self.fusion_type in ['concat', 'add']:
                return bev_features
            elif self.fusion_type == 'attention':
                return self.bev_proj(bev_features)
            elif self.fusion_type == 'adaptive':
                return self.bev_conv(bev_features)
        
        # Ensure spatial dimensions match
        if bev_features.shape[-2:] != satellite_features.shape[-2:]:
            satellite_features = F.interpolate(
                satellite_features,
                size=bev_features.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
        
        # Apply fusion strategy
        if self.fusion_type == 'concat':
            # Concatenate along channel dimension
            fused = torch.cat([bev_features, satellite_features], dim=1)
            fused = self.fusion_conv(fused)
            
        elif self.fusion_type == 'add':
            # Element-wise addition
            fused = bev_features + satellite_features
            fused = self.projection(fused)
            
        elif self.fusion_type == 'attention':
            # Project features
            bev_proj = self.bev_proj(bev_features)
            sat_proj = self.satellite_proj(satellite_features)
            
            # Compute attention weights
            concat_feat = torch.cat([bev_proj, sat_proj], dim=1)
            attention = self.attention_conv(concat_feat)
            
            # Apply attention
            attended_bev = bev_proj * attention
            attended_sat = sat_proj * (1 - attention)
            
            # Fuse attended features
            fused = torch.cat([attended_bev, attended_sat], dim=1)
            fused = self.fusion_conv(fused)
            
        elif self.fusion_type == 'adaptive':
            # Process each stream
            bev_feat = self.bev_conv(bev_features)
            sat_feat = self.satellite_conv(satellite_features)
            
            # Compute adaptive weights
            concat_feat = torch.cat([bev_feat, sat_feat], dim=1)
            weights = self.weight_conv(concat_feat)
            
            # Weighted fusion
            fused = bev_feat * weights[:, 0:1] + sat_feat * weights[:, 1:2]
        
        return fused
    
    def init_weights(self):
        """Initialize weights of the fusion module."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
