"""
MapTR Perception Transformer with BEV-Satellite Fusion
Based on: "Complementing Onboard Sensors with Satellite Map: A New Perspective for HD Map Construction"
"""

import torch
import torch.nn as nn
from mmcv.cnn import build_conv_layer
from mmcv.runner import auto_fp16
from mmdet.models.utils.builder import TRANSFORMER

from projects.mmdet3d_plugin.bevformer.modules.transformer import PerceptionTransformer
from mmcv.cnn import build_model_from_cfg
from mmdet.models import HEADS


@TRANSFORMER.register_module()
class MapTRPerceptionTransformer(PerceptionTransformer):
    """
    Extended Perception Transformer for HD Map Construction with satellite map fusion.
    
    This transformer extends the base PerceptionTransformer by adding a fusion module
    that combines BEV features from onboard sensors with satellite map features.
    
    Args:
        fusion (dict): Config for the fusion module
        All other args are inherited from PerceptionTransformer
    """
    
    def __init__(self,
                 num_feature_levels=4,
                 num_cams=6,
                 two_stage_num_proposals=300,
                 encoder=None,
                 decoder=None,
                 embed_dims=256,
                 rotate_prev_bev=True,
                 use_shift=True,
                 use_can_bus=True,
                 can_bus_norm=True,
                 use_cams_embeds=True,
                 rotate_center=[100, 100],
                 fusion=None,
                 **kwargs):
        
        # Initialize parent class
        super(MapTRPerceptionTransformer, self).__init__(
            num_feature_levels=num_feature_levels,
            num_cams=num_cams,
            two_stage_num_proposals=two_stage_num_proposals,
            encoder=encoder,
            decoder=decoder,
            embed_dims=embed_dims,
            rotate_prev_bev=rotate_prev_bev,
            use_shift=use_shift,
            use_can_bus=use_can_bus,
            can_bus_norm=can_bus_norm,
            use_cams_embeds=use_cams_embeds,
            rotate_center=rotate_center,
            **kwargs
        )
        
        # Build fusion module
        self.fusion = None
        if fusion is not None:
            self.fusion = build_model_from_cfg(fusion, HEADS)
    
    @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'prev_bev', 'bev_pos', 'satellite_feats'))
    def get_bev_features(
            self,
            mlvl_feats,
            bev_queries,
            bev_h,
            bev_w,
            grid_length=[0.512, 0.512],
            bev_pos=None,
            prev_bev=None,
            satellite_feats=None,
            **kwargs):
        """
        Obtain BEV features with optional satellite map fusion.
        
        Args:
            mlvl_feats: Multi-level image features
            bev_queries: BEV query embeddings
            bev_h: Height of BEV grid
            bev_w: Width of BEV grid
            grid_length: Grid resolution
            bev_pos: BEV positional encoding
            prev_bev: Previous BEV features (for temporal modeling)
            satellite_feats: Satellite map features (optional)
            **kwargs: Additional arguments
        
        Returns:
            Fused BEV features
        """
        # Get BEV features from parent class
        bev_embed = super().get_bev_features(
            mlvl_feats=mlvl_feats,
            bev_queries=bev_queries,
            bev_h=bev_h,
            bev_w=bev_w,
            grid_length=grid_length,
            bev_pos=bev_pos,
            prev_bev=prev_bev,
            **kwargs
        )  # Shape: (bs, bev_h*bev_w, embed_dims)
        
        # Apply fusion if satellite features are provided and fusion module exists
        if self.fusion is not None and satellite_feats is not None:
            bs = bev_embed.shape[1] if bev_embed.dim() == 3 else bev_embed.shape[0]
            
            # Reshape BEV embed from (num_query, bs, embed_dims) or (bs, num_query, embed_dims) 
            # to (bs, embed_dims, bev_h, bev_w)
            if bev_embed.dim() == 3 and bev_embed.shape[0] == bev_h * bev_w:
                # Shape: (bev_h*bev_w, bs, embed_dims) -> (bs, embed_dims, bev_h, bev_w)
                bev_embed_spatial = bev_embed.permute(1, 2, 0).reshape(bs, -1, bev_h, bev_w)
            elif bev_embed.dim() == 3 and bev_embed.shape[1] == bev_h * bev_w:
                # Shape: (bs, bev_h*bev_w, embed_dims) -> (bs, embed_dims, bev_h, bev_w)
                bev_embed_spatial = bev_embed.permute(0, 2, 1).reshape(bs, -1, bev_h, bev_w)
            else:
                # Already in spatial format
                bev_embed_spatial = bev_embed
            
            # Apply fusion
            fused_bev = self.fusion(bev_embed_spatial, satellite_feats)
            
            # Reshape back to original format (num_query, bs, embed_dims)
            if bev_embed.dim() == 3 and bev_embed.shape[0] == bev_h * bev_w:
                bev_embed = fused_bev.flatten(2).permute(2, 0, 1)
            elif bev_embed.dim() == 3:
                bev_embed = fused_bev.flatten(2).permute(0, 2, 1)
            else:
                bev_embed = fused_bev
        
        return bev_embed
    
    @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'object_query_embed', 'prev_bev', 'bev_pos', 'satellite_feats'))
    def forward(self,
                mlvl_feats,
                bev_queries,
                object_query_embed,
                bev_h,
                bev_w,
                grid_length=[0.512, 0.512],
                bev_pos=None,
                reg_branches=None,
                cls_branches=None,
                prev_bev=None,
                satellite_feats=None,
                **kwargs):
        """
        Forward function with satellite map fusion support.
        
        Args:
            mlvl_feats: Multi-level image features
            bev_queries: BEV query embeddings
            object_query_embed: Object query embeddings for decoder
            bev_h: Height of BEV grid
            bev_w: Width of BEV grid
            grid_length: Grid resolution
            bev_pos: BEV positional encoding
            reg_branches: Regression branches (for box refinement)
            cls_branches: Classification branches
            prev_bev: Previous BEV features (for temporal modeling)
            satellite_feats: Satellite map features (optional)
            **kwargs: Additional arguments
        
        Returns:
            tuple: (bev_embed, inter_states, init_reference_out, inter_references_out)
        """
        # Get fused BEV features
        bev_embed = self.get_bev_features(
            mlvl_feats,
            bev_queries,
            bev_h,
            bev_w,
            grid_length=grid_length,
            bev_pos=bev_pos,
            prev_bev=prev_bev,
            satellite_feats=satellite_feats,
            **kwargs
        )  # bev_embed shape: (num_query, bs, embed_dims) or (bs, bev_h*bev_w, embed_dims)
        
        # Prepare decoder inputs
        bs = mlvl_feats[0].size(0)
        query_pos, query = torch.split(
            object_query_embed, self.embed_dims, dim=1)
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        query = query.unsqueeze(0).expand(bs, -1, -1)
        
        # Generate reference points
        reference_points = self.reference_points(query_pos)
        reference_points = reference_points.sigmoid()
        init_reference_out = reference_points
        
        # Permute for decoder input
        query = query.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        
        # Ensure bev_embed has correct shape for decoder
        if bev_embed.dim() == 3 and bev_embed.shape[1] == bs:
            # Shape is (num_query, bs, embed_dims) - correct format
            bev_embed_for_decoder = bev_embed
        elif bev_embed.dim() == 3 and bev_embed.shape[0] == bs:
            # Shape is (bs, num_query, embed_dims) - need to permute
            bev_embed_for_decoder = bev_embed.permute(1, 0, 2)
        else:
            bev_embed_for_decoder = bev_embed
        
        # Decoder forward
        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=bev_embed_for_decoder,
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=reg_branches,
            cls_branches=cls_branches,
            spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device),
            level_start_index=torch.tensor([0], device=query.device),
            **kwargs
        )
        
        inter_references_out = inter_references
        
        return bev_embed, inter_states, init_reference_out, inter_references_out
