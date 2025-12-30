"""
MapTR Head for HD Map Construction
Supports vectorized map element detection with points prediction
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear, bias_init_with_prob
from mmcv.utils import TORCH_VERSION, digit_version
from mmcv.runner import force_fp32, auto_fp16

from mmdet.core import (multi_apply, reduce_mean)
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.models import HEADS
from mmdet.models.dense_heads import DETRHead
from mmdet3d.core.bbox.coders import build_bbox_coder
from mmcv.cnn.bricks.transformer import build_positional_encoding
import numpy as np


@HEADS.register_module()
class MapTRHead(DETRHead):
    """
    Head for MapTR - HD Map Construction with Vectorized Representation.
    
    Predicts vectorized map elements (e.g., dividers, crossings, boundaries) as sequences of points.
    
    Args:
        bev_h (int): Height of BEV feature map
        bev_w (int): Width of BEV feature map
        num_query (int): Number of object queries
        num_vec (int): Maximum number of vectors per image
        num_pts_per_vec (int): Number of points per predicted vector
        num_pts_per_gt_vec (int): Number of points per ground truth vector
        dir_interval (int): Interval for direction prediction
        query_embed_type (str): Type of query embedding ('instance_pts' or others)
        transform_method (str): Method for coordinate transformation
        gt_shift_pts_pattern (str): Pattern for ground truth point shifting
        with_box_refine (bool): Whether to refine predictions
        as_two_stage (bool): Whether to use two-stage detection
        transformer (dict): Transformer configuration
        bbox_coder (dict): Bbox coder configuration
        code_size (int): Size of encoded vector representation
        code_weights (list): Weights for each code element
    """
    
    def __init__(self,
                 *args,
                 bev_h=200,
                 bev_w=100,
                 num_query=900,
                 num_vec=50,
                 num_pts_per_vec=20,
                 num_pts_per_gt_vec=20,
                 dir_interval=1,
                 query_embed_type='instance_pts',
                 transform_method='minmax',
                 gt_shift_pts_pattern='v2',
                 with_box_refine=False,
                 as_two_stage=False,
                 transformer=None,
                 bbox_coder=None,
                 num_cls_fcs=2,
                 code_size=2,
                 code_weights=None,
                 **kwargs):
        
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.num_query = num_query
        self.num_vec = num_vec
        self.num_pts_per_vec = num_pts_per_vec
        self.num_pts_per_gt_vec = num_pts_per_gt_vec
        self.dir_interval = dir_interval
        self.query_embed_type = query_embed_type
        self.transform_method = transform_method
        self.gt_shift_pts_pattern = gt_shift_pts_pattern
        self.fp16_enabled = False
        
        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage
        
        self.code_size = code_size
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0] * code_size
        
        if bbox_coder is not None:
            self.bbox_coder = build_bbox_coder(bbox_coder)
            self.pc_range = self.bbox_coder.pc_range
        else:
            self.pc_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
        
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.num_cls_fcs = num_cls_fcs - 1
        
        super(MapTRHead, self).__init__(
            *args, transformer=transformer, **kwargs)
        
        self.code_weights = nn.Parameter(torch.tensor(
            self.code_weights, requires_grad=False), requires_grad=False)
    
    def _init_layers(self):
        """Initialize classification, regression and points prediction branches."""
        # Classification branch
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)
        
        # Regression branch (for bounding box if needed)
        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)
        
        # Points prediction branch (for vectorized map elements)
        pts_branch = []
        for _ in range(self.num_reg_fcs):
            pts_branch.append(Linear(self.embed_dims, self.embed_dims))
            pts_branch.append(nn.ReLU())
        pts_branch.append(Linear(self.embed_dims, self.num_pts_per_vec * 2))  # x, y for each point
        pts_branch = nn.Sequential(*pts_branch)
        
        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
        
        # Number of prediction layers
        num_pred = (self.transformer.decoder.num_layers + 1) if \
            self.as_two_stage else self.transformer.decoder.num_layers
        
        if self.with_box_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
            self.pts_branches = _get_clones(pts_branch, num_pred)
        else:
            self.cls_branches = nn.ModuleList([fc_cls for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList([reg_branch for _ in range(num_pred)])
            self.pts_branches = nn.ModuleList([pts_branch for _ in range(num_pred)])
        
        if not self.as_two_stage:
            # BEV embedding
            self.bev_embedding = nn.Embedding(
                self.bev_h * self.bev_w, self.embed_dims)
            # Query embedding for map elements
            self.query_embedding = nn.Embedding(self.num_query, self.embed_dims * 2)
    
    def init_weights(self):
        """Initialize weights of the head."""
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)
    
    @auto_fp16(apply_to=('mlvl_feats'))
    def forward(self, mlvl_feats, img_metas, prev_bev=None, only_bev=False):
        """
        Forward function.
        
        Args:
            mlvl_feats (tuple[Tensor]): Features from upstream network
                Shape: (B, N, C, H, W) where N is number of cameras
            img_metas (list[dict]): Meta information for each image
            prev_bev (Tensor, optional): Previous BEV features for temporal modeling
            only_bev (bool): If True, only return BEV features
        
        Returns:
            dict: Predictions including:
                - all_cls_scores: Classification scores
                - all_bbox_preds: Bounding box predictions
                - all_pts_preds: Points predictions for vectorized elements
        """
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        
        # Get query embeddings
        object_query_embeds = self.query_embedding.weight.to(dtype)
        bev_queries = self.bev_embedding.weight.to(dtype)
        
        # Get positional encoding for BEV
        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                               device=bev_queries.device).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)
        
        # Get BEV features only
        if only_bev:
            return self.transformer.get_bev_features(
                mlvl_feats,
                bev_queries,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
                bev_pos=bev_pos,
                img_metas=img_metas,
                prev_bev=prev_bev,
            )
        
        # Full forward pass through transformer
        outputs = self.transformer(
            mlvl_feats,
            bev_queries,
            object_query_embeds,
            self.bev_h,
            self.bev_w,
            grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
            bev_pos=bev_pos,
            reg_branches=self.reg_branches if self.with_box_refine else None,
            cls_branches=self.cls_branches if self.as_two_stage else None,
            img_metas=img_metas,
            prev_bev=prev_bev
        )
        
        bev_embed, hs, init_reference, inter_references = outputs
        hs = hs.permute(0, 2, 1, 3)  # (num_layers, bs, num_query, embed_dims)
        
        outputs_classes = []
        outputs_coords = []
        outputs_pts_coords = []
        
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            
            reference = inverse_sigmoid(reference)
            
            # Classification
            outputs_class = self.cls_branches[lvl](hs[lvl])
            
            # Bounding box regression (optional, for compatibility)
            tmp = self.reg_branches[lvl](hs[lvl])
            
            # Points prediction for vectorized map elements
            pts_coordinate = self.pts_branches[lvl](hs[lvl])
            pts_coordinate = pts_coordinate.view(bs, self.num_query, self.num_pts_per_vec, 2)
            pts_coordinate = pts_coordinate.sigmoid()
            
            # Denormalize points to real-world coordinates
            pts_coordinate[..., 0:1] = (pts_coordinate[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + 
                                        self.pc_range[0])
            pts_coordinate[..., 1:2] = (pts_coordinate[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + 
                                        self.pc_range[1])
            
            # Process bbox predictions (if used)
            if reference.shape[-1] == 2:
                tmp[..., 0:2] += reference
                tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
                tmp[..., 0:1] = (tmp[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0])
                tmp[..., 1:2] = (tmp[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1])
            
            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
            outputs_pts_coords.append(pts_coordinate)
        
        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)
        outputs_pts_coords = torch.stack(outputs_pts_coords)
        
        outs = {
            'bev_embed': bev_embed,
            'all_cls_scores': outputs_classes,
            'all_bbox_preds': outputs_coords,
            'all_pts_preds': outputs_pts_coords,
            'enc_cls_scores': None,
            'enc_bbox_preds': None,
        }
        
        return outs
    
    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             preds_dicts,
             gt_bboxes_ignore=None,
             img_metas=None):
        """
        Calculate losses.
        
        Args:
            gt_bboxes_list (list[Tensor]): Ground truth bboxes/points
            gt_labels_list (list[Tensor]): Ground truth labels
            preds_dicts (dict): Predictions from forward()
            gt_bboxes_ignore: Ignored gt bboxes
            img_metas: Image meta information
        
        Returns:
            dict: Loss components
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports for gt_bboxes_ignore setting to None.'
        
        all_cls_scores = preds_dicts['all_cls_scores']
        all_bbox_preds = preds_dicts['all_bbox_preds']
        all_pts_preds = preds_dicts['all_pts_preds']
        enc_cls_scores = preds_dicts['enc_cls_scores']
        enc_bbox_preds = preds_dicts['enc_bbox_preds']
        
        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device
        
        # Format ground truth
        if hasattr(gt_bboxes_list[0], 'tensor'):
            # If gt_bboxes are in LiDARInstance format
            gt_bboxes_list = [gt_bboxes.tensor.to(device) for gt_bboxes in gt_bboxes_list]
        
        # Compute losses for each decoder layer
        losses_cls, losses_bbox, losses_iou, losses_pts, losses_dir = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds, all_pts_preds,
            [gt_bboxes_list for _ in range(num_dec_layers)],
            [gt_labels_list for _ in range(num_dec_layers)],
            [gt_bboxes_ignore for _ in range(num_dec_layers)])
        
        loss_dict = dict()
        # Loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]
        loss_dict['loss_iou'] = losses_iou[-1]
        loss_dict['loss_pts'] = losses_pts[-1]
        loss_dict['loss_dir'] = losses_dir[-1]
        
        # Loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i, loss_iou_i, loss_pts_i, loss_dir_i in zip(
                losses_cls[:-1], losses_bbox[:-1], losses_iou[:-1], 
                losses_pts[:-1], losses_dir[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            loss_dict[f'd{num_dec_layer}.loss_iou'] = loss_iou_i
            loss_dict[f'd{num_dec_layer}.loss_pts'] = loss_pts_i
            loss_dict[f'd{num_dec_layer}.loss_dir'] = loss_dir_i
            num_dec_layer += 1
        
        return loss_dict
    
    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    pts_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
        """
        Loss function for a single decoder layer.
        
        Args:
            cls_scores (Tensor): Classification scores (bs, num_query, num_classes)
            bbox_preds (Tensor): Bbox predictions (bs, num_query, code_size)
            pts_preds (Tensor): Points predictions (bs, num_query, num_pts, 2)
            gt_bboxes_list (list[Tensor]): Ground truth bboxes/points for each image
            gt_labels_list (list[Tensor]): Ground truth labels for each image
            gt_bboxes_ignore_list: Ignored ground truth
        
        Returns:
            tuple: losses (loss_cls, loss_bbox, loss_iou, loss_pts, loss_dir)
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        pts_preds_list = [pts_preds[i] for i in range(num_imgs)]
        
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list, pts_preds_list,
                                          gt_bboxes_list, gt_labels_list, 
                                          gt_bboxes_ignore_list)
        
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         pts_targets_list, pts_weights_list, num_total_pos, num_total_neg) = cls_reg_targets
        
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)
        pts_targets = torch.cat(pts_targets_list, 0)
        pts_weights = torch.cat(pts_weights_list, 0)
        
        # Classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        cls_avg_factor = num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)
        
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)
        
        # Bounding box loss (optional, set to 0 if not used)
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        
        bbox_weights = bbox_weights * self.code_weights
        
        loss_bbox = self.loss_bbox(
            bbox_preds[isnotnan, :self.code_size],
            normalized_bbox_targets[isnotnan, :self.code_size],
            bbox_weights[isnotnan, :self.code_size],
            avg_factor=max(num_total_pos, 1))
        
        # IoU loss (optional)
        loss_iou = loss_bbox.new_zeros(1)
        
        # Points loss
        pts_preds = pts_preds.reshape(-1, pts_preds.size(-2), pts_preds.size(-1))
        num_pts = pts_preds.size(1)
        
        normalized_pts_targets = pts_targets.clone()
        normalized_pts_targets[..., 0] = (pts_targets[..., 0] - self.pc_range[0]) / \
                                         (self.pc_range[3] - self.pc_range[0])
        normalized_pts_targets[..., 1] = (pts_targets[..., 1] - self.pc_range[1]) / \
                                         (self.pc_range[4] - self.pc_range[1])
        
        isnotnan_pts = torch.isfinite(normalized_pts_targets).all(dim=-1).all(dim=-1)
        
        loss_pts = self.loss_pts(
            pts_preds[isnotnan_pts],
            normalized_pts_targets[isnotnan_pts],
            pts_weights[isnotnan_pts],
            avg_factor=max(num_total_pos, 1))
        
        # Direction loss (optional)
        loss_dir = loss_bbox.new_zeros(1)
        if hasattr(self, 'loss_dir') and self.loss_dir is not None:
            # Compute direction loss from consecutive points
            pts_diff = pts_preds[:, 1:] - pts_preds[:, :-1]
            pts_dir = F.normalize(pts_diff, p=2, dim=-1)
            
            gt_pts_diff = pts_targets[:, 1:] - pts_targets[:, :-1]
            gt_pts_dir = F.normalize(gt_pts_diff, p=2, dim=-1)
            
            isnotnan_dir = torch.isfinite(gt_pts_dir).all(dim=-1).all(dim=-1)
            loss_dir = self.loss_dir(
                pts_dir[isnotnan_dir],
                gt_pts_dir[isnotnan_dir],
                avg_factor=max(num_total_pos, 1))
        
        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            loss_cls = torch.nan_to_num(loss_cls)
            loss_bbox = torch.nan_to_num(loss_bbox)
            loss_pts = torch.nan_to_num(loss_pts)
            loss_dir = torch.nan_to_num(loss_dir)
        
        return loss_cls, loss_bbox, loss_iou, loss_pts, loss_dir
    
    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    pts_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
        """
        Compute regression and classification targets for all images.
        
        This is a placeholder - actual implementation depends on the matching strategy.
        You may need to implement a custom assigner for vectorized map elements.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [gt_bboxes_ignore_list for _ in range(num_imgs)]
        
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         pts_targets_list, pts_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
            self._get_target_single, cls_scores_list, bbox_preds_list, pts_preds_list,
            gt_labels_list, gt_bboxes_list, gt_bboxes_ignore_list)
        
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        
        return (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
                pts_targets_list, pts_weights_list, num_total_pos, num_total_neg)
    
    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           pts_pred,
                           gt_labels,
                           gt_bboxes,
                           gt_bboxes_ignore=None):
        """
        Compute targets for a single image.
        
        This is a simplified version - you need to implement proper matching
        based on your assigner configuration (e.g., MapTRAssigner).
        """
        num_bboxes = bbox_pred.size(0)
        # Assigner and sampler would be used here
        # For now, create dummy targets
        
        # Assign all as negative samples
        assigned_gt_inds = bbox_pred.new_zeros(num_bboxes, dtype=torch.long)
        assigned_labels = bbox_pred.new_full((num_bboxes,), self.num_classes, dtype=torch.long)
        
        label_weights = bbox_pred.new_ones(num_bboxes)
        bbox_targets = torch.zeros_like(bbox_pred)
        bbox_weights = torch.zeros_like(bbox_pred)
        pts_targets = torch.zeros_like(pts_pred)
        pts_weights = torch.zeros_like(pts_pred)
        
        pos_inds = torch.nonzero(assigned_gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(assigned_gt_inds == 0, as_tuple=False).squeeze(-1).unique()
        
        return (assigned_labels, label_weights, bbox_targets, bbox_weights,
                pts_targets, pts_weights, pos_inds, neg_inds)


def normalize_bbox(bboxes, pc_range):
    """Normalize bounding boxes to [0, 1] range based on point cloud range."""
    cx = bboxes[..., 0:1]
    cy = bboxes[..., 1:2]
    
    cx = (cx - pc_range[0]) / (pc_range[3] - pc_range[0])
    cy = (cy - pc_range[1]) / (pc_range[4] - pc_range[1])
    
    normalized = bboxes.clone()
    normalized[..., 0:1] = cx
    normalized[..., 1:2] = cy
    
    return normalized
