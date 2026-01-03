"""
Example configuration for AGFusion with satellite image integration.

This config shows how to use satellite imagery alongside camera and LiDAR
data for HD map construction with BEV-satellite fusion.
"""

_base_ = [
    '../_base_/datasets/custom_nus-3d.py',
    '../_base_/default_runtime.py'
]

# Model configuration
model = dict(
    type='AGFusion',
    
    # Satellite feature encoder
    satellite_encoder=dict(
        type='AlignedSatelliteEncoder',
        in_channels=3,
        out_channels=256,
        bev_h=200,
        bev_w=100,
        base_channels=64,
        use_geo_transform=True,
    ),
    
    # BEV encoder (camera-based)
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch'),
    
    # Fusion module
    fusion_module=dict(
        type='MapFusion',
        in_channels=256,
        out_channels=256,
        fusion_method='attention',  # Options: 'concat', 'add', 'attention', 'adaptive'
        num_heads=8,
        dropout=0.1,
    ),
    
    # MapTR Transformer
    pts_bbox_head=dict(
        type='MapTRHead',
        num_classes=3,
        in_channels=256,
        embed_dims=256,
        num_query=900,
        num_pts_per_vec=20,
        # ... other head params ...
    ),
)

# Dataset configuration
dataset_type = 'CustomNuScenesLocalMapDataset'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')

# Map classes
map_classes = ['divider', 'ped_crossing', 'boundary']
num_map_classes = len(map_classes)

# Point cloud range (for BEV)
point_cloud_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
voxel_size = [0.15, 0.15, 4]

# BEV grid size
bev_h_ = 200
bev_w_ = 100

# Image normalization
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True
)

# Training pipeline
train_pipeline = [
    # Load camera images
    dict(
        type='CustomLoadMultiViewImageFromFiles',
        to_float32=True,
        color_type='color'
    ),
    
    # Load satellite images
    dict(
        type='LoadSatelliteImageFromFile',
        to_float32=True,
        color_type='color',
        satellite_size=(512, 512),  # Target satellite image size
        default_value=0,  # Fill value for missing images
    ),
    
    # Data augmentation
    dict(
        type='PhotoMetricDistortionMultiViewImage',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18
    ),
    
    # Normalize images
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    
    # Pad to divisible size
    dict(type='PadMultiViewImage', size_divisor=32),
    
    # Format bundle
    dict(
        type='CustomDefaultFormatBundle3D',
        class_names=map_classes
    ),
    
    # Collect keys
    dict(
        type='CustomCollect3D',
        keys=[
            'img',              # Camera images
            'satellite_img',    # Satellite images
            'gt_bboxes_3d',     # Map element annotations
            'gt_labels_3d'      # Map element labels
        ],
        meta_keys=[
            'filename',
            'satellite_img_filename',
            'satellite_metadata',
            'ori_shape',
            'img_shape',
            'satellite_img_shape',
            'lidar2img',
            'lidar2global',
            'can_bus',
            'sample_idx',
            'scene_token',
        ]
    ),
]

# Test pipeline
test_pipeline = [
    dict(
        type='CustomLoadMultiViewImageFromFiles',
        to_float32=True,
        color_type='color'
    ),
    dict(
        type='LoadSatelliteImageFromFile',
        to_float32=True,
        color_type='color',
        satellite_size=(512, 512),
    ),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1600, 900),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='CustomDefaultFormatBundle3D',
                class_names=map_classes,
                with_label=False
            ),
            dict(
                type='CustomCollect3D',
                keys=['img', 'satellite_img'],
                meta_keys=[
                    'filename',
                    'satellite_img_filename',
                    'satellite_metadata',
                    'ori_shape',
                    'img_shape',
                    'satellite_img_shape',
                    'lidar2img',
                    'lidar2global',
                    'can_bus',
                    'sample_idx',
                    'scene_token',
                ]
            )
        ]
    )
]

# Dataset settings
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_train_with_ego_pose.pkl',
        
        # Satellite configuration
        use_satellite=True,
        satellite_dir=data_root + 'satellite_maps/',
        satellite_size=(512, 512),
        satellite_format='png',
        
        # Map configuration
        map_classes=map_classes,
        fixed_ptsnum_per_line=20,
        eval_use_same_gt_sample_num_flag=True,
        padding_value=-10000,
        
        # Spatial configuration
        bev_size=(bev_h_, bev_w_),
        pc_range=point_cloud_range,
        
        # Temporal configuration
        queue_length=4,
        overlap_test=False,
        
        pipeline=train_pipeline,
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False,
        ),
        test_mode=False,
    ),
    
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_val_with_ego_pose.pkl',
        
        # Satellite configuration
        use_satellite=True,
        satellite_dir=data_root + 'satellite_maps/',
        satellite_size=(512, 512),
        satellite_format='png',
        
        # Map configuration
        map_classes=map_classes,
        fixed_ptsnum_per_line=20,
        eval_use_same_gt_sample_num_flag=True,
        
        # Spatial configuration
        bev_size=(bev_h_, bev_w_),
        pc_range=point_cloud_range,
        
        # Temporal configuration
        queue_length=4,
        overlap_test=False,
        
        pipeline=test_pipeline,
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False,
        ),
        test_mode=True,
    ),
    
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_val.pkl',
        
        # Satellite configuration
        use_satellite=True,
        satellite_dir=data_root + 'satellite_maps/',
        satellite_size=(512, 512),
        satellite_format='png',
        
        # Map configuration
        map_classes=map_classes,
        fixed_ptsnum_per_line=20,
        
        # Spatial configuration
        bev_size=(bev_h_, bev_w_),
        pc_range=point_cloud_range,
        
        pipeline=test_pipeline,
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False,
        ),
        test_mode=True,
    )
)

# Training schedule
optimizer = dict(
    type='AdamW',
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
            'satellite_encoder': dict(lr_mult=1.0),
        }
    ),
    weight_decay=0.01
)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3
)

total_epochs = 24
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)

evaluation = dict(
    interval=1,
    pipeline=test_pipeline,
    metric='chamfer'
)

# Checkpointing
checkpoint_config = dict(interval=1, max_keep_ckpts=5)

# Logging
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ]
)

# Runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/agfusion_with_satellite'
load_from = None
resume_from = None
workflow = [('train', 1)]

# fp16 settings
fp16 = dict(loss_scale=512.)

# Note: Before running, ensure your satellite images are organized as:
# data/nuscenes/satellite_maps/
#   ├── boston-seaport_{sample_token}.png
#   ├── singapore-hollandvillage_{sample_token}.png
#   └── ...
