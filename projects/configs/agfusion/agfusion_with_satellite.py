_base_ = [
    '../datasets/custom_nus-3d.py',
    '../_base_/default_runtime.py'
]

plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

# Point cloud range
point_cloud_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
voxel_size = [0.15, 0.15, 4]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

# Map classes
map_classes = ['divider', 'ped_crossing','boundary']
fixed_ptsnum_per_gt_line = 20
fixed_ptsnum_per_pred_line = 20
eval_use_same_gt_sample_num_flag = True
num_map_classes = len(map_classes)

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)

_dim_ = 256
_pos_dim_ = _dim_//2
_ffn_dim_ = _dim_*2
_num_levels_ = 1
bev_h_ = 200
bev_w_ = 100
queue_length = 1

# Satellite configuration
use_satellite = True
satellite_dir = '/media/kanke/easystore/Lambda/Data_set/data_nuscences/satellite_map_dataset/satellite_map_trainval'  # Update this path
satellite_size = (400, 200)  # (height, width) matching PriorMap default

# Dataset configuration with satellite support
data_config = {
    'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
             'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
    'Ncams': 6,
    'input_size': (256, 704),
    'src_size': (900, 1600),
    'thickness': 5,
    'angle_class': 36,
}

# Update data pipeline to include satellite images
train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='CustomCollect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img', 'satellite_img'])
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='MultiScaleFlipAug3D',
         img_scale=(1333, 800),
         pts_scale_ratio=1,
         flip=False,
         transforms=[
             dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
             dict(type='CustomCollect3D', keys=['img', 'satellite_img'])
         ])
]

# Dataset settings
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type='CustomNuScenesLocalMapDataset',
        data_root='data/nuscenes/',
        ann_file='data/nuscenes/nuscenes_infos_temporal_train_with_ego_pose.pkl',
        map_ann_file='data/nuscenes/nuscenes_map_anns_train.json',
        pipeline=train_pipeline,
        classes=class_names,
        map_classes=map_classes,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        bev_size=(bev_h_, bev_w_),
        pc_range=point_cloud_range,
        queue_length=queue_length,
        fixed_ptsnum_per_line=fixed_ptsnum_per_gt_line,
        eval_use_same_gt_sample_num_flag=eval_use_same_gt_sample_num_flag,
        padding_value=-10000,
        # Satellite image settings
        use_satellite=use_satellite,
        satellite_dir=satellite_dir,
        satellite_size=satellite_size,
        box_type_3d='LiDAR'),
    val=dict(
        type='CustomNuScenesLocalMapDataset',
        data_root='data/nuscenes/',
        ann_file='data/nuscenes/nuscenes_infos_temporal_val_with_ego_pose.pkl',
        map_ann_file='data/nuscenes/nuscenes_map_anns_val.json',
        pipeline=test_pipeline,
        classes=class_names,
        map_classes=map_classes,
        modality=input_modality,
        test_mode=True,
        bev_size=(bev_h_, bev_w_),
        pc_range=point_cloud_range,
        queue_length=queue_length,
        fixed_ptsnum_per_line=fixed_ptsnum_per_gt_line,
        eval_use_same_gt_sample_num_flag=eval_use_same_gt_sample_num_flag,
        padding_value=-10000,
        # Satellite image settings
        use_satellite=use_satellite,
        satellite_dir=satellite_dir,
        satellite_size=satellite_size,
        box_type_3d='LiDAR'),
    test=dict(
        type='CustomNuScenesLocalMapDataset',
        data_root='data/nuscenes/',
        ann_file='data/nuscenes/nuscenes_infos_temporal_val_with_ego_pose.pkl',
        map_ann_file='data/nuscenes/nuscenes_map_anns_val.json',
        pipeline=test_pipeline,
        classes=class_names,
        map_classes=map_classes,
        modality=input_modality,
        test_mode=True,
        bev_size=(bev_h_, bev_w_),
        pc_range=point_cloud_range,
        queue_length=queue_length,
        fixed_ptsnum_per_line=fixed_ptsnum_per_gt_line,
        eval_use_same_gt_sample_num_flag=eval_use_same_gt_sample_num_flag,
        padding_value=-10000,
        # Satellite image settings
        use_satellite=use_satellite,
        satellite_dir=satellite_dir,
        satellite_size=satellite_size,
        box_type_3d='LiDAR'))

# Model configuration would go here
# This is a minimal config focused on satellite integration
