# Satellite Image Integration for AGFusion

This guide explains how to integrate satellite imagery with the CustomNuScenesLocalMapDataset for BEV-satellite fusion in HD map construction.

## Overview

The dataset now supports loading and processing satellite imagery alongside camera and LiDAR data. Satellite images are aligned with the BEV grid coordinate system and can be fused with onboard sensor features using the AGFusion module.

## Dataset Configuration

### Basic Setup

Add the following parameters to your dataset configuration:

```python
dataset_config = dict(
    type='CustomNuScenesLocalMapDataset',
    data_root='data/nuscenes/',
    # ... other parameters ...
    
    # Satellite image configuration
    use_satellite=True,
    satellite_dir='data/nuscenes/satellite_images/',
    satellite_size=(512, 512),  # Target size (H, W)
    satellite_format='png',  # Image format: 'png', 'jpg', etc.
)
```

### Configuration Parameters

- **`use_satellite`** (bool): Enable satellite image loading. Default: `False`
- **`satellite_dir`** (str): Directory containing satellite images. Required if `use_satellite=True`
- **`satellite_size`** (tuple): Target size for satellite images `(height, width)`. Default: `(512, 512)`
- **`satellite_format`** (str): Image file format. Default: `'png'`

## Satellite Image Organization

### Directory Structure

Organize your satellite images in the following structure:

```
data/nuscenes/
├── satellite_images/
│   ├── boston-seaport_<sample_token_1>.png
│   ├── boston-seaport_<sample_token_2>.png
│   ├── singapore-hollandvillage_<sample_token_3>.png
│   └── ...
```

### Naming Convention

Satellite images should be named using the format:
```
{location}_{sample_token}.{format}
```

Where:
- `location`: NuScenes map location (e.g., 'boston-seaport', 'singapore-hollandvillage')
- `sample_token`: Unique sample identifier from NuScenes dataset
- `format`: Image file extension (png, jpg, etc.)

**Example:**
```
boston-seaport_fd8420396768427d8374d0eb17c14f2c.png
```

### Custom Naming Convention

If you need a different naming convention, override the `_get_satellite_filename` method:

```python
class CustomSatelliteDataset(CustomNuScenesLocalMapDataset):
    def _get_satellite_filename(self, sample_token, location):
        # Your custom naming logic
        return f"sat_{sample_token}.png"
```

## Pipeline Configuration

Add the satellite image loader to your data pipeline:

```python
train_pipeline = [
    # ... existing pipeline steps ...
    
    # Load camera images
    dict(type='CustomLoadMultiViewImageFromFiles', to_float32=True),
    
    # Load satellite images
    dict(
        type='LoadSatelliteImageFromFile',
        to_float32=True,
        color_type='color',
        satellite_size=(512, 512),  # Should match dataset config
        default_value=0,  # Pixel value if image not found
    ),
    
    # ... rest of pipeline ...
]
```

### Pipeline Parameters

**LoadSatelliteImageFromFile:**
- **`to_float32`** (bool): Convert image to float32. Default: `False`
- **`color_type`** (str): Color type ('color', 'grayscale', 'unchanged'). Default: `'color'`
- **`satellite_size`** (tuple): Target image size `(H, W)`. Default: `None`
- **`default_value`** (int): Fill value for missing images. Default: `0`

## Data Dictionary

After loading, the following keys are added to the data dictionary:

### Dataset Level
- **`satellite_img_path`** (str): Path to satellite image file
- **`satellite_metadata`** (dict): Metadata for alignment
  - `ego2global_translation`: Vehicle position in global coordinates
  - `ego2global_rotation`: Vehicle rotation quaternion
  - `lidar2global`: 4x4 transformation matrix
  - `patch_size`: BEV patch size (H, W)
  - `satellite_size`: Satellite image size (H, W)

### Pipeline Level
- **`satellite_img`** (np.ndarray): Satellite image array, shape `(H, W, 3)`
- **`satellite_img_filename`** (str): Image filename
- **`satellite_img_shape`** (tuple): Image shape `(H, W, C)`
- **`satellite_img_available`** (bool): Whether valid image was loaded

## Model Integration

### Using with AGFusion

The satellite images are automatically batched and can be accessed in your model:

```python
class AGFusionModel(nn.Module):
    def forward(self, img, satellite_img, **kwargs):
        # Extract features from camera images
        bev_features = self.bev_encoder(img)
        
        # Extract features from satellite images
        satellite_features = self.satellite_encoder(satellite_img)
        
        # Fuse features
        fused_features = self.fusion_module(bev_features, satellite_features)
        
        return fused_features
```

### Temporal Fusion

For temporal models with queue_length > 1, satellite images are stacked:

```python
# satellite_img shape: (batch, queue_length, C, H, W)
# Access current frame: satellite_img[:, -1]
# Access all frames: satellite_img
```

## Complete Configuration Example

```python
# Dataset configuration
train_dataset = dict(
    type='CustomNuScenesLocalMapDataset',
    data_root='data/nuscenes/',
    ann_file='data/nuscenes/nuscenes_infos_train.pkl',
    
    # Map configuration
    map_classes=['divider', 'ped_crossing', 'boundary'],
    fixed_ptsnum_per_line=20,
    
    # Satellite configuration
    use_satellite=True,
    satellite_dir='data/nuscenes/satellite_maps/',
    satellite_size=(512, 512),
    satellite_format='png',
    
    # Other parameters
    queue_length=4,
    bev_size=(200, 100),
    pc_range=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
    
    pipeline=train_pipeline,
)

# Pipeline configuration
train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(
        type='LoadSatelliteImageFromFile',
        to_float32=True,
        satellite_size=(512, 512),
    ),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='CustomCollect3D',
        keys=['img', 'satellite_img', 'gt_bboxes_3d', 'gt_labels_3d']
    ),
]

# Model configuration with satellite encoder
model = dict(
    type='AGFusion',
    
    # Satellite feature encoder
    satellite_encoder=dict(
        type='AlignedSatelliteEncoder',
        in_channels=3,
        out_channels=256,
        bev_h=200,
        bev_w=100,
    ),
    
    # BEV encoder
    bev_encoder=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
    ),
    
    # Fusion module
    fusion_module=dict(
        type='MapFusion',
        in_channels=256,
        out_channels=256,
        fusion_method='attention',
    ),
    
    # ... rest of model config ...
)
```

## Satellite Image Preparation

### Recommended Specifications

- **Resolution**: 0.1-0.5 meters per pixel
- **Format**: PNG or JPEG
- **Size**: 512x512 or 1024x1024 pixels
- **Coverage**: Should cover the BEV patch area (e.g., 30m x 60m)
- **Alignment**: Top-down view, north-aligned

### Data Sources

Common satellite imagery sources:
- Google Maps Static API
- Mapbox Static Images API
- Bing Maps API
- OpenStreetMap aerial imagery
- Commercial providers (Planet, Maxar, etc.)

### Pre-processing Script Example

```python
import requests
from PIL import Image
import numpy as np

def download_satellite_image(lat, lon, zoom, token, size=512):
    """Download satellite image from Mapbox."""
    url = f"https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/{lon},{lat},{zoom}/{size}x{size}?access_token={token}"
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return np.array(img)

# Process NuScenes dataset
for sample in nuscenes_samples:
    lat, lon = sample['ego2global_translation'][:2]
    img = download_satellite_image(lat, lon, zoom=18, token=YOUR_TOKEN)
    
    # Save with proper naming
    location = sample['map_location']
    token = sample['token']
    filename = f"{location}_{token}.png"
    Image.fromarray(img).save(f"data/nuscenes/satellite_maps/{filename}")
```

## Troubleshooting

### Missing Satellite Images

If satellite images are missing, the pipeline will create blank images filled with `default_value`. Check:
1. Image filenames match the naming convention
2. `satellite_dir` path is correct
3. Images have correct format extension

### Memory Issues

For large satellite images or long queues:
1. Reduce `satellite_size` (e.g., from 1024 to 512)
2. Reduce `queue_length`
3. Use mixed precision training (fp16)
4. Enable gradient checkpointing

### Alignment Issues

If satellite features don't align with BEV:
1. Verify satellite image orientation (should be north-aligned)
2. Check coordinate system matches NuScenes global coordinates
3. Adjust `patch_size` in dataset config
4. Use `AlignedSatelliteEncoder` with `use_geo_transform=True`

## Performance Tips

1. **Pre-resize images**: Resize satellite images to target size offline to speed up loading
2. **Use SSD storage**: Store satellite images on fast storage for I/O performance
3. **Cache frequently**: Enable dataset caching for repeated training runs
4. **Batch efficiently**: Use larger batch sizes with smaller satellite images

## Citation

If you use satellite imagery integration in your research, please cite:

```bibtex
@article{agfusion2023,
  title={Complementing Onboard Sensors with Satellite Map: A New Perspective for HD Map Construction},
  author={Your Name},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2023}
}
```
