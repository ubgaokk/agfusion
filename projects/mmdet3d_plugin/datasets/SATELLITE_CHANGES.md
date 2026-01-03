# Satellite Image Integration Summary

## Changes Made

This document summarizes the modifications made to integrate satellite imagery support into the CustomNuScenesLocalMapDataset.

## Modified Files

### 1. `nuscenes_map_dataset.py`

**Location:** `projects/mmdet3d_plugin/datasets/nuscenes_map_dataset.py`

**Changes:**

#### `CustomNuScenesLocalMapDataset.__init__()` (lines ~905-960)
Added parameters:
- `use_satellite` (bool): Enable/disable satellite image loading
- `satellite_dir` (str): Directory path for satellite images
- `satellite_size` (tuple): Target satellite image size (H, W)
- `satellite_format` (str): Image file format extension

```python
def __init__(
    self,
    # ... existing parameters ...
    use_satellite=False,
    satellite_dir=None,
    satellite_size=(512, 512),
    satellite_format='png',
    *args, 
    **kwargs
):
```

#### New method: `_get_satellite_filename()` (lines ~987-1003)
Generates satellite image filename from sample token and location:
```python
def _get_satellite_filename(self, sample_token, location):
    """Generate satellite image filename."""
    filename = f"{location}_{sample_token}.{self.satellite_format}"
    return filename
```

#### Modified: `get_data_info()` (lines ~1100-1250)
Added satellite image path and metadata to `input_dict`:
```python
if self.use_satellite:
    satellite_filename = self._get_satellite_filename(info['token'], input_dict['map_location'])
    satellite_path = osp.join(self.satellite_dir, satellite_filename)
    input_dict['satellite_img_path'] = satellite_path
    
    input_dict['satellite_metadata'] = dict(
        ego2global_translation=translation,
        ego2global_rotation=rotation,
        lidar2global=lidar2global,
        patch_size=self.patch_size,
        satellite_size=self.satellite_size,
    )
```

#### Modified: `union2one()` (lines ~1068-1110)
Added satellite image stacking for temporal sequences:
```python
# Collect satellite images if available
satellite_imgs_list = []
has_satellite = self.use_satellite and 'satellite_img' in queue[0]

# ... loop through queue ...
if has_satellite:
    satellite_imgs_list.append(each['satellite_img'].data)

# Stack satellite images
if has_satellite and len(satellite_imgs_list) > 0:
    queue[-1]['satellite_img'] = DC(torch.stack(satellite_imgs_list),
                                   cpu_only=False, stack=True)
```

### 2. `pipelines/loading.py`

**Location:** `projects/mmdet3d_plugin/datasets/pipelines/loading.py`

**Changes:**

#### New class: `LoadSatelliteImageFromFile` (lines ~370-490)
Pipeline component for loading satellite images:

```python
@PIPELINES.register_module()
class LoadSatelliteImageFromFile(object):
    """Load satellite image from file."""
    
    def __init__(
        self,
        to_float32=False,
        color_type='color',
        file_client_args=dict(backend='disk'),
        default_value=0,
        satellite_size=None,
    ):
        # ...
    
    def __call__(self, results):
        # Load satellite image or create default if missing
        # Resize to target size
        # Add to results dict
        # ...
```

**Features:**
- Loads satellite images from disk
- Handles missing images gracefully (creates blank image)
- Automatic resizing to target dimensions
- Supports multiple color types and formats
- Compatible with mmcv FileClient system

### 3. `pipelines/__init__.py`

**Location:** `projects/mmdet3d_plugin/datasets/pipelines/__init__.py`

**Changes:**
Added `LoadSatelliteImageFromFile` to imports and exports:

```python
from .loading import (
    CustomLoadPointsFromFile, 
    CustomLoadPointsFromMultiSweeps, 
    CustomLoadMultiViewImageFromFiles,
    LoadSatelliteImageFromFile  # NEW
)

__all__ = [
    # ... existing exports ...
    'LoadSatelliteImageFromFile'  # NEW
]
```

## New Files

### 4. `SATELLITE_INTEGRATION.md`

**Location:** `projects/mmdet3d_plugin/datasets/SATELLITE_INTEGRATION.md`

Comprehensive documentation covering:
- Configuration guide
- Satellite image organization and naming
- Pipeline setup
- Model integration examples
- Data dictionary reference
- Complete configuration examples
- Satellite image preparation guidelines
- Troubleshooting tips
- Performance optimization

## Data Flow

### Training Pipeline

```
1. Dataset.__getitem__()
   ↓
2. get_data_info()
   ├── Adds 'satellite_img_path'
   └── Adds 'satellite_metadata'
   ↓
3. Pipeline: LoadSatelliteImageFromFile
   ├── Loads image from disk
   ├── Resizes to target size
   └── Adds 'satellite_img' to results
   ↓
4. Pipeline: Other transforms
   ├── Normalize, pad, augment, etc.
   └── satellite_img treated like regular image
   ↓
5. union2one() [for temporal models]
   ├── Stacks satellite images across time
   └── Shape: (queue_length, C, H, W)
   ↓
6. Batch collation
   └── Final shape: (B, queue_length, C, H, W)
   ↓
7. Model forward()
   ├── satellite_encoder(satellite_img)
   ├── bev_encoder(camera_img)
   └── fusion_module(bev_feat, sat_feat)
```

## Usage Example

### Minimal Configuration

```python
# Dataset
train_dataset = dict(
    type='CustomNuScenesLocalMapDataset',
    use_satellite=True,
    satellite_dir='data/nuscenes/satellite_maps/',
    satellite_size=(512, 512),
    # ... other params ...
)

# Pipeline
train_pipeline = [
    dict(type='CustomLoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadSatelliteImageFromFile', to_float32=True, satellite_size=(512, 512)),
    # ... other transforms ...
    dict(type='CustomCollect3D', keys=['img', 'satellite_img', 'gt_bboxes_3d', 'gt_labels_3d']),
]
```

### Model Integration

```python
model = dict(
    type='AGFusion',
    satellite_encoder=dict(type='AlignedSatelliteEncoder', out_channels=256),
    fusion_module=dict(type='MapFusion', fusion_method='attention'),
    # ... other components ...
)
```

## Key Features

✅ **Flexible naming**: Customizable filename generation
✅ **Graceful degradation**: Handles missing images with defaults
✅ **Temporal support**: Automatic stacking for temporal models
✅ **Memory efficient**: Configurable image sizes
✅ **Metadata rich**: Includes alignment info for coordinate transforms
✅ **Pipeline compatible**: Integrates seamlessly with MMDetection3D pipelines
✅ **Format agnostic**: Supports PNG, JPEG, and other formats

## Integration with AGFusion Components

The satellite image data integrates with previously created components:

1. **SatelliteFeatureExtractor**: Processes loaded satellite images
2. **MapFusion**: Fuses satellite features with BEV features
3. **MapTRPerceptionTransformer**: Accepts satellite features as input
4. **MapTRHead**: Uses fused features for map prediction

## Testing

To verify the integration:

```python
from mmdet3d.datasets import build_dataset

# Build dataset
dataset = build_dataset(cfg.data.train)

# Get a sample
data = dataset[0]

# Check satellite image presence
assert 'satellite_img' in data
print(f"Satellite image shape: {data['satellite_img'].shape}")
print(f"Satellite metadata: {data['img_metas']['satellite_metadata']}")
```

## Backward Compatibility

✅ Fully backward compatible:
- `use_satellite=False` by default
- Existing configs work without modification
- No breaking changes to existing functionality

## Performance Considerations

- **I/O**: Satellite images loaded on-the-fly (consider pre-caching)
- **Memory**: ~3MB per 512x512 RGB image (uncompressed)
- **Compute**: Minimal overhead, handled by existing pipeline infrastructure

## Future Enhancements

Potential improvements:
1. Pre-cached satellite features (load features instead of images)
2. Multi-resolution satellite pyramids
3. Temporal satellite image sequences
4. Dynamic satellite image retrieval from APIs
5. Automatic alignment/registration with BEV grid
