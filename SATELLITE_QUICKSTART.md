# Satellite Image Integration - Quick Start Guide

## What Was Added

Satellite image support has been integrated into the `CustomNuScenesLocalMapDataset` to enable BEV-satellite fusion for HD map construction.

## Quick Setup (3 Steps)

### 1. Prepare Satellite Images

Organize your satellite images:
```bash
data/nuscenes/satellite_maps/
├── boston-seaport_<token1>.png
├── singapore-hollandvillage_<token2>.png
└── ...
```

Naming format: `{location}_{sample_token}.{format}`

### 2. Update Dataset Config

```python
train_dataset = dict(
    type='CustomNuScenesLocalMapDataset',
    # ... existing params ...
    
    # Add these 3 lines:
    use_satellite=True,
    satellite_dir='data/nuscenes/satellite_maps/',
    satellite_size=(512, 512),
)
```

### 3. Update Pipeline

```python
train_pipeline = [
    dict(type='CustomLoadMultiViewImageFromFiles', to_float32=True),
    
    # Add this line:
    dict(type='LoadSatelliteImageFromFile', to_float32=True, satellite_size=(512, 512)),
    
    # ... rest of pipeline ...
    dict(type='CustomCollect3D', keys=['img', 'satellite_img', 'gt_bboxes_3d', 'gt_labels_3d']),
]
```

## Files Modified

1. **`nuscenes_map_dataset.py`** - Added satellite loading to dataset
2. **`pipelines/loading.py`** - Added `LoadSatelliteImageFromFile` pipeline
3. **`pipelines/__init__.py`** - Exported new pipeline component

## Files Created

1. **`SATELLITE_INTEGRATION.md`** - Comprehensive documentation
2. **`SATELLITE_CHANGES.md`** - Detailed change summary
3. **`agfusion_with_satellite_example.py`** - Complete config example

## Usage in Model

```python
class YourModel(nn.Module):
    def forward(self, img, satellite_img, **kwargs):
        # Extract BEV features from camera
        bev_feat = self.bev_encoder(img)
        
        # Extract features from satellite
        sat_feat = self.satellite_encoder(satellite_img)
        
        # Fuse features
        fused_feat = self.fusion_module(bev_feat, sat_feat)
        
        # Predict map
        output = self.head(fused_feat)
        return output
```

## What You Get

After integration, each data sample contains:
- `satellite_img`: Satellite image array (H, W, 3)
- `satellite_img_path`: Path to the image file
- `satellite_metadata`: Alignment info (pose, coordinates, etc.)
- `satellite_img_available`: Whether valid image was loaded

## Example Data Shapes

```python
# Single sample
img.shape = (6, 3, 928, 1600)           # 6 cameras
satellite_img.shape = (3, 512, 512)      # 1 satellite image

# With temporal queue (queue_length=4)
img.shape = (4, 6, 3, 928, 1600)        # 4 frames × 6 cameras
satellite_img.shape = (4, 3, 512, 512)   # 4 frames × satellite

# Batched (batch_size=2)
img.shape = (2, 4, 6, 3, 928, 1600)     
satellite_img.shape = (2, 4, 3, 512, 512)
```

## Testing

```python
from mmdet3d.datasets import build_dataset
from mmcv import Config

# Load config
cfg = Config.fromfile('configs/agfusion/agfusion_with_satellite_example.py')

# Build dataset
dataset = build_dataset(cfg.data.train)

# Test loading
data = dataset[0]
print(f"Keys: {data.keys()}")
print(f"Satellite image shape: {data['satellite_img'].shape}")
print(f"Satellite available: {data['img_metas']['satellite_img_available']}")
```

## Compatible with AGFusion Components

✅ **SatelliteFeatureExtractor** - Process satellite images
✅ **MapFusion** - Fuse BEV and satellite features  
✅ **MapTRPerceptionTransformer** - Transform with fusion
✅ **MapTRHead** - Predict maps from fused features

## Troubleshooting

**Q: Images not loading?**
- Check filename format: `{location}_{token}.{format}`
- Verify `satellite_dir` path exists
- Ensure file extensions match `satellite_format`

**Q: Out of memory?**
- Reduce `satellite_size` (e.g., 512→256)
- Reduce `queue_length`
- Enable fp16 training

**Q: Features misaligned?**
- Use `AlignedSatelliteEncoder` 
- Set `use_geo_transform=True`
- Verify satellite images are north-aligned

## Documentation

📖 **Full Guide**: `projects/mmdet3d_plugin/datasets/SATELLITE_INTEGRATION.md`
📋 **Changes**: `projects/mmdet3d_plugin/datasets/SATELLITE_CHANGES.md`
⚙️ **Example Config**: `projects/configs/agfusion/agfusion_with_satellite_example.py`

## Next Steps

1. Prepare/download satellite imagery for your dataset
2. Update your config with satellite parameters
3. Add satellite encoder to your model
4. Train with BEV-satellite fusion
5. Evaluate on validation set

## Support

For issues or questions about satellite integration:
1. Check the documentation files
2. Review the example config
3. Verify your satellite image format and naming
4. Test with a small subset first

---

**Note**: Satellite images are optional. Set `use_satellite=False` or omit the parameters to use the dataset without satellite imagery (backward compatible).
