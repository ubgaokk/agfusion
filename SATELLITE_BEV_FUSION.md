# Satellite-BEV Fusion Integration

**Date:** January 8, 2026  
**Module:** Fusion_Atten_Masked  
**Status:** ✅ COMPLETE

---

## Overview

Successfully integrated masked attention-based fusion between BEV features and satellite features in the MapTR detector. The fusion uses cross-attention with distance-based masking to effectively combine multi-modal information.

---

## Architecture

### Fusion_Atten_Masked Module

```
BEV Features (B, C_bev, H, W)
      ↓
  Patch Embedding → Query (B, n_patches, hidden_dim)
      ↓                             ↓
Satellite Features              Cross-Attention
(B, C_sat, H, W)                with Distance Mask
      ↓                             ↓
  Patch Embedding → Key/Value   Transformer Decoder
                                (3 layers default)
                                    ↓
                                Expand & Reshape
                                    ↓
                    Fused BEV Features (B, C_bev, H, W)
```

### Key Components

1. **Patch Embedding**
   - Converts both BEV and satellite features into patches
   - Adds learnable positional embeddings
   - Projects to hidden dimension

2. **Transformer Decoder**
   - BEV features as Query
   - Satellite features as Key/Value
   - Multi-head cross-attention (8 heads default)
   - 3 decoder layers with residual connections

3. **Distance-Based Masking**
   - Masks attention based on spatial distance
   - Configurable distance threshold (default: 5 patches)
   - Prevents over-smoothing from distant regions

4. **Output Modes**
   - **'fused'**: Single feature map (C_bev channels)
   - **'concat'**: Concatenated features (C_bev + C_sat channels)

---

## Integration in MapTR Detector

### Modified Files

1. **`masked_t.py`**
   - Updated `Fusion_Atten_Masked` class
   - Added satellite feature handling
   - Added resize capability for mismatched dimensions
   - Added output mode selection

2. **`detectors/maptr.py`**
   - Added `sat_fusion` parameter to `__init__`
   - Integrated fusion in `forward_pts_train`
   - Integrated fusion in `simple_test_pts`
   - Pass satellite features through pipeline

### Data Flow

```python
# Training
satellite_img → extract_sat_feat() → sat_feats
                                         ↓
img → extract_feat() → img_feats (BEV)  ↓
                          ↓              ↓
                    forward_pts_train(img_feats, sat_feats)
                          ↓
                    sat_fusion(bev_feat, sat_feats)
                          ↓
                    pts_bbox_head(fused_feats)

# Inference
Same flow through simple_test → simple_test_pts
```

---

## Configuration

### Basic Configuration

```python
model = dict(
    type='MapTR',
    
    # Satellite encoder
    sat_backbone=dict(
        type='AlignedSatelliteEncoder',
        in_channels=3,
        out_channels=256,
        bev_h=200,
        bev_w=400,
        # ... other params
    ),
    
    # NEW: Satellite-BEV fusion module
    sat_fusion=dict(
        bev_channels=256,      # Must match BEV feature channels
        sat_channels=256,      # Must match satellite encoder output
        hidden_c=384,          # Transformer hidden dimension
        img_size=(200, 400),   # BEV size (H, W)
        patch_size=(10, 10),   # Patch size for attention
        dis=5,                 # Distance threshold for masking
        decoder_layers=3,      # Number of transformer layers
        num_heads=8,           # Number of attention heads
        dropout=0.1,           # Dropout rate
        mlp_ratio=4,           # MLP expansion ratio
        output_mode='fused'    # 'fused' or 'concat'
    ),
    
    # ... rest of config
)
```

### Advanced Configuration

```python
# For larger BEV grids
sat_fusion=dict(
    bev_channels=256,
    sat_channels=256,
    hidden_c=512,           # Larger hidden dim
    img_size=(400, 800),    # Larger BEV
    patch_size=(20, 20),    # Larger patches for efficiency
    dis=3,                  # Smaller distance threshold
    decoder_layers=4,       # More layers
    num_heads=16,           # More attention heads
    output_mode='concat'    # Keep both feature types
)
```

---

## Key Features

### 1. Automatic Feature Resize
```python
# Handles mismatched dimensions automatically
if sat_features.shape[2:] != bev_features.shape[2:]:
    sat_features = F.interpolate(sat_features, ...)
```

### 2. Graceful Degradation
```python
# Works without satellite features
if sat_features is None:
    return bev_features  # Skip fusion
```

### 3. Distance-Based Attention Mask
```python
# Only attend to nearby patches
mask[x1][y1][x2][y2] = float('-inf') if distance > threshold
```

### 4. Multi-Scale Support
```python
# Fuses with highest resolution BEV feature
if isinstance(pts_feats, list):
    bev_feat = pts_feats[-1]  # Last feature map
    fused_feat = self.sat_fusion(bev_feat, sat_feats)
    pts_feats[-1] = fused_feat
```

---

## Parameters Guide

| Parameter | Default | Description |
|-----------|---------|-------------|
| `bev_channels` | - | BEV feature channels (must match) |
| `sat_channels` | - | Satellite feature channels (must match) |
| `hidden_c` | 384 | Transformer hidden dimension |
| `img_size` | (200, 400) | BEV spatial size (H, W) |
| `patch_size` | (10, 10) | Patch size for attention |
| `dis` | 5 | Distance threshold (in patches) |
| `decoder_layers` | 3 | Number of transformer layers |
| `num_heads` | 8 | Number of attention heads |
| `dropout` | 0.1 | Dropout rate |
| `mlp_ratio` | 4 | MLP expansion ratio |
| `drop` | 0.0 | Additional drop rate |
| `drop_path` | 0.0 | DropPath rate |
| `output_mode` | 'fused' | 'fused' or 'concat' |

---

## Memory and Computation

### Complexity
- **Time:** O(n² × d) where n = number of patches, d = hidden_dim
- **Space:** O(n² + n × d)

### Optimization Tips

1. **Larger Patches**: Reduce `n` by increasing `patch_size`
   - (10,10) → (20,20): 4× fewer patches, 16× faster attention

2. **Smaller Distance**: Reduce `dis` for sparser attention
   - dis=5 → dis=3: ~2× faster

3. **Fewer Layers**: Reduce `decoder_layers`
   - 3 → 2 layers: 1.5× faster

4. **Output Mode**: Use 'fused' instead of 'concat'
   - Reduces memory for subsequent layers

### Example: Lightweight Config
```python
sat_fusion=dict(
    bev_channels=256,
    sat_channels=256,
    hidden_c=256,           # Smaller
    img_size=(200, 400),
    patch_size=(20, 20),    # Larger patches
    dis=3,                  # Smaller distance
    decoder_layers=2,       # Fewer layers
    num_heads=4,            # Fewer heads
    output_mode='fused'     # Fused output
)
```

---

## Usage Examples

### Training
```python
# In your training script
python projects/tools/train.py \
    projects/configs/agfusion/agfusion_with_satellite_fusion.py \
    --work-dir work_dirs/agfusion_satellite_fusion
```

### Inference
```python
# Satellite fusion happens automatically
# Just ensure satellite_img is provided in the data dict
results = model(
    img=camera_images,
    satellite_img=satellite_images,
    img_metas=img_metas,
    return_loss=False
)
```

---

## Verification

### Check Integration
```python
# Test fusion module standalone
from projects.mmdet3d_plugin.agfusion.masked_t import Fusion_Atten_Masked

fusion = Fusion_Atten_Masked(
    bev_channels=256,
    sat_channels=256,
    hidden_c=384,
    img_size=(200, 400),
    patch_size=(10, 10)
)

# Test forward pass
import torch
bev_feat = torch.randn(2, 256, 200, 400)
sat_feat = torch.randn(2, 256, 200, 400)
fused = fusion(bev_feat, sat_feat)

print(f"Input BEV: {bev_feat.shape}")
print(f"Input Sat: {sat_feat.shape}")
print(f"Fused: {fused.shape}")
# Expected: torch.Size([2, 256, 200, 400])
```

### Check MapTR Integration
```python
# Verify fusion is initialized
from projects.mmdet3d_plugin.agfusion.detectors.maptr import MapTR

model = MapTR(...)
print(f"Satellite backbone: {model.sat_backbone}")
print(f"Satellite fusion: {model.sat_fusion}")
# Should show module instances
```

---

## Benefits

✅ **Multi-Modal Fusion**: Combines complementary BEV and satellite information  
✅ **Attention-Based**: Learns important spatial relationships  
✅ **Distance Masking**: Focuses on local context, prevents over-smoothing  
✅ **Flexible**: Works with or without satellite features  
✅ **Configurable**: Many parameters for customization  
✅ **Efficient**: Patch-based attention reduces computation  

---

## Next Steps

1. **Tune Hyperparameters**
   - Adjust `dis`, `patch_size`, `decoder_layers` for your data
   - Try both 'fused' and 'concat' output modes

2. **Visualization**
   - Visualize attention maps to understand what the model learns
   - Compare predictions with/without fusion

3. **Ablation Studies**
   - Measure impact of satellite features on performance
   - Test different fusion configurations

4. **Multi-Scale Fusion**
   - Extend to fuse multiple feature scales
   - Add skip connections between scales

---

## Troubleshooting

### Issue: Shape Mismatch
```
RuntimeError: The size of tensor a (200) must match the size of tensor b (400)
```
**Solution**: Check `img_size` matches actual BEV feature size

### Issue: CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solutions**:
- Increase `patch_size` (e.g., 10→20)
- Decrease `hidden_c` (e.g., 384→256)
- Reduce `decoder_layers` (e.g., 3→2)
- Use `output_mode='fused'` instead of 'concat'

### Issue: No Performance Improvement
**Check**:
- Satellite images are properly aligned to BEV
- Satellite features have meaningful information
- Try different `dis` values (3, 5, 7)
- Increase `decoder_layers` for more capacity

---

**Status: ✅ READY FOR TRAINING**

The satellite-BEV fusion module is fully integrated and ready to use!
