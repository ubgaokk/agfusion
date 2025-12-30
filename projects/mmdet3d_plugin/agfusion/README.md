# AGFusion - BEV-Satellite Map Fusion Module

This module implements the fusion of BEV (Bird's Eye View) features from onboard sensors with satellite map features, as inspired by the paper **"Complementing Onboard Sensors with Satellite Map: A New Perspective for HD Map Construction"**.

## Overview

The fusion module enhances HD map construction by combining:
- **BEV features**: Generated from multi-view camera images using onboard sensors
- **Satellite map features**: Pre-processed satellite imagery providing additional context

## Architecture

### 1. MapFusion Module (`map_fusion.py`)

The core fusion module that supports multiple fusion strategies:

#### Fusion Strategies

- **concat**: Simple concatenation followed by 1×1 convolution
  - Concatenates BEV and satellite features along channel dimension
  - Projects to output channels using convolution + normalization + activation

- **add**: Element-wise addition (requires same input channels)
  - Direct addition of aligned features
  - Optional projection to match output channels

- **attention**: Cross-attention based fusion
  - Computes attention weights to balance BEV and satellite contributions
  - Applies complementary attention (BEV gets α, satellite gets 1-α)
  - Fuses attended features

- **adaptive**: Learnable adaptive fusion weights
  - Learns spatial-adaptive weights for each feature stream
  - Uses softmax to ensure weights sum to 1
  - Weighted combination: `output = w1 * BEV + w2 * satellite`

#### Usage Example

```python
fusion = dict(
    type='MapFusion',
    in_channels=[256, 256],      # [bev_channels, satellite_channels]
    out_channels=256,             # Output channels
    num_levels=1,                 # Number of feature pyramid levels
    fusion_type='concat'          # Fusion strategy
)
```

### 2. MapTRPerceptionTransformer (`maptr_transformer.py`)

Extended transformer that integrates the fusion module into the perception pipeline:

- Extends `PerceptionTransformer` from BEVFormer
- Integrates `MapFusion` module to fuse BEV features with satellite features
- Supports optional satellite feature input (gracefully handles None)
- Maintains compatibility with standard BEVFormer pipeline

#### Key Features

- **Flexible Fusion**: Can work with or without satellite features
- **Shape Handling**: Automatically handles different tensor shapes (spatial vs. sequence format)
- **End-to-End**: Integrated into the full perception pipeline

## Configuration

Example configuration in `agfusion_tiny_r50_24e.py`:

```python
transformer=dict(
    type='MapTRPerceptionTransformer',
    rotate_prev_bev=True,
    use_shift=True,
    use_can_bus=True,
    embed_dims=256,
    encoder=dict(
        type='BEVFormerEncoder',
        num_layers=1,
        pc_range=point_cloud_range,
        num_points_in_pillar=4,
        return_intermediate=False,
        transformerlayers=dict(
            type='BEVFormerLayer',
            attn_cfgs=[...],
            feedforward_channels=512,
            ffn_dropout=0.1,
            operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')
        )
    ),
    fusion=dict(
        type='MapFusion',
        in_channels=[256, 256],   # BEV and satellite feature channels
        out_channels=256,
        num_levels=1,
        fusion_type='concat'      # Options: 'concat', 'add', 'attention', 'adaptive'
    ),
    decoder=dict(...)
)
```

## Implementation Details

### Input Requirements

1. **BEV Features**:
   - Shape: `(bs, embed_dims, bev_h, bev_w)` or `(num_query, bs, embed_dims)`
   - Generated from multi-view camera images via BEVFormer encoder

2. **Satellite Features** (optional):
   - Shape: `(bs, sat_channels, bev_h, bev_w)`
   - Should be aligned with BEV grid (same spatial dimensions)
   - If dimensions differ, automatic interpolation is applied

### Output

- Fused BEV features with shape matching the input BEV feature shape
- Maintains compatibility with downstream decoder

## Fusion Strategies Comparison

| Strategy   | Pros | Cons | When to Use |
|-----------|------|------|-------------|
| **concat** | Simple, effective | Increases channels | Default choice, general use |
| **add** | No channel increase | Requires same input channels | When memory is constrained |
| **attention** | Adaptive weighting | More parameters | When features have varying quality |
| **adaptive** | Spatially adaptive | Most parameters, slower | When local adaptation is critical |

## Paper Reference

This implementation is inspired by:

```
"Complementing Onboard Sensors with Satellite Map: A New Perspective for HD Map Construction"
```

The paper proposes using satellite maps as complementary information to onboard sensor data for more accurate and complete HD map construction.

### 3. MapTRHead (`maptr_head.py`)

Detection head for vectorized HD map construction:

- Extends `DETRHead` from MMDetection
- Predicts vectorized map elements as sequences of points
- Supports multiple map classes (dividers, crossings, boundaries)
- Integrated loss functions for classification, points, and direction

#### Key Features

- **Vectorized Prediction**: Predicts map elements as ordered point sequences
- **Multi-Loss Training**: Combines classification, bbox, points, and direction losses
- **Flexible Architecture**: Supports both with and without box refinement
- **Compatibility**: Works seamlessly with MapTRPerceptionTransformer

## File Structure

```
projects/mmdet3d_plugin/agfusion/
├── __init__.py                  # Module exports
├── map_fusion.py               # Core fusion module
├── maptr_transformer.py        # Extended transformer with fusion
├── maptr_head.py               # MapTR detection head
└── README.md                   # This file
```

## Training Notes

1. **Satellite Feature Extraction**: 
   - Satellite features should be pre-processed to match BEV resolution
   - Consider using a separate backbone for satellite imagery

2. **Fusion Timing**:
   - Fusion occurs after BEV encoder, before decoder
   - This allows the decoder to work with enriched features

3. **Ablation Studies**:
   - Test different fusion strategies for your specific dataset
   - `concat` is recommended as a baseline
   - `attention` or `adaptive` may work better with noisy satellite data

## Future Extensions

Potential improvements:
- Multi-scale fusion at different feature pyramid levels
- Temporal fusion of satellite and BEV features across frames
- Learnable alignment between BEV and satellite coordinate systems
- Uncertainty-aware fusion weights based on satellite image quality
