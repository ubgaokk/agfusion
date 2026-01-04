# Training Guide - AGFusion with Satellite Images

## Issue Fixed: ModuleNotFoundError for 'projects'

### Problem
When running `train.py`, you encountered:
```
ModuleNotFoundError: No module named 'projects'
```

### Solution
Updated `projects/tools/train.py` to automatically add the project root to Python's sys.path:
```python
# Add the project root directory to Python path
project_root = osp.abspath(osp.join(osp.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
```

### Verification
Run the test script to verify the fix:
```bash
cd /home/kanke/Documents/new_github/ai_test/agfusion
python test_imports.py
```

Expected output: ✓ All checks pass

---

## How to Run Training

### Prerequisites

1. **Activate your conda environment:**
```bash
conda activate maptr
```

2. **Navigate to project root:**
```bash
cd /home/kanke/Documents/new_github/ai_test/agfusion
```

3. **Verify data and config paths:**
- NuScenes data: `/media/kanke/easystore/Lambda/Data_set/data_nuscences/nuscenes/`
- Pickle file: `nuscenes_infos_temporal_train_with_ego_pose.pkl`
- Config file: `projects/configs/agfusion/agfusion_tiny_r50_24e.py`

### Single GPU Training

```bash
python projects/tools/train.py \
    projects/configs/agfusion/agfusion_tiny_r50_24e.py \
    --work-dir ./work_dirs/agfusion_tiny
```

### Multi-GPU Training (Recommended)

```bash
# Using 2 GPUs
bash projects/tools/dist_train.sh \
    projects/configs/agfusion/agfusion_tiny_r50_24e.py \
    2 \
    --work-dir ./work_dirs/agfusion_tiny

# Using 4 GPUs
bash projects/tools/dist_train.sh \
    projects/configs/agfusion/agfusion_tiny_r50_24e.py \
    4 \
    --work-dir ./work_dirs/agfusion_tiny
```

### Training with Satellite Images

To train with satellite image integration:

1. **Update config file** to use satellite images:
```bash
# Use the satellite-enabled config
python projects/tools/train.py \
    projects/configs/agfusion/agfusion_with_satellite.py \
    --work-dir ./work_dirs/agfusion_satellite
```

2. **Make sure to update paths in config:**
```python
# In agfusion_with_satellite.py
satellite_dir = '/path/to/your/satellite/prior_map_trainval'
ann_file = 'data/nuscenes/nuscenes_infos_temporal_train_with_ego_pose.pkl'
```

### Resume Training

```bash
python projects/tools/train.py \
    projects/configs/agfusion/agfusion_tiny_r50_24e.py \
    --work-dir ./work_dirs/agfusion_tiny \
    --resume-from ./work_dirs/agfusion_tiny/latest.pth
```

### Training Options

Common command-line options:

- `--work-dir`: Directory to save logs and models
- `--resume-from`: Resume from a checkpoint
- `--no-validate`: Skip validation during training
- `--gpus`: Number of GPUs (non-distributed)
- `--gpu-ids`: Specific GPU IDs to use
- `--seed`: Random seed
- `--deterministic`: Use deterministic mode

Example with options:
```bash
python projects/tools/train.py \
    projects/configs/agfusion/agfusion_tiny_r50_24e.py \
    --work-dir ./work_dirs/agfusion_tiny \
    --seed 0 \
    --deterministic
```

---

## Debugging Training Issues

### Check CUDA availability:
```python
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}')"
```

### Verify config file:
```bash
python -c "from mmcv import Config; cfg = Config.fromfile('projects/configs/agfusion/agfusion_tiny_r50_24e.py'); print('Config loaded successfully')"
```

### Test data loading:
```bash
cd debug
python test_satellite_integration.py
```

### Check GPU memory:
```bash
nvidia-smi
```

---

## Project Structure

```
agfusion/
├── projects/
│   ├── configs/
│   │   └── agfusion/
│   │       ├── agfusion_tiny_r50_24e.py
│   │       └── agfusion_with_satellite.py
│   ├── mmdet3d_plugin/
│   │   ├── datasets/
│   │   │   ├── nuscenes_map_dataset.py  (✓ Satellite integrated)
│   │   │   └── prior_map.py
│   │   └── bevformer/
│   │       └── apis/
│   │           └── train.py
│   └── tools/
│       ├── train.py  (✓ Fixed import issue)
│       └── dist_train.sh
├── work_dirs/  (Created during training)
├── debug/
│   ├── test_imports.py  (✓ Fixed)
│   ├── test_satellite_integration.py
│   ├── check_prior_map.py
│   └── SATELLITE_INTEGRATION.md
└── data/  (Symlink or actual data)
    └── nuscenes/
        ├── nuscenes_infos_temporal_train_with_ego_pose.pkl  (✓ Created)
        └── ...
```

---

## Expected Training Output

When training starts successfully, you should see:
```
2026-01-04 XX:XX:XX,XXX - INFO - Environment info:
...
2026-01-04 XX:XX:XX,XXX - INFO - Config:
...
2026-01-04 XX:XX:XX,XXX - INFO - Start running, host: ...
2026-01-04 XX:XX:XX,XXX - INFO - Workflow: [('train', 1)]
2026-01-04 XX:XX:XX,XXX - INFO - Epoch [1][1/XXXX] ...
```

---

## Monitoring Training

### TensorBoard (if configured):
```bash
tensorboard --logdir ./work_dirs/agfusion_tiny
```

### Check logs:
```bash
tail -f ./work_dirs/agfusion_tiny/*.log
```

### Check latest checkpoint:
```bash
ls -lh ./work_dirs/agfusion_tiny/*.pth
```

---

## Common Issues and Solutions

### Issue: Out of Memory (OOM)
**Solution:** Reduce batch size in config:
```python
data = dict(samples_per_gpu=1)  # Reduce from default
```

### Issue: Dataset not found
**Solution:** Check data paths in config match actual data location

### Issue: CUDA version mismatch
**Solution:** Reinstall PyTorch matching your CUDA version:
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### Issue: Config file errors
**Solution:** Validate config syntax:
```bash
python -c "from mmcv import Config; Config.fromfile('your_config.py')"
```

---

## Quick Start Checklist

- [ ] Conda environment activated: `conda activate maptr`
- [ ] In project root: `cd /home/kanke/Documents/new_github/ai_test/agfusion`
- [ ] test_imports.py passes: `python test_imports.py`
- [ ] Data paths verified in config
- [ ] GPU available: `nvidia-smi`
- [ ] Run training command

---

## Contact & References

- Original AGFusion paper: "Complementing Onboard Sensors with Satellite Map"
- MMDetection3D: https://github.com/open-mmlab/mmdetection3d
- NuScenes Dataset: https://www.nuscenes.org/

For issues related to satellite integration, see:
- `debug/SATELLITE_INTEGRATION.md`
- `debug/test_satellite_integration.py`
