#!/usr/bin/env python3
"""
Comprehensive verification script for satellite image integration in MapTR detector.
"""

import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def check_maptr_detector():
    """Verify MapTR detector satellite integration."""
    print("=" * 80)
    print("CHECKING MAPTR DETECTOR SATELLITE INTEGRATION")
    print("=" * 80)
    
    maptr_file = "projects/mmdet3d_plugin/agfusion/detectors/maptr.py"
    
    if not os.path.exists(maptr_file):
        print(f"❌ File not found: {maptr_file}")
        return False
    
    with open(maptr_file, 'r') as f:
        content = f.read()
    
    checks = {
        "Import builder": "from mmdet3d.models import builder" in content,
        "sat_backbone parameter in __init__": "sat_backbone=None" in content,
        "Satellite encoder building": "self.sat_backbone = builder.build_backbone(sat_backbone)" in content,
        "Null check for sat_backbone": "if sat_backbone is not None:" in content,
        "extract_sat_feat method": "def extract_sat_feat(self, sat_img" in content,
        "Handle temporal dimension": "if sat_img.dim() == 5:" in content,
        "Return None if no satellite": "if sat_img is None or self.sat_backbone is None:" in content,
        "satellite_img in forward_train": "satellite_img=None" in content and "def forward_train(" in content,
        "Extract sat feats in forward_train": "sat_feats = self.extract_sat_feat(satellite_img" in content,
        "Store sat_feats in img_metas": "meta['sat_feats'] = sat_feats" in content,
        "satellite_img in simple_test": "def simple_test(self, img_metas, img=None, points=None, prev_bev=None, rescale=False, satellite_img=None" in content,
        "Extract sat feats in simple_test": content.count("sat_feats = self.extract_sat_feat") >= 2,
    }
    
    all_passed = True
    for check_name, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"{status} {check_name}")
        if not passed:
            all_passed = False
    
    print()
    return all_passed

def check_dataset_integration():
    """Verify dataset has satellite support."""
    print("=" * 80)
    print("CHECKING DATASET SATELLITE INTEGRATION")
    print("=" * 80)
    
    dataset_file = "projects/mmdet3d_plugin/datasets/nuscenes_map_dataset.py"
    
    if not os.path.exists(dataset_file):
        print(f"❌ File not found: {dataset_file}")
        return False
    
    with open(dataset_file, 'r') as f:
        content = f.read()
    
    checks = {
        "PriorMap import": "from .map_utils.prior_map import PriorMap" in content,
        "PriorMap initialization": "self.prior_map = PriorMap(" in content,
        "get_satellite_image method": "def get_satellite_image(self" in content,
        "ego_pose_token handling": "ego_pose_token" in content,
    }
    
    all_passed = True
    for check_name, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"{status} {check_name}")
        if not passed:
            all_passed = False
    
    print()
    return all_passed

def check_module_registration():
    """Verify MapTR is properly registered."""
    print("=" * 80)
    print("CHECKING MODULE REGISTRATION")
    print("=" * 80)
    
    try:
        from mmdet.models import DETECTORS
        from projects.mmdet3d_plugin.agfusion import MapTR
        
        print("✓ MapTR imported successfully")
        print(f"✓ MapTR class: {MapTR}")
        
        # Check if registered
        if 'MapTR' in DETECTORS.module_dict:
            print("✓ MapTR registered in DETECTORS")
            return True
        else:
            print("✗ MapTR NOT registered in DETECTORS")
            print(f"  Available detectors: {list(DETECTORS.module_dict.keys())[:10]}...")
            return False
            
    except Exception as e:
        print(f"✗ Error importing MapTR: {e}")
        return False

def check_pickle_file():
    """Verify the pickle file with ego_pose_token exists."""
    print("=" * 80)
    print("CHECKING PICKLE FILE WITH EGO_POSE_TOKEN")
    print("=" * 80)
    
    pickle_files = [
        "data/infos/nuscenes_infos_temporal_train_with_ego_pose.pkl",
        "data/infos/nuscenes_infos_train_with_ego_pose.pkl",
    ]
    
    found = False
    for pickle_file in pickle_files:
        if os.path.exists(pickle_file):
            print(f"✓ Found: {pickle_file}")
            size_mb = os.path.getsize(pickle_file) / (1024 * 1024)
            print(f"  Size: {size_mb:.2f} MB")
            found = True
        else:
            print(f"✗ Not found: {pickle_file}")
    
    print()
    return found

def summary_report(results):
    """Print summary report."""
    print("=" * 80)
    print("SUMMARY REPORT")
    print("=" * 80)
    
    total = len(results)
    passed = sum(results.values())
    
    for check_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {check_name}")
    
    print()
    print(f"Overall: {passed}/{total} checks passed")
    
    if passed == total:
        print("✓ ALL CHECKS PASSED - Satellite integration is complete!")
        print("\nNext steps:")
        print("1. Update your training config to include sat_backbone configuration")
        print("2. Run training with the satellite-enabled dataset")
        print("3. Monitor satellite feature flow through the model")
    else:
        print("✗ SOME CHECKS FAILED - Please review the output above")
    
    print("=" * 80)

if __name__ == "__main__":
    results = {
        "MapTR Detector": check_maptr_detector(),
        "Dataset Integration": check_dataset_integration(),
        "Module Registration": check_module_registration(),
        "Pickle File": check_pickle_file(),
    }
    
    summary_report(results)
