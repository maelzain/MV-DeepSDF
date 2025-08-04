#!/usr/bin/env python3
"""
Test the retrained model with corrected latent predictor
"""

import os, sys, json
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))
from data.dataset import MVDeepSDFDataset
from networks.mv_deepsdf import MVDeepSDF

def test_fixed_model():
    print("🔧 Testing Retrained Model (No Tanh)")
    print("=" * 40)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load config
    with open('configs/mvdeepsdf_config.json') as f:
        cfg = json.load(f)
    
    # Load FIXED model (retrained without Tanh)
    print("Loading FIXED model (retrained without Tanh)...")
    fixed_model = MVDeepSDF(
        num_points=cfg['stage2']['num_points'],
        global_feature_dim=cfg['stage2']['global_feature_dim'],
        latent_dim=cfg['stage2']['latent_dim'],
        element_feature_dim=cfg['stage2']['element_feature_dim'],
        num_views=cfg['stage2']['num_views']
    ).to(device)
    
    # Try latest checkpoint
    try:
        ck = torch.load('experiments/mvdeepsdf_stage2_fixed_latent_scale/checkpoints/epoch_17.pth', map_location=device)
        epoch_loaded = 17
    except:
        try:
            ck = torch.load('experiments/mvdeepsdf_stage2_fixed_latent_scale/checkpoints/epoch_16.pth', map_location=device)
            epoch_loaded = 16
        except:
            ck = torch.load('experiments/mvdeepsdf_stage2_fixed_latent_scale/checkpoints/epoch_15.pth', map_location=device)
            epoch_loaded = 15
            
    fixed_model.load_state_dict(ck['model_state_dict'])
    fixed_model.eval()
    print(f"✅ Loaded checkpoint from epoch {epoch_loaded}")
    
    # Load OLD model for comparison
    print("Loading OLD model (with Tanh) for comparison...")
    old_model = MVDeepSDF(
        num_points=cfg['stage2']['num_points'],
        global_feature_dim=cfg['stage2']['global_feature_dim'],
        latent_dim=cfg['stage2']['latent_dim'],
        element_feature_dim=cfg['stage2']['element_feature_dim'],
        num_views=cfg['stage2']['num_views']
    ).to(device)
    
    old_ck = torch.load('experiments/mvdeepsdf_stage2_proper_split/checkpoints/epoch_19.pth', map_location=device)
    old_model.load_state_dict(old_ck['model_state_dict'])
    old_model.eval()
    
    # Load dataset
    ds = MVDeepSDFDataset(
        data_root='experiments/multisweep_data_final',
        split_file='configs/mvdeepsdf_stage2_test.json',
        gt_latent_path='examples/cars/LatentCodes/latest.pth',
        num_views=cfg['stage2']['num_views']
    )
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    
    print("\n🧪 Testing on first 3 instances...")
    
    with torch.no_grad():
        for i, batch in enumerate(dl):
            if i >= 3:  # Test first 3
                break
                
            iid = batch['instance_id'][0]
            pcs = batch['point_clouds'].to(device)
            pl = batch['partial_latents'].to(device)
            gt_latent = batch['gt_latent'].to(device)
            
            # Get predictions from both models
            old_pred = old_model(pcs, pl)
            fixed_pred = fixed_model(pcs, pl)
            
            print(f"\n--- Instance {i+1}: {iid} ---")
            print(f"GT latent norm:    {gt_latent.norm():.6f}")
            print(f"OLD model norm:    {old_pred.norm():.6f}")
            print(f"FIXED model norm:  {fixed_pred.norm():.6f}")
            
            # Calculate improvements
            old_diff = abs(gt_latent.norm() - old_pred.norm())
            fixed_diff = abs(gt_latent.norm() - fixed_pred.norm())
            
            print(f"Distance to GT norm:")
            print(f"  OLD model:   {old_diff:.6f}")
            print(f"  FIXED model: {fixed_diff:.6f}")
            
            improvement = old_diff - fixed_diff
            if improvement > 0:
                print(f"✅ IMPROVEMENT: {improvement:.6f} closer to GT!")
            else:
                print(f"❌ Still issues: {abs(improvement):.6f} further from GT")
                
            # Show latent ranges
            print(f"OLD latent range:   [{old_pred.min():.4f}, {old_pred.max():.4f}]")
            print(f"FIXED latent range: [{fixed_pred.min():.4f}, {fixed_pred.max():.4f}]")
            print(f"GT latent range:    [{gt_latent.min():.4f}, {gt_latent.max():.4f}]")
    
    print(f"\n🎯 Fixed model test complete!")
    print("Expected result: FIXED model should have latent norms much closer to GT")

if __name__ == '__main__':
    test_fixed_model()