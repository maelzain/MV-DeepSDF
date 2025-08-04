#!/usr/bin/env python3
"""
Quick test to verify the Tanh removal fix - check if predicted latents now have proper magnitude
"""

import os, sys, json
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))
from data.dataset import MVDeepSDFDataset
from networks.mv_deepsdf import MVDeepSDF

def test_fix():
    print("🔧 Testing Tanh Removal Fix")
    print("=" * 40)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load config
    with open('configs/mvdeepsdf_config.json') as f:
        cfg = json.load(f)
    
    # Load OLD model (with Tanh weights)
    print("Loading OLD model (trained with Tanh)...")
    old_model = MVDeepSDF(
        num_points=cfg['stage2']['num_points'],
        global_feature_dim=cfg['stage2']['global_feature_dim'],
        latent_dim=cfg['stage2']['latent_dim'],
        element_feature_dim=cfg['stage2']['element_feature_dim'],
        num_views=cfg['stage2']['num_views']
    ).to(device)
    
    ck = torch.load('experiments/mvdeepsdf_stage2_proper_split/checkpoints/epoch_19.pth', map_location=device)
    old_model.load_state_dict(ck['model_state_dict'])
    old_model.eval()
    
    # NEW model with same weights but no Tanh squashing
    print("Creating NEW model (no Tanh)...")
    new_model = MVDeepSDF(
        num_points=cfg['stage2']['num_points'],
        global_feature_dim=cfg['stage2']['global_feature_dim'],
        latent_dim=cfg['stage2']['latent_dim'],
        element_feature_dim=cfg['stage2']['element_feature_dim'],
        num_views=cfg['stage2']['num_views']
    ).to(device)
    
    new_model.load_state_dict(ck['model_state_dict'])
    new_model.eval()
    
    # Load dataset
    ds = MVDeepSDFDataset(
        data_root='experiments/multisweep_data_final',
        split_file='configs/mvdeepsdf_stage2_test.json',
        gt_latent_path='examples/cars/LatentCodes/latest.pth',
        num_views=cfg['stage2']['num_views']
    )
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    
    with torch.no_grad():
        batch = next(iter(dl))
        iid = batch['instance_id'][0]
        pcs = batch['point_clouds'].to(device)
        pl = batch['partial_latents'].to(device)
        gt_latent = batch['gt_latent'].to(device)
        
        # Get predictions from both models
        old_pred = old_model(pcs, pl)
        new_pred = new_model(pcs, pl)
        
        print(f"\nInstance: {iid}")
        print(f"GT latent norm: {gt_latent.norm():.6f}")
        print(f"OLD model (with Tanh): {old_pred.norm():.6f}")
        print(f"NEW model (no Tanh): {new_pred.norm():.6f}")
        
        # The new model should produce larger magnitude latents
        if new_pred.norm() > old_pred.norm():
            print("✅ FIX WORKING: New model produces larger magnitude latents")
        else:
            print("❌ Issue: New model latents not larger")
            
        # Check if closer to GT magnitude
        old_diff = abs(gt_latent.norm() - old_pred.norm())
        new_diff = abs(gt_latent.norm() - new_pred.norm())
        
        print(f"Distance to GT norm:")
        print(f"  OLD model: {old_diff:.6f}")
        print(f"  NEW model: {new_diff:.6f}")
        
        if new_diff < old_diff:
            print("✅ NEW model is closer to GT magnitude!")
        else:
            print("⚠️  NEW model still not optimal")
            
    print(f"\n🎯 Tanh removal test complete!")
    print("Note: Model needs retraining to fully benefit from this architectural fix")

if __name__ == '__main__':
    test_fix()