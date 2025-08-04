#!/usr/bin/env python3
"""
Quick test of latent scaling approaches using SDF evaluation only
"""

import os, sys, json
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))
from data.dataset import MVDeepSDFDataset
from networks.mv_deepsdf import MVDeepSDF
from networks.deep_sdf_decoder import Decoder

def quick_scaling_test():
    print("⚡ Quick Latent Scaling Test")
    print("=" * 40)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load config
    with open('configs/mvdeepsdf_config.json') as f:
        cfg = json.load(f)
    
    # Load models
    print("Loading models...")
    model = MVDeepSDF(
        num_points=cfg['stage2']['num_points'],
        global_feature_dim=cfg['stage2']['global_feature_dim'],
        latent_dim=cfg['stage2']['latent_dim'],
        element_feature_dim=cfg['stage2']['element_feature_dim'],
        num_views=cfg['stage2']['num_views']
    ).to(device)
    
    ck = torch.load('experiments/mvdeepsdf_stage2_proper_split/checkpoints/epoch_19.pth', map_location=device)
    model.load_state_dict(ck['model_state_dict'])
    model.eval()
    
    s1 = cfg['stage1']
    decoder = Decoder(
        latent_size=s1['latent_dim'],
        dims=s1['decoder_dims'],
        dropout=s1.get('dropout', None),
        dropout_prob=s1.get('dropout_prob', 0.0),
        norm_layers=s1.get('norm_layers', ()),
        latent_in=s1.get('latent_in', ()),
        weight_norm=s1.get('weight_norm', False),
        xyz_in_all=s1.get('xyz_in_all', None),
        use_tanh=s1.get('use_tanh', False),
        latent_dropout=s1.get('latent_dropout', False)
    ).to(device)
    
    dck = torch.load('examples/cars/ModelParameters/latest.pth', map_location=device)['model_state_dict']
    new_sd = {k.replace('module.', ''): v for k, v in dck.items()}
    decoder.load_state_dict(new_sd)
    decoder.eval()
    
    # Load dataset
    ds = MVDeepSDFDataset(
        data_root='experiments/multisweep_data_final',
        split_file='configs/mvdeepsdf_stage2_test.json',
        gt_latent_path='examples/cars/LatentCodes/latest.pth',
        num_views=cfg['stage2']['num_views']
    )
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    
    # Test points for SDF evaluation
    test_points = torch.tensor([
        [0.0, 0.0, 0.0],   # Center
        [0.2, 0.0, 0.0],   # X edge
        [0.0, 0.2, 0.0],   # Y edge
        [0.1, 0.1, 0.1],   # Corner
    ]).float().to(device)
    
    results = []
    
    with torch.no_grad():
        for i, batch in enumerate(dl):
            if i >= 5:  # Test first 5 instances
                break
                
            iid = batch['instance_id'][0]
            pcs = batch['point_clouds'].to(device)
            pl = batch['partial_latents'].to(device)
            gt_latent = batch['gt_latent'].to(device)
            
            # Get predicted latent
            z_pred = model(pcs, pl)
            
            # Calculate different scaling approaches
            gt_norm = gt_latent.norm()
            pred_norm = z_pred.norm()
            scale_factor = gt_norm / pred_norm if pred_norm > 0 else 1.0
            
            approaches = {
                'original': z_pred,
                'norm_scaled': z_pred * scale_factor,
                'fixed_7x': z_pred * 7.5,
                'gt': gt_latent
            }
            
            instance_results = {'id': iid, 'approaches': {}}
            
            for approach_name, latent in approaches.items():
                sdf_values = []
                for point in test_points:
                    decoder_input = torch.cat([latent, point.unsqueeze(0)], dim=1)
                    sdf = decoder(decoder_input).item()
                    sdf_values.append(sdf)
                
                instance_results['approaches'][approach_name] = {
                    'norm': latent.norm().item(),
                    'sdf_values': sdf_values
                }
            
            results.append(instance_results)
            
            print(f"\nInstance {i+1}: {iid}")
            print(f"  GT norm: {gt_norm:.4f}")
            print(f"  Pred norm: {pred_norm:.4f}")
            print(f"  Scale factor: {scale_factor:.4f}")
            
            for approach_name in approaches.keys():
                norm = instance_results['approaches'][approach_name]['norm']
                sdfs = instance_results['approaches'][approach_name]['sdf_values']
                print(f"  {approach_name:>12}: norm={norm:6.4f}, SDFs={[f'{s:.4f}' for s in sdfs]}")
    
    print("\n" + "=" * 40)
    print("📊 Analysis")
    print("-" * 40)
    
    # Calculate variation statistics
    for approach in ['original', 'norm_scaled', 'fixed_7x', 'gt']:
        all_sdfs = []
        for result in results:
            all_sdfs.extend(result['approaches'][approach]['sdf_values'])
        
        variation = np.std(all_sdfs)
        mean_sdf = np.mean(all_sdfs)
        
        print(f"{approach:>12}: std={variation:.6f}, mean={mean_sdf:.6f}")
    
    print("\n🔍 Recommendations:")
    print("- 'original': Current approach (low variation = similar shapes)")
    print("- 'norm_scaled': Scale by GT norm (should increase variation)")
    print("- 'fixed_7x': Scale by observed factor (simple fix)")
    print("- 'gt': Ground truth reference (maximum variation)")
    
    print("\n🎯 Quick scaling test complete!")

if __name__ == '__main__':
    quick_scaling_test()