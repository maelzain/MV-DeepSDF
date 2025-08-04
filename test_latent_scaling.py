#!/usr/bin/env python3
"""
Test different latent scaling approaches to fix the similar shape issue
"""

import os, sys, json
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))
from data.dataset import MVDeepSDFDataset
from networks.mv_deepsdf import MVDeepSDF
from networks.deep_sdf_decoder import Decoder
from deep_sdf.mesh import create_mesh

def test_latent_scaling():
    print("🔧 Testing Latent Scaling Solutions")
    print("=" * 50)
    
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
    
    # Test first 3 instances with different scaling approaches
    os.makedirs('debug_meshes', exist_ok=True)
    
    with torch.no_grad():
        for i, batch in enumerate(dl):
            if i >= 3:  # Test first 3
                break
                
            iid = batch['instance_id'][0]
            pcs = batch['point_clouds'].to(device)
            pl = batch['partial_latents'].to(device)
            gt_latent = batch['gt_latent'].to(device)
            
            # Get predicted latent
            z_pred = model(pcs, pl)
            
            print(f"\n--- Instance {i+1}: {iid} ---")
            print(f"GT latent norm: {gt_latent.norm():.6f}")
            print(f"Predicted latent norm: {z_pred.norm():.6f}")
            
            # Calculate scaling factor based on GT norm
            gt_norm = gt_latent.norm()
            pred_norm = z_pred.norm()
            scale_factor = gt_norm / pred_norm if pred_norm > 0 else 1.0
            
            print(f"Scale factor: {scale_factor:.6f}")
            
            # Test different approaches
            approaches = {
                'original': z_pred,
                'scaled_by_norm': z_pred * scale_factor,
                'scaled_by_std': z_pred * (gt_latent.std() / z_pred.std()) if z_pred.std() > 0 else z_pred,
                'scaled_fixed_7': z_pred * 7.5,  # Based on our observation (7.5x smaller)
                'gt_latent': gt_latent  # Reference
            }
            
            for approach_name, latent in approaches.items():
                try:
                    mesh_path = f'debug_meshes/{iid}_{approach_name}.ply'
                    create_mesh(decoder, latent, mesh_path, N=256, max_batch=2**15)
                    
                    # Test SDF at center point
                    query_pt = torch.tensor([[0.0, 0.0, 0.0]]).float().to(device)
                    decoder_input = torch.cat([latent, query_pt], dim=1)
                    sdf_center = decoder(decoder_input).item()
                    
                    print(f"  {approach_name:>15}: norm={latent.norm():.4f}, SDF(0,0,0)={sdf_center:.6f}")
                    
                except Exception as e:
                    print(f"  {approach_name:>15}: FAILED - {str(e)}")
    
    print(f"\n✅ Generated test meshes in debug_meshes/ directory")
    print("🔍 Compare the meshes to see which scaling approach works best:")
    print("   - 'original': No scaling (current problematic approach)")
    print("   - 'scaled_by_norm': Scale to match GT latent norm")  
    print("   - 'scaled_by_std': Scale to match GT latent standard deviation")
    print("   - 'scaled_fixed_7': Scale by fixed factor 7.5")
    print("   - 'gt_latent': Ground truth reference")
    
    print("\n" + "=" * 50)
    print("🎯 Latent scaling test complete!")

if __name__ == '__main__':
    test_latent_scaling()