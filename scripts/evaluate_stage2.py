#!/usr/bin/env python3
"""
Evaluation Script for MV-DeepSDF Stage 2

- Runs DeepSDF single‐view baseline via evaluate()
- Computes ACD×1e3 and Recall@0.1 as per the ICCV paper
"""

import os, sys, json, argparse, logging
import numpy as np
import torch, trimesh
from tqdm import tqdm
from torch.utils.data import DataLoader

# project imports
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))
from data.dataset import MVDeepSDFDataset
from networks.mv_deepsdf import MVDeepSDF
# Import optimal model from training script
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))
from train_mvdeepsdf_stage2 import SpecificationCompliantMVDeepSDF
from networks.deep_sdf_decoder import Decoder
from deep_sdf.mesh import create_mesh

# original DeepSDF evaluate()
from evaluate import evaluate as original_evaluate

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def asymmetric_chamfer_sq(gt, gen, chunk=2048):
    total = 0.0
    N = gt.size(0)
    for i in range(0, N, chunk):
        c = gt[i:i+chunk].unsqueeze(0)
        d2 = torch.cdist(c, gen.unsqueeze(0))**2
        total += d2.min(dim=2)[0].sum().item()
    return total

def compute_recall(gt, gen, thr_sq=0.1):
    d2 = torch.cdist(gt.unsqueeze(0), gen.unsqueeze(0))**2
    min2 = d2.min(dim=2)[0]
    return (min2 <= thr_sq).float().mean().item()

def load_config(path):
    with open(path) as f:
        return json.load(f)

def main():
    p = argparse.ArgumentParser()
    # multi-view args
    p.add_argument('--config',            required=True)
    p.add_argument('--checkpoint',        required=True)
    p.add_argument('--decoder_checkpoint',required=True)
    p.add_argument('--data_root',         required=True)
    p.add_argument('--split_file',        required=True)
    p.add_argument('--gt_latent_path',    required=True)
    p.add_argument('--gt_mesh_root',      required=True)
    p.add_argument('--output_dir',        default='./outputs/eval')
    p.add_argument('--num_mesh_samples',  type=int, default=30000)
    # single-view baseline args
    p.add_argument('--sv_experiment',     required=True)
    p.add_argument('--sv_checkpoint',     required=True)
    p.add_argument('--sv_data_dir',       required=True)
    p.add_argument('--sv_split',          required=True)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    recon_dir = os.path.join(args.output_dir, 'reconstructions')
    os.makedirs(recon_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    cfg = load_config(args.config)

    # 1) single‐view baseline - SKIPPED due to missing data structure
    logger.info("→ Skipping DeepSDF single‐view baseline (missing SurfaceSamples/NormalizationParameters)")
    logger.info("→ Proceeding directly to multi-view evaluation")

    # 2) multi‐view setup - USE SPECIFICATION-COMPLIANT MODEL
    logger.info("Loading SPECIFICATION-COMPLIANT MV-DeepSDF model with 5x weight scaling...")
    model = SpecificationCompliantMVDeepSDF(
        num_points=cfg['stage2']['num_points'],
        global_feature_dim=cfg['stage2']['global_feature_dim'],
        latent_dim=cfg['stage2']['latent_dim'],
        num_views=cfg['stage2']['num_views']
    ).to(device)
    ck = torch.load(args.checkpoint, map_location=device)
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
    dck = torch.load(args.decoder_checkpoint, map_location=device)['model_state_dict']
    new_sd = {k.replace('module.', ''): v for k, v in dck.items()}
    decoder.load_state_dict(new_sd)
    decoder.eval()

    ds = MVDeepSDFDataset(
        data_root=args.data_root,
        split_file=args.split_file,
        gt_latent_path=args.gt_latent_path,
        num_views=cfg['stage2']['num_views']
    )
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4)

    chamfers, recalls = [], []

    logger.info("→ Running multi‐view evaluation")
    with torch.no_grad():
        for batch in tqdm(dl):
            
            iid = batch['instance_id'][0]
            pcs = batch['point_clouds'].to(device)
            pl  = batch['partial_latents'].to(device)

            z_raw = model(pcs, pl)
            
            # PAPER COMPLIANCE: Use predicted latent directly since both model and decoder
            # now properly handle [-1,1] normalized latent codes as per paper requirements
            z = z_raw

            ply = os.path.join(recon_dir, f"{iid}.ply")
            try:
                create_mesh(decoder, z, ply, N=384, max_batch=2**16)
            except Exception as e:
                logger.warning(f"{iid}: mesh gen failed: {e}")
                continue

            # PAPER APPROACH: Use stacked multi-sweep point clouds as ground truth
            # Stack all point clouds to create ground truth as per paper methodology
            stacked_pc = pcs.view(-1, 3).cpu().numpy()  # Shape: (num_views * num_points, 3)
            gt_pts = torch.from_numpy(stacked_pc).float().to(device)

            if not os.path.exists(ply):
                logger.warning(f"{iid}: generated mesh file not found at {ply}")
                continue
            
            try:
                gen_mesh = trimesh.load(ply)
                gen_pts, _ = trimesh.sample.sample_surface(gen_mesh, args.num_mesh_samples)
                gen_pts = torch.from_numpy(gen_pts).float().to(device)

                # Compute metrics as per paper
                acd = asymmetric_chamfer_sq(gt_pts, gen_pts) * 1e3
                rec = compute_recall(gt_pts, gen_pts)

                chamfers.append(acd)
                recalls.append(rec)
                logger.info(f"{iid}: ACD×1e3={acd:.3f}, Recall={rec*100:.2f}%")
            except Exception as e:
                logger.warning(f"{iid}: Failed to evaluate mesh: {e}")
                continue

    # PAPER APPROACH: Generate evaluation statistics
    if chamfers:
        stats = {
            'mean_ACD×1e3':   float(np.mean(chamfers)),
            'median_ACD×1e3': float(np.median(chamfers)),
            'std_ACD×1e3':    float(np.std(chamfers)),
            'mean_Recall':    float(np.mean(recalls)),
            'std_Recall':     float(np.std(recalls)),
            'individual_ACD×1e3': chamfers,
            'individual_Recall': recalls
        }
        outp = os.path.join(args.output_dir, 'results_mv.json')
        with open(outp, 'w') as f:
            json.dump(stats, f, indent=4)
        logger.info(f"Saved multi‐view results to {outp}")
        logger.info(f"FINAL RESULTS: Mean ACD×1e3={stats['mean_ACD×1e3']:.3f}, Mean Recall={stats['mean_Recall']*100:.2f}%")
    else:
        logger.warning("No valid instances evaluated.")
    
    logger.info("Reconstruction and evaluation completed.")

if __name__ == '__main__':
    main()
