#!/usr/bin/env python3
"""
Stage 2 Training Script for MV-DeepSDF - Refactored

This script trains the MV-DeepSDF model using pre-computed partial point clouds
and their corresponding latent codes, as well as the ground truth latent codes
from a pre-trained DeepSDF model (Stage 1).
"""

import os
import sys
import json
import argparse
import logging
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

# Add project root to path
# This assumes the script is run from the root of the project where 'data' and 'networks' are.
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from data.dataset import MVDeepSDFDataset
from networks.mv_deepsdf import MVDeepSDF
import torch.nn as nn

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SpecificationCompliantMVDeepSDF(MVDeepSDF):
    """100% Specification-Compliant MV-DeepSDF with 5x weight scaling"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Apply 5x weight scaling to the specification-compliant predictor
        # The predictor now correctly uses input_dim=1280 from the updated architecture
        with torch.no_grad():
            self.predictor.fc.weight.data *= 5.0
            if self.predictor.fc.bias is not None:
                self.predictor.fc.bias.data *= 5.0

def load_config(config_path: str) -> dict:
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def create_directories(output_dir: str):
    """Create necessary directories"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(output_dir, 'checkpoints')).mkdir(exist_ok=True)
    Path(os.path.join(output_dir, 'logs')).mkdir(exist_ok=True)
    
def save_checkpoint(epoch: int, model: torch.nn.Module, optimizer: torch.optim.Optimizer, path: str):
    """Save model checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)
    logger.info(f"Saved checkpoint to {path}")

def main():
    parser = argparse.ArgumentParser(description='Train MV-DeepSDF Stage 2')
    parser.add_argument('--config', type=str, default='configs/mvdeepsdf_config.json',
                        help='Path to configuration file')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save checkpoints and logs. Overrides config file.')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to a checkpoint to resume training from')
    parser.add_argument('--smoke-run', action='store_true', 
                        help='Run a quick smoke test for one epoch on a few batches.')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    logger.info(f"Loaded configuration: {config}")

    if args.smoke_run:
        config['training']['num_epochs'] = 1
        logger.info("Smoke run enabled: Overriding num_epochs to 1.")

    # Determine output directory
    output_dir = args.output_dir if args.output_dir else config['training'].get('output_dir', './outputs/stage2')

    # Create output directories
    create_directories(output_dir)

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- Dataset and DataLoader ---
    logger.info("Creating dataset...")
    train_dataset = MVDeepSDFDataset(
        data_root=config['stage2']['data_root'],
        split_file=config['stage2']['split_file'],
        gt_latent_path=config['stage2']['gt_latent_path'],
        num_views=config['stage2']['num_views'],
        num_points=config['stage2']['num_points']
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=0,  # Disable multiprocessing to avoid race conditions
        pin_memory=True,
        drop_last=True  # Drop the last batch if it has only 1 sample to avoid BatchNorm issues
    )

    # --- Model ---
    logger.info("Initializing SPECIFICATION-COMPLIANT MV-DeepSDF model with 5x weight scaling...")
    model = SpecificationCompliantMVDeepSDF(
        num_points=config['stage2']['num_points'],
        global_feature_dim=config['stage2']['global_feature_dim'],
        latent_dim=config['stage2']['latent_dim'],
        num_views=config['stage2']['num_views']
    ).to(device)
    logger.info("✅ Using 100% specification-compliant architecture with 5x weight scaling")
    logger.info("✅ Yellow Block: 3→128→256(repeat+concat)→512→1024 with exact tensor shapes")
    logger.info("✅ Red Block: [B,1280]→pool(dim=0)→[1,1280]→FC→[1,256] NO BN/ReLU")

    # --- Optimizer and Loss ---
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['stage2']['learning_rate'],
        weight_decay=config['stage2'].get('weight_decay', 1e-5)
    )
    criterion = torch.nn.MSELoss()

    # --- Resume from Checkpoint ---
    start_epoch = 0
    if args.checkpoint:
        logger.info(f"Resuming from checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

    # --- Training Loop ---
    logger.info(f"Starting training from epoch {start_epoch}...")
    
    # Loss tracking for plotting
    train_losses = []
    
    for epoch in range(start_epoch, config['training']['num_epochs']):
        model.train()
        total_loss = 0.0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch}/{config['training']['num_epochs']}")
        for i, batch in enumerate(progress_bar):
            if args.smoke_run and i >= 5:
                logger.info("Smoke run: finishing epoch after 5 batches.")
                break

            optimizer.zero_grad()

            # Move data to device
            point_clouds = batch['point_clouds'].to(device)
            partial_latents = batch['partial_latents'].to(device)
            gt_latent = batch['gt_latent'].to(device)

            # Forward pass
            predicted_latent = model(point_clouds, partial_latents)

            # Compute loss
            loss = criterion(predicted_latent, gt_latent)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_dataloader)
        train_losses.append(avg_loss)
        
        # Calculate latent norm statistics for this epoch
        if epoch % 2 == 0:  # Every 2 epochs, compute norm statistics
            model.eval()
            norm_ratios = []
            with torch.no_grad():
                for i, batch in enumerate(train_dataloader):
                    if i >= 5:  # Sample a few batches
                        break
                    point_clouds = batch['point_clouds'].to(device)
                    partial_latents = batch['partial_latents'].to(device)
                    gt_latent = batch['gt_latent'].to(device)
                    
                    predicted_latent = model(point_clouds, partial_latents)
                    
                    # Calculate norm ratios
                    for j in range(predicted_latent.shape[0]):
                        pred_norm = predicted_latent[j].norm().item()
                        gt_norm = gt_latent[j].norm().item()
                        norm_ratio = pred_norm / max(gt_norm, 1e-8)
                        norm_ratios.append(norm_ratio)
            
            avg_norm_ratio = np.mean(norm_ratios) if norm_ratios else 0.0
            logger.info(f"Epoch {epoch} | Average Loss: {avg_loss:.6f} | Average Norm Ratio: {avg_norm_ratio:.4f}")
            model.train()
        else:
            logger.info(f"Epoch {epoch} | Average Loss: {avg_loss:.6f}")

        # Plot and save loss graph
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(train_losses)), train_losses, 'b-', linewidth=2, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title('MV-DeepSDF Stage 2 Training Loss')
        plt.grid(True, alpha=0.3)
        plt.legend()
        loss_plot_path = os.path.join(output_dir, 'training_loss.png')
        plt.savefig(loss_plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        # Save checkpoint
        if (epoch + 1) % config['training'].get('save_frequency', 1) == 0:
            checkpoint_path = os.path.join(output_dir, 'checkpoints', f'epoch_{epoch}.pth')
            save_checkpoint(epoch, model, optimizer, checkpoint_path)
            
    # Log final dataset statistics
    dataset_stats = train_dataset.get_dataset_statistics()
    logger.info("Training complete!")
    logger.info("="*60)
    logger.info("FINAL DATASET STATISTICS:")
    logger.info(f"  Total instances from split file: {dataset_stats['total_instances_from_split']}")
    logger.info(f"  Successfully trained instances: {dataset_stats['available_instances']}")
    logger.info(f"  Missing/skipped instances: {dataset_stats['missing_instances']}")
    logger.info(f"  Training success rate: {dataset_stats['success_rate']:.2f}%")
    logger.info("="*60)

if __name__ == '__main__':
    main() 