#!/usr/bin/env python3
"""
Hyperparameter sweep for MV-DeepSDF Stage 2 latent scaling fix
Tests weight initialization scaling and learning rate adjustments
"""

import os, sys, json, argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import logging
from typing import Optional, Dict, Any
from datetime import datetime
import copy

sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))
from data.dataset import MVDeepSDFDataset
from networks.mv_deepsdf import MVDeepSDF, LatentCodePredictor
from networks.deep_sdf_decoder import Decoder

class EnhancedLatentCodePredictor(nn.Module):
    """Enhanced latent predictor with weight scaling and optional learnable scalar"""
    def __init__(self, input_dim: int = 128, output_dim: int = 256, 
                 weight_scale: float = 1.0, use_learnable_alpha: bool = False):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.use_learnable_alpha = use_learnable_alpha
        
        # Apply weight initialization scaling
        with torch.no_grad():
            self.fc.weight.data *= weight_scale
            if self.fc.bias is not None:
                self.fc.bias.data *= weight_scale
        
        # Optional learnable scalar
        if use_learnable_alpha:
            self.alpha = nn.Parameter(torch.ones(1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc(x)
        if self.use_learnable_alpha:
            out = out * self.alpha
        return out

class EnhancedMVDeepSDF(MVDeepSDF):
    """Enhanced MV-DeepSDF with configurable latent predictor"""
    def __init__(self, weight_scale: float = 1.0, use_learnable_alpha: bool = False, **kwargs):
        super().__init__(**kwargs)
        # Replace the predictor with enhanced version
        self.predictor = EnhancedLatentCodePredictor(
            input_dim=kwargs.get('element_feature_dim', 128),
            output_dim=kwargs.get('latent_dim', 256),
            weight_scale=weight_scale,
            use_learnable_alpha=use_learnable_alpha
        )

def setup_logging(exp_dir: str, config: Dict[str, Any]) -> logging.Logger:
    """Setup logging with detailed experiment configuration"""
    log_file = os.path.join(exp_dir, 'sweep.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info(f"Experiment configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    return logger

def set_reproducible_seed(seed: int = 42):
    """Set seeds for reproducible results"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def compute_latent_metrics(pred_latent: torch.Tensor, gt_latent: torch.Tensor) -> Dict[str, float]:
    """Compute latent space metrics without requiring decoder"""
    metrics = {}
    
    # Basic latent metrics
    metrics['pred_norm'] = pred_latent.norm().item()
    metrics['gt_norm'] = gt_latent.norm().item()
    metrics['norm_ratio'] = metrics['pred_norm'] / max(metrics['gt_norm'], 1e-8)
    metrics['norm_diff'] = abs(metrics['pred_norm'] - metrics['gt_norm'])
    
    # Cosine similarity in latent space
    cos_sim = torch.nn.functional.cosine_similarity(
        pred_latent.view(-1), gt_latent.view(-1), dim=0
    )
    metrics['latent_cosine_sim'] = cos_sim.item()
    
    # MSE loss in latent space
    mse_loss = torch.nn.functional.mse_loss(pred_latent, gt_latent)
    metrics['latent_mse'] = mse_loss.item()
    
    return metrics

def train_configuration(config: Dict[str, Any], logger: logging.Logger) -> Dict[str, float]:
    """Train a single hyperparameter configuration"""
    
    # Set reproducible seed
    set_reproducible_seed(config['seed'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load base config
    with open('configs/mvdeepsdf_config.json') as f:
        base_cfg = json.load(f)
    
    # Create enhanced model
    model = EnhancedMVDeepSDF(
        num_points=base_cfg['stage2']['num_points'],
        global_feature_dim=base_cfg['stage2']['global_feature_dim'],
        latent_dim=base_cfg['stage2']['latent_dim'],
        element_feature_dim=base_cfg['stage2']['element_feature_dim'],
        num_views=base_cfg['stage2']['num_views'],
        weight_scale=config['weight_scale'],
        use_learnable_alpha=config['use_learnable_alpha']
    ).to(device)
    
    # Setup optimizer with different learning rates
    predictor_params = list(model.predictor.parameters())
    predictor_param_ids = {id(p) for p in predictor_params}
    other_params = [p for p in model.parameters() if id(p) not in predictor_param_ids]
    
    optimizer = optim.Adam([
        {'params': other_params, 'lr': config['base_lr']},
        {'params': predictor_params, 'lr': config['predictor_lr']}
    ])
    
    # Load dataset
    dataset = MVDeepSDFDataset(
        data_root='experiments/multisweep_data_final',
        split_file='configs/mvdeepsdf_stage2_train.json',
        gt_latent_path='examples/cars/LatentCodes/latest.pth',
        num_views=base_cfg['stage2']['num_views']
    )
    
    val_dataset = MVDeepSDFDataset(
        data_root='experiments/multisweep_data_final',
        split_file='configs/mvdeepsdf_stage2_test.json',
        gt_latent_path='examples/cars/LatentCodes/latest.pth',
        num_views=base_cfg['stage2']['num_views']
    )
    
    train_loader = DataLoader(dataset, batch_size=config['batch_size'], 
                             shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)
    
    # Training loop
    model.train()
    best_norm_ratio = float('inf')
    patience_counter = 0
    
    for epoch in range(config['max_epochs']):
        epoch_losses = []
        
        for batch_idx, batch in enumerate(train_loader):
            point_clouds = batch['point_clouds'].to(device)
            partial_latents = batch['partial_latents'].to(device)
            gt_latents = batch['gt_latent'].to(device)
            
            optimizer.zero_grad()
            
            pred_latents = model(point_clouds, partial_latents)
            loss = torch.nn.functional.mse_loss(pred_latents, gt_latents)
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            epoch_losses.append(loss.item())
            
            if batch_idx % 50 == 0:
                logger.info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}")
        
        avg_loss = np.mean(epoch_losses)
        
        # Validation
        if epoch % config['val_every'] == 0:
            model.eval()
            val_metrics = []
            
            with torch.no_grad():
                for val_batch in val_loader:
                    point_clouds = val_batch['point_clouds'].to(device)
                    partial_latents = val_batch['partial_latents'].to(device)
                    gt_latents = val_batch['gt_latent'].to(device)
                    
                    pred_latents = model(point_clouds, partial_latents)
                    
                    # Compute metrics for this batch
                    batch_metrics = compute_latent_metrics(
                        pred_latents[0], gt_latents[0]
                    )
                    val_metrics.append(batch_metrics)
            
            # Average validation metrics
            avg_metrics = {}
            for key in val_metrics[0].keys():
                avg_metrics[key] = np.mean([m[key] for m in val_metrics])
            
            logger.info(f"Epoch {epoch} Validation:")
            logger.info(f"  Loss: {avg_loss:.6f}")
            logger.info(f"  Norm ratio: {avg_metrics['norm_ratio']:.4f}")
            logger.info(f"  Norm diff: {avg_metrics['norm_diff']:.4f}")
            logger.info(f"  Latent MSE: {avg_metrics['latent_mse']:.6f}")
            logger.info(f"  Latent cosine sim: {avg_metrics['latent_cosine_sim']:.4f}")
            
            # Early stopping based on norm ratio (closer to 1.0 is better)
            norm_ratio_error = abs(avg_metrics['norm_ratio'] - 1.0)
            if norm_ratio_error < best_norm_ratio:
                best_norm_ratio = norm_ratio_error
                patience_counter = 0
                # Save best model
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': config,
                    'metrics': avg_metrics,
                    'epoch': epoch
                }, os.path.join(config['exp_dir'], 'best_model.pth'))
            else:
                patience_counter += 1
                
            if patience_counter >= config['patience']:
                logger.info(f"Early stopping at epoch {epoch}")
                break
                
            model.train()
    
    # Return best metrics
    best_checkpoint = torch.load(os.path.join(config['exp_dir'], 'best_model.pth'))
    return best_checkpoint['metrics']

def run_hyperparameter_sweep(args):
    """Run the complete hyperparameter sweep"""
    
    # Create base experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_exp_dir = f"experiments/mvdeepsdf_hyperparam_sweep_{timestamp}"
    os.makedirs(base_exp_dir, exist_ok=True)
    
    # Define sweep configurations
    weight_scales = args.weight_scales
    lr_scales = args.lr_scales
    base_lr = args.base_lr
    
    configs = []
    for ws in weight_scales:
        for lr_scale in lr_scales:
            for use_alpha in [False, True] if args.test_learnable_alpha else [False]:
                config = {
                    'weight_scale': ws,
                    'lr_scale': lr_scale,
                    'base_lr': base_lr,
                    'predictor_lr': base_lr * lr_scale,
                    'use_learnable_alpha': use_alpha,
                    'batch_size': args.batch_size,
                    'max_epochs': args.max_epochs,
                    'val_every': args.val_every,
                    'patience': args.patience,
                    'seed': args.seed,
                    'exp_name': f"ws{ws}_lr{lr_scale}_alpha{use_alpha}",
                    'exp_dir': os.path.join(base_exp_dir, f"ws{ws}_lr{lr_scale}_alpha{use_alpha}")
                }
                configs.append(config)
    
    # Create summary logger
    summary_logger = setup_logging(base_exp_dir, {'sweep_configs': len(configs)})
    
    # Run all configurations
    results = []
    
    for i, config in enumerate(configs):
        summary_logger.info(f"\n{'='*50}")
        summary_logger.info(f"Running configuration {i+1}/{len(configs)}: {config['exp_name']}")
        summary_logger.info(f"{'='*50}")
        
        # Create experiment directory
        os.makedirs(config['exp_dir'], exist_ok=True)
        
        # Setup logger for this configuration
        logger = setup_logging(config['exp_dir'], config)
        
        try:
            # Train this configuration
            metrics = train_configuration(config, logger)
            
            # Store results
            result = {
                'config': config,
                'metrics': metrics,
                'success': True
            }
            results.append(result)
            
            summary_logger.info(f"✅ Configuration {config['exp_name']} completed successfully")
            summary_logger.info(f"   Norm ratio: {metrics['norm_ratio']:.4f}")
            summary_logger.info(f"   Latent MSE: {metrics['latent_mse']:.6f}")
            
        except Exception as e:
            summary_logger.error(f"❌ Configuration {config['exp_name']} failed: {str(e)}")
            result = {
                'config': config,
                'metrics': None,
                'success': False,
                'error': str(e)
            }
            results.append(result)
    
    # Analyze results
    successful_results = [r for r in results if r['success']]
    
    if successful_results:
        summary_logger.info(f"\n{'='*50}")
        summary_logger.info("FINAL RESULTS SUMMARY")
        summary_logger.info(f"{'='*50}")
        
        # Sort by norm ratio (closest to 1.0 is best)
        successful_results.sort(key=lambda x: abs(x['metrics']['norm_ratio'] - 1.0))
        
        summary_logger.info("Top 3 configurations (by norm ratio):")
        for i, result in enumerate(successful_results[:3]):
            config = result['config']
            metrics = result['metrics']
            summary_logger.info(f"\n{i+1}. {config['exp_name']}")
            summary_logger.info(f"   Weight scale: {config['weight_scale']}")
            summary_logger.info(f"   LR scale: {config['lr_scale']}")
            summary_logger.info(f"   Learnable alpha: {config['use_learnable_alpha']}")
            summary_logger.info(f"   Norm ratio: {metrics['norm_ratio']:.4f}")
            summary_logger.info(f"   Norm diff: {metrics['norm_diff']:.4f}")
            summary_logger.info(f"   Latent MSE: {metrics['latent_mse']:.6f}")
            summary_logger.info(f"   Latent cosine sim: {metrics['latent_cosine_sim']:.4f}")
        
        # Save results summary
        results_file = os.path.join(base_exp_dir, 'sweep_results.json')
        with open(results_file, 'w') as f:
            # Convert tensors to lists for JSON serialization
            json_results = []
            for result in results:
                json_result = copy.deepcopy(result)
                if json_result['success'] and json_result['metrics']:
                    # Ensure all metrics are JSON serializable
                    for key, value in json_result['metrics'].items():
                        if isinstance(value, (torch.Tensor, np.ndarray)):
                            json_result['metrics'][key] = float(value)
                json_results.append(json_result)
            
            json.dump(json_results, f, indent=2)
        
        summary_logger.info(f"\nResults saved to: {results_file}")
        
        # Return path to best model
        best_result = successful_results[0]
        best_model_path = os.path.join(best_result['config']['exp_dir'], 'best_model.pth')
        summary_logger.info(f"Best model saved at: {best_model_path}")
        return best_model_path
    
    else:
        summary_logger.error("❌ No configurations succeeded!")
        return None

def main():
    parser = argparse.ArgumentParser(description='MV-DeepSDF Hyperparameter Sweep')
    parser.add_argument('--weight-scales', nargs='+', type=float, default=[1.0, 2.0, 5.0],
                       help='Weight initialization scaling factors')
    parser.add_argument('--lr-scales', nargs='+', type=float, default=[1.0, 2.0],
                       help='Learning rate scaling factors for predictor')
    parser.add_argument('--base-lr', type=float, default=1e-4,
                       help='Base learning rate')
    parser.add_argument('--test-learnable-alpha', action='store_true',
                       help='Test configurations with learnable alpha scalar')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Training batch size')
    parser.add_argument('--max-epochs', type=int, default=20,
                       help='Maximum training epochs per configuration')
    parser.add_argument('--val-every', type=int, default=2,
                       help='Validation frequency (epochs)')
    parser.add_argument('--patience', type=int, default=5,
                       help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    print("🔥 Starting MV-DeepSDF Hyperparameter Sweep")
    print(f"Weight scales: {args.weight_scales}")
    print(f"LR scales: {args.lr_scales}")
    print(f"Test learnable alpha: {args.test_learnable_alpha}")
    print(f"Seed: {args.seed}")
    
    best_model_path = run_hyperparameter_sweep(args)
    
    if best_model_path:
        print(f"\n✅ Sweep completed! Best model at: {best_model_path}")
    else:
        print("\n❌ Sweep failed!")

if __name__ == '__main__':
    main()