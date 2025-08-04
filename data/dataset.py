import os
import json
import logging
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import List, Dict, Any, Tuple
import collections
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from networks.mv_deepsdf import FarthestPointSampling

logger = logging.getLogger(__name__)

class MVDeepSDFDataset(Dataset):
    """
    Dataset for loading multi-view point clouds and their corresponding
    pre-computed latent codes for Stage 2 training of MV-DeepSDF.
    """
    def __init__(self, data_root: str, split_file: str, gt_latent_path: str, num_views: int = 6, num_points: int = 256):
        """
        Args:
            data_root (str): Path to the directory containing the data.
            split_file (str): Path to the JSON file defining the data split.
            gt_latent_path (str): Path to the .pth file with ground truth latent codes.
            num_views (int): Number of views per instance.
            num_points (int): Number of points to sample in each view.
        """
        self.data_root = data_root
        self.split_file = split_file
        self.gt_latent_path = gt_latent_path
        self.num_views = num_views
        self.num_points = num_points
        self.fps_sampler = FarthestPointSampling(num_points)

        # Statistics for professional logging
        self.total_instances_from_split = 0
        self.available_instances = 0
        self.missing_instances = 0
        self.missing_instance_ids = []

        self.instance_ids, self.instance_id_to_idx = self._load_instance_ids()
        self.gt_latents = self._load_gt_latents()

        logger.info(f"Dataset Statistics:")
        logger.info(f"  Total instances from split file: {self.total_instances_from_split}")
        logger.info(f"  Available instances with data: {self.available_instances}")
        logger.info(f"  Missing instances (skipped): {self.missing_instances}")
        logger.info(f"  Dataset ready with {len(self.instance_ids)} trainable instances.")

    def _load_instance_ids(self) -> Tuple[List[Tuple[str, str]], Dict[str, int]]:
        """Loads instance IDs from a JSON split file, filtering only available data files."""
        if not os.path.exists(self.split_file):
            raise FileNotFoundError(f"Split file not found at {self.split_file}")
        
        with open(self.split_file, 'r') as f:
            split_data = json.load(f)

        all_instance_pairs = []
        available_instance_pairs = []
        
        # Handles nested dictionary structure like {"ShapeNetV2": {"02958343": ["id1", ...]}}
        for dataset_key in split_data:
            if isinstance(split_data[dataset_key], dict):
                for class_key in split_data[dataset_key]:
                    if isinstance(split_data[dataset_key][class_key], list):
                        for iid in split_data[dataset_key][class_key]:
                            all_instance_pairs.append((class_key, iid))
                            
                            # Check if data file exists and is loadable
                            npz_path = os.path.join(self.data_root, f"{class_key}_{iid}.npz")
                            if os.path.exists(npz_path):
                                # Test if file can be loaded
                                try:
                                    test_data = np.load(npz_path, allow_pickle=True)
                                    # Check if required keys exist
                                    if 'point_clouds' in test_data or 'point_cloud' in test_data:
                                        available_instance_pairs.append((class_key, iid))
                                    else:
                                        self.missing_instance_ids.append(iid)
                                        logger.warning(f"Invalid data file (missing point clouds) for instance {iid}: {npz_path}")
                                except Exception as e:
                                    self.missing_instance_ids.append(iid)
                                    logger.warning(f"Corrupted data file for instance {iid}: {npz_path} - {e}")
                            else:
                                self.missing_instance_ids.append(iid)
                                logger.warning(f"Missing data file for instance {iid}: {npz_path}")

        # Update statistics
        self.total_instances_from_split = len(all_instance_pairs)
        self.available_instances = len(available_instance_pairs)
        self.missing_instances = len(self.missing_instance_ids)

        # The instance_id_to_idx maps the instance_id (iid) to its index in the latent code tensor
        instance_id_to_idx = {pair[1]: i for i, pair in enumerate(available_instance_pairs)}
        
        logger.info(f"Processed {self.total_instances_from_split} instances from split file")
        logger.info(f"Found {self.available_instances} available data files, {self.missing_instances} missing")
        
        return available_instance_pairs, instance_id_to_idx

    def _load_gt_latents(self) -> Dict[str, torch.Tensor]:
        """Loads ground truth latent codes from Stage 1."""
        if not os.path.exists(self.gt_latent_path):
            raise FileNotFoundError(f"Ground truth latent codes not found at {self.gt_latent_path}")
        
        checkpoint = torch.load(self.gt_latent_path)
        
        # Check if this is our generated test latent codes with instance_order
        if 'instance_order' in checkpoint:
            # This is our generated test latent codes format
            latents_tensor = checkpoint['latent_codes']
            instance_order = checkpoint['instance_order']
            logger.info(f"Loaded generated test latent codes with shape: {latents_tensor.shape}")
            logger.info(f"Instance order contains {len(instance_order)} instances")
            
            # Create mapping from instance ID to latent vector
            gt_latents = {}
            for class_id, instance_id in self.instance_ids:
                if instance_id in instance_order:
                    idx = instance_order.index(instance_id)
                    gt_latents[instance_id] = latents_tensor[idx]
                else:
                    logger.warning(f"Instance ID {instance_id} from MV-DeepSDF split not found in generated test latents")
            
            logger.info(f"Successfully mapped {len(gt_latents)} ground truth latent codes")
            return gt_latents
        
        # Latent vectors could be stored directly or in an embedding layer (standard training latents)
        if 'latent_codes' in checkpoint:
            # Handle cases where latent_codes could be a tensor or a state_dict
            if isinstance(checkpoint['latent_codes'], torch.Tensor):
                latents_tensor = checkpoint['latent_codes']
            elif isinstance(checkpoint['latent_codes'], collections.OrderedDict):
                if 'weight' in checkpoint['latent_codes']:
                    latents_tensor = checkpoint['latent_codes']['weight']
                else:
                    raise KeyError("Could not find 'weight' key in latent_codes OrderedDict")
            elif hasattr(checkpoint['latent_codes'], 'weight'):
                latents_tensor = checkpoint['latent_codes'].weight
            else:
                 raise TypeError("Unsupported type for 'latent_codes' in checkpoint")
        elif 'model_state_dict' in checkpoint and 'latent_vectors.weight' in checkpoint['model_state_dict']:
             latents_tensor = checkpoint['model_state_dict']['latent_vectors.weight']
        elif 'latent_vecs.weight' in checkpoint: # from original deepsdf repo
             latents_tensor = checkpoint['latent_vecs.weight']
        else:
            raise KeyError("Could not find latent vectors in the provided checkpoint file.")

        latents_tensor = latents_tensor.detach()
        logger.info(f"Loaded latent tensor with shape: {latents_tensor.shape}")
        
        # We need to load the Stage 1 training split to get the correct index mapping
        # The latent codes are stored in the same order as the Stage 1 training split
        stage1_split_file = "configs/mvdeepsdf_train_split_available.json"
        if not os.path.exists(stage1_split_file):
            raise FileNotFoundError(f"Stage 1 training split not found at {stage1_split_file}")
            
        with open(stage1_split_file, 'r') as f:
            stage1_split = json.load(f)
        
        # Extract instance IDs from Stage 1 split in the correct order
        stage1_instances = []
        for dataset_key in stage1_split:
            if isinstance(stage1_split[dataset_key], dict):
                for class_key in stage1_split[dataset_key]:
                    if isinstance(stage1_split[dataset_key][class_key], list):
                        stage1_instances.extend(stage1_split[dataset_key][class_key])
        
        logger.info(f"Stage 1 split contains {len(stage1_instances)} instances")
        
        # Verify dimensions match
        if len(stage1_instances) != latents_tensor.shape[0]:
            raise ValueError(f"Mismatch: Stage 1 split has {len(stage1_instances)} instances but latent tensor has {latents_tensor.shape[0]} rows")
        
        # Create mapping from instance ID to latent vector
        gt_latents = {}
        for class_id, instance_id in self.instance_ids:
            if instance_id in stage1_instances:
                idx = stage1_instances.index(instance_id)
                gt_latents[instance_id] = latents_tensor[idx]
            else:
                logger.warning(f"Instance ID {instance_id} from MV-DeepSDF split not found in Stage 1 split")

        logger.info(f"Successfully mapped {len(gt_latents)} ground truth latent codes")
        return gt_latents

    def __len__(self) -> int:
        return len(self.instance_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        class_id, instance_id = self.instance_ids[idx]
        
        # Path to the NPZ file generated by generate_multi_sweep_data.py
        npz_path = os.path.join(self.data_root, f"{class_id}_{instance_id}.npz")

        # This should not happen now since we filter during initialization
        if not os.path.exists(npz_path):
             logger.error(f"Unexpected missing file during training: {npz_path}")
             raise FileNotFoundError(f"Data file not found for instance {instance_id} at {npz_path}")

        try:
            data = np.load(npz_path, allow_pickle=True)
        except Exception as e:
            logger.error(f"Failed to load NPZ file {npz_path}: {e}")
            raise
        
        # Load multi-view data with better error handling
        available_keys = list(data.keys())
        if 'point_clouds' in data:
            key = 'point_clouds'
        elif 'point_cloud' in data:
            key = 'point_cloud'
        elif 'sweeps' in data:
            key = 'sweeps'  # Handle legacy format from earlier data generation
        else:
            raise KeyError(f"No point cloud data found in {npz_path}. Available keys: {available_keys}. Expected one of: ['point_clouds', 'point_cloud', 'sweeps']")
        
        try:
            point_clouds = torch.from_numpy(data[key]).float()
        except KeyError as e:
            logger.error(f"KeyError loading {key} from {npz_path}. Available keys: {available_keys}")
            raise
        
        # If the number of views in the file is different from what is expected,
        # we can choose to either raise an error or handle it gracefully.
        # Here we'll just log a warning.
        if point_clouds.shape[0] != self.num_views:
            logger.warning(
                f"Expected {self.num_views} views for instance {instance_id}, but found {point_clouds.shape[0]}. "
                f"Using the views available in the file."
            )

        # The FPS sampling is expected to be done in the data generation script.
        # If num_points mismatches, resample using FPS.
        if point_clouds.shape[1] != self.num_points:
             resampled_clouds = []
             for pc in point_clouds:
                 resampled_clouds.append(self.fps_sampler(pc))
             point_clouds = torch.stack(resampled_clouds, dim=0)

        # Partial latents might not always be present, handle this case.
        if 'partial_latents' in data:
            partial_latents = torch.from_numpy(data['partial_latents']).float()
        else:
            # If no partial latents, create random ones as a placeholder
            gt_latent_dim = self.gt_latents.get(instance_id, torch.zeros(256)).shape[0]
            partial_latents = torch.randn(point_clouds.shape[0], gt_latent_dim)
            logger.warning(f"No partial latents found for {instance_id}, using random placeholder.")
        
        gt_latent = self.gt_latents.get(instance_id)
        if gt_latent is None:
            logger.warning(f"Ground truth latent not found for instance {instance_id}, using zero placeholder.")
            gt_latent = torch.zeros(256)  # Use zero placeholder for pure reconstruction

        return {
            'instance_id': instance_id,
            'point_clouds': point_clouds,
            'partial_latents': partial_latents,
            'gt_latent': gt_latent
        }
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """Returns comprehensive dataset statistics for logging."""
        return {
            'total_instances_from_split': self.total_instances_from_split,
            'available_instances': self.available_instances,
            'missing_instances': self.missing_instances,
            'success_rate': (self.available_instances / self.total_instances_from_split) * 100 if self.total_instances_from_split > 0 else 0,
            'missing_instance_ids': self.missing_instance_ids[:10] if len(self.missing_instance_ids) > 10 else self.missing_instance_ids  # Limit for logging
        } 