#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import argparse
import json
import logging
import os
import random
import time
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors

import deep_sdf
import deep_sdf.workspace as ws


def detect_point_cloud_scale_and_convert(point_cloud, target_clamp_distance=0.05):
    """
    SCALABLE: Auto-detect point cloud scale and convert to target DeepSDF scale
    
    This function makes the reconstruction pipeline work with any coordinate system:
    - Paper's [-1,1] range → converts to DeepSDF scale
    - Already in DeepSDF scale → uses as-is
    - Any other scale → adapts appropriately
    """
    if point_cloud.shape[0] == 0:
        return point_cloud, 1.0, target_clamp_distance
    
    current_max = np.abs(point_cloud).max()
    
    # Detect likely coordinate system based on magnitude
    if current_max > 2.0:
        # Likely raw mesh coordinates - shouldn't happen but handle gracefully
        logging.warning(f"Point cloud appears to be in raw coordinates (max: {current_max:.3f})")
        scale_factor = target_clamp_distance / current_max
        converted_pc = point_cloud * scale_factor
        logging.info(f"Converted raw coordinates to DeepSDF scale by factor {scale_factor:.6f}")
        return converted_pc, scale_factor, target_clamp_distance * scale_factor
        
    elif current_max > 0.8:
        # Likely paper's [-1,1] coordinate system
        paper_range = 1.0
        scale_factor = target_clamp_distance / paper_range
        converted_pc = point_cloud * scale_factor
        logging.info(f"Detected paper coordinate system ([-1,1], max: {current_max:.3f})")
        logging.info(f"Converted to DeepSDF scale (±{target_clamp_distance}) by factor {scale_factor:.6f}")
        return converted_pc, scale_factor, target_clamp_distance * scale_factor
        
    else:
        # Likely already in DeepSDF coordinate system
        logging.info(f"Point cloud appears to be in DeepSDF coordinate system (max: {current_max:.3f})")
        return point_cloud, 1.0, target_clamp_distance


def point_cloud_to_sdf_samples(point_cloud, num_samples=30000, surface_normal_offset=0.005, target_clamp_distance=0.05):
    """
    SCALABLE: Convert point cloud to SDF samples with automatic scale handling.
    
    This function now works with any input coordinate system and automatically
    converts to the appropriate DeepSDF scale based on the model's ClampingDistance.
    
    Args:
        point_cloud: (N, 3) numpy array of surface points (any coordinate system)
        num_samples: Number of SDF samples to generate
        surface_normal_offset: Distance along normal (will be scaled appropriately)
        target_clamp_distance: Target DeepSDF ClampingDistance (auto-detected or specified)
        
    Returns:
        sdf_samples: (pos_tensor, neg_tensor) - tuple of tensors [x, y, z, sdf_value]
    """
    if point_cloud.shape[0] == 0:
        logging.warning("Empty point cloud provided")
        return torch.zeros((0, 4)), torch.zeros((0, 4))
    
    # Remove duplicate/zero points
    valid_mask = np.any(point_cloud != 0, axis=1)
    if not np.any(valid_mask):
        logging.warning("Point cloud contains only zero points")
        return torch.zeros((0, 4)), torch.zeros((0, 4))
    
    surface_points = point_cloud[valid_mask]
    
    # SCALABLE: Auto-detect and convert coordinate system
    surface_points, scale_factor, adjusted_offset = detect_point_cloud_scale_and_convert(
        surface_points, target_clamp_distance
    )
    
    # Use the properly scaled surface normal offset
    surface_normal_offset = adjusted_offset
        
    logging.info(f"Final point cloud range: ±{np.abs(surface_points).max():.6f}, surface_normal_offset: {surface_normal_offset:.6f}")
    
    # Estimate surface normals using local neighborhood
    if surface_points.shape[0] < 3:
        logging.warning(f"Insufficient points for normal estimation: {surface_points.shape[0]}")
        # Use random unit vectors as fallback
        normals = np.random.randn(surface_points.shape[0], 3)
        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    else:
        # Use k-NN to estimate local surface normals
        k = min(10, surface_points.shape[0] - 1)
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(surface_points)
        distances, indices = nbrs.kneighbors(surface_points)
        
        normals = []
        for i in range(surface_points.shape[0]):
            neighbors = surface_points[indices[i][1:]]  # Exclude self
            
            if neighbors.shape[0] >= 2:
                # PCA-based normal estimation
                centered = neighbors - neighbors.mean(axis=0)
                _, _, V = np.linalg.svd(centered)
                normal = V[-1]  # Last component is normal direction
                
                # Ensure consistent orientation (pointing outward)
                center_to_point = surface_points[i] - surface_points.mean(axis=0)
                if np.dot(normal, center_to_point) < 0:
                    normal = -normal
                    
                normals.append(normal)
            else:
                # Fallback for insufficient neighbors
                normals.append(np.array([0, 0, 1]))
        
        normals = np.array(normals)
    
    # Generate SDF samples along surface normals (MV-DeepSDF methodology)
    samples = []
    sdf_values = []
    
    samples_per_surface_point = max(1, num_samples // surface_points.shape[0])
    
    for i in range(surface_points.shape[0]):
        surface_pt = surface_points[i]
        normal = normals[i]
        
        # Generate samples along normal direction - ensure balanced pos/neg
        for j in range(samples_per_surface_point):
            # FIXED: Use small, controlled offset that respects DeepSDF bounds
            # Use uniform distribution with small range instead of exponential
            max_offset = min(surface_normal_offset * 3, 0.01)  # Cap at 0.01 to stay within ±0.05 range
            offset_distance = np.random.uniform(0.0001, max_offset)  # Small positive offset
            
            # Alternate between interior and exterior for balance
            if j % 2 == 0:
                # Interior sample (negative SDF)
                sample_point = surface_pt - normal * offset_distance
                sdf_value = -offset_distance
            else:
                # Exterior sample (positive SDF)
                sample_point = surface_pt + normal * offset_distance  
                sdf_value = offset_distance
            
            # SCALABLE: Ensure sample stays within target DeepSDF bounds
            if np.abs(sample_point).max() <= target_clamp_distance:
                samples.append(sample_point)
                sdf_values.append(sdf_value)
            else:
                # Skip samples that exceed bounds
                logging.debug(f"Skipped sample outside bounds: {np.abs(sample_point).max():.6f}")
                continue
    
    # Fill remaining samples if needed - ensure balance
    fill_count = 0
    max_fill_attempts = num_samples * 2  # Prevent infinite loop
    attempts = 0
    
    while len(samples) < num_samples and attempts < max_fill_attempts:
        attempts += 1
        
        # Random surface point
        idx = np.random.randint(surface_points.shape[0])
        surface_pt = surface_points[idx]
        normal = normals[idx]
        
        # Use same controlled offset as above
        max_offset = min(surface_normal_offset * 3, 0.01)
        offset_distance = np.random.uniform(0.0001, max_offset)
        
        # Alternate to maintain balance
        if fill_count % 2 == 0:
            sample_point = surface_pt - normal * offset_distance  # Interior (negative)
            sdf_value = -offset_distance
        else:
            sample_point = surface_pt + normal * offset_distance  # Exterior (positive)
            sdf_value = offset_distance
        
        # Only add if within bounds
        if np.abs(sample_point).max() <= target_clamp_distance:
            samples.append(sample_point)
            sdf_values.append(sdf_value)
            fill_count += 1
    
    if len(samples) < num_samples:
        logging.warning(f"Could only generate {len(samples)}/{num_samples} samples within DeepSDF bounds")
    
    # Convert to required format and truncate to exact count
    samples = np.array(samples[:num_samples])
    sdf_values = np.array(sdf_values[:num_samples])
    
    # Separate positive and negative samples as expected by DeepSDF data format
    samples = np.array(samples)
    sdf_values = np.array(sdf_values)
    
    # Split into positive (exterior) and negative (interior) samples
    pos_mask = sdf_values >= 0
    neg_mask = sdf_values < 0
    
    pos_samples = samples[pos_mask]
    pos_sdf = sdf_values[pos_mask]
    neg_samples = samples[neg_mask]  
    neg_sdf = sdf_values[neg_mask]
    
    # Debug: Check sample distribution
    logging.info(f"Generated samples: {len(pos_samples)} positive, {len(neg_samples)} negative")
    
    # Ensure we have both positive and negative samples
    if len(pos_samples) == 0 or len(neg_samples) == 0:
        logging.warning(f"Imbalanced samples: {len(pos_samples)} positive, {len(neg_samples)} negative")
        # The reconstruct function will handle this appropriately
    
    # Create tensors in [x, y, z, sdf] format as expected by DeepSDF
    if len(pos_samples) > 0:
        pos_tensor = torch.from_numpy(np.column_stack([pos_samples, pos_sdf]).astype(np.float32))
    else:
        pos_tensor = torch.zeros((0, 4), dtype=torch.float32)
        
    if len(neg_samples) > 0:
        neg_tensor = torch.from_numpy(np.column_stack([neg_samples, neg_sdf]).astype(np.float32))
    else:
        neg_tensor = torch.zeros((0, 4), dtype=torch.float32)
    
    # Return as tuple expected by unpack_sdf_samples_from_ram
    return (pos_tensor, neg_tensor)


def reconstruct_from_point_cloud(
    decoder, 
    point_cloud, 
    num_iterations, 
    latent_size, 
    stat, 
    clamp_dist,
    num_samples=30000,
    lr=5e-4,
    l2reg=False
):
    """
    Reconstruct mesh from point cloud using DeepSDF baseline (MV-DeepSDF comparison).
    
    This implements the paper's DeepSDF baseline methodology for single-sweep reconstruction.
    Point clouds are converted to SDF samples following Section 3.1 approach.
    
    Args:
        decoder: Trained DeepSDF decoder network
        point_cloud: (N, 3) numpy array from single LiDAR sweep
        num_iterations: MAP estimation iterations
        latent_size: Latent code dimension
        stat: Statistical prior for latent code initialization
        clamp_dist: SDF clamping distance
        num_samples: Number of SDF samples for optimization
        lr: Learning rate
        l2reg: L2 regularization on latent code
        
    Returns:
        loss_num: Final reconstruction loss
        latent: Optimized latent code
    """
    # Convert point cloud to SDF samples - ensure we generate enough for sampling
    # DeepSDF reconstruction needs enough samples for random sampling (requires subsample/2 of each type)
    min_required_samples = max(num_samples * 2, 8000)  # Ensure we have plenty for sampling  
    adjusted_samples = max(min_required_samples, point_cloud.shape[0] * 50)  # More samples per surface point
    logging.info(f"Converting point cloud ({point_cloud.shape[0]} points) to {adjusted_samples} SDF samples")
    pos_tensor, neg_tensor = point_cloud_to_sdf_samples(point_cloud, adjusted_samples)
    
    # Create tuple format expected by reconstruct()
    test_sdf = (pos_tensor, neg_tensor)
    
    # Use existing reconstruct function for MAP estimation
    # Use the total available samples (pos + neg) for reconstruction
    available_samples = len(pos_tensor) + len(neg_tensor)
    reconstruction_samples = min(num_samples, available_samples)
    return reconstruct(
        decoder, num_iterations, latent_size, test_sdf, stat, clamp_dist,
        reconstruction_samples, lr, l2reg
    )


def reconstruct(
    decoder,
    num_iterations,
    latent_size,
    test_sdf,
    stat,
    clamp_dist,
    num_samples=30000,
    lr=5e-4,
    l2reg=False,
):
    def adjust_learning_rate(
        initial_lr, optimizer, num_iterations, decreased_by, adjust_lr_every
    ):
        lr = initial_lr * ((1 / decreased_by) ** (num_iterations // adjust_lr_every))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    decreased_by = 10
    adjust_lr_every = int(num_iterations / 2)

    if type(stat) == type(0.1):
        latent = torch.ones(1, latent_size).normal_(mean=0, std=stat).cuda()
    else:
        latent = torch.normal(stat[0].detach(), stat[1].detach()).cuda()

    latent.requires_grad = True

    optimizer = torch.optim.Adam([latent], lr=lr)

    loss_num = 0
    loss_l1 = torch.nn.L1Loss()

    for e in range(num_iterations):

        decoder.eval()
        sdf_data = deep_sdf.data.unpack_sdf_samples_from_ram(
            test_sdf, num_samples
        ).cuda()
        xyz = sdf_data[:, 0:3]
        sdf_gt = sdf_data[:, 3].unsqueeze(1)

        sdf_gt = torch.clamp(sdf_gt, -clamp_dist, clamp_dist)

        adjust_learning_rate(lr, optimizer, e, decreased_by, adjust_lr_every)

        optimizer.zero_grad()

        actual_samples = xyz.shape[0]
        latent_inputs = latent.expand(actual_samples, -1)

        inputs = torch.cat([latent_inputs, xyz], 1).cuda()

        pred_sdf = decoder(inputs)

        # TODO: why is this needed?
        if e == 0:
            pred_sdf = decoder(inputs)

        pred_sdf = torch.clamp(pred_sdf, -clamp_dist, clamp_dist)

        loss = loss_l1(pred_sdf, sdf_gt)
        if l2reg:
            loss += 1e-4 * torch.mean(latent.pow(2))
        loss.backward()
        optimizer.step()

        if e % 50 == 0:
            logging.debug(loss.cpu().data.numpy())
            logging.debug(e)
            logging.debug(latent.norm())
        loss_num = loss.cpu().data.numpy()

    return loss_num, latent


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(
        description="Use a trained DeepSDF decoder to reconstruct a shape given SDF "
        + "samples."
    )
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory which includes specifications and saved model "
        + "files to use for reconstruction",
    )
    arg_parser.add_argument(
        "--checkpoint",
        "-c",
        dest="checkpoint",
        default="latest",
        help="The checkpoint weights to use. This can be a number indicated an epoch "
        + "or 'latest' for the latest weights (this is the default)",
    )
    arg_parser.add_argument(
        "--data",
        "-d",
        dest="data_source",
        required=True,
        help="The data source directory.",
    )
    arg_parser.add_argument(
        "--split",
        "-s",
        dest="split_filename",
        required=True,
        help="The split to reconstruct.",
    )
    arg_parser.add_argument(
        "--iters",
        dest="iterations",
        default=800,
        help="The number of iterations of latent code optimization to perform.",
    )
    arg_parser.add_argument(
        "--skip",
        dest="skip",
        action="store_true",
        help="Skip meshes which have already been reconstructed.",
    )
    arg_parser.add_argument(
        "--point_cloud_dir",
        dest="point_cloud_dir",
        help="Directory containing multi-sweep point cloud .npz files for DeepSDF baseline comparison",
    )
    arg_parser.add_argument(
        "--baseline_mode",
        dest="baseline_mode",
        action="store_true",
        help="Run DeepSDF baseline reconstruction from single point cloud sweeps",
    )
    deep_sdf.add_common_args(arg_parser)

    args = arg_parser.parse_args()

    deep_sdf.configure_logging(args)

    def empirical_stat(latent_vecs, indices):
        lat_mat = torch.zeros(0).cuda()
        for ind in indices:
            lat_mat = torch.cat([lat_mat, latent_vecs[ind]], 0)
        mean = torch.mean(lat_mat, 0)
        var = torch.var(lat_mat, 0)
        return mean, var

    specs_filename = os.path.join(args.experiment_directory, "specs.json")

    if not os.path.isfile(specs_filename):
        raise Exception(
            'The experiment directory does not include specifications file "specs.json"'
        )

    specs = json.load(open(specs_filename))

    arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder"])

    latent_size = specs["CodeLength"]

    decoder = arch.Decoder(latent_size, **specs["NetworkSpecs"])

    decoder = torch.nn.DataParallel(decoder)

    saved_model_state = torch.load(
        os.path.join(
            args.experiment_directory, ws.model_params_subdir, args.checkpoint + ".pth"
        )
    )
    saved_model_epoch = saved_model_state["epoch"]

    decoder.load_state_dict(saved_model_state["model_state_dict"])

    decoder = decoder.module.cuda()

    with open(args.split_filename, "r") as f:
        split = json.load(f)

    if args.baseline_mode and args.point_cloud_dir:
        # DeepSDF baseline mode: load point cloud files instead of SDF data
        point_cloud_files = []
        for class_instances in split.values():
            for class_id, instances in class_instances.items():
                for instance_id in instances:
                    pc_file = f"{class_id}_{instance_id}.npz"
                    pc_path = os.path.join(args.point_cloud_dir, pc_file)
                    if os.path.exists(pc_path):
                        point_cloud_files.append((class_id, instance_id, pc_path))
                    else:
                        logging.warning(f"Point cloud file not found: {pc_path}")
        
        npz_filenames = point_cloud_files
        logging.info(f"DeepSDF baseline mode: {len(point_cloud_files)} point cloud files found")
    else:
        # Standard DeepSDF mode: use preprocessed SDF data
        npz_filenames = deep_sdf.data.get_instance_filenames(args.data_source, split)

    random.shuffle(npz_filenames)

    logging.debug(decoder)

    err_sum = 0.0
    repeat = 1
    save_latvec_only = False
    rerun = 0

    reconstruction_dir = os.path.join(
        args.experiment_directory, ws.reconstructions_subdir, str(saved_model_epoch)
    )

    if not os.path.isdir(reconstruction_dir):
        os.makedirs(reconstruction_dir)

    reconstruction_meshes_dir = os.path.join(
        reconstruction_dir, ws.reconstruction_meshes_subdir
    )
    if not os.path.isdir(reconstruction_meshes_dir):
        os.makedirs(reconstruction_meshes_dir)

    reconstruction_codes_dir = os.path.join(
        reconstruction_dir, ws.reconstruction_codes_subdir
    )
    if not os.path.isdir(reconstruction_codes_dir):
        os.makedirs(reconstruction_codes_dir)

    for ii, npz in enumerate(npz_filenames):

        if args.baseline_mode and args.point_cloud_dir:
            # Baseline mode: process point cloud files
            class_id, instance_id, pc_path = npz
            instance_name = f"{class_id}_{instance_id}"
            
            logging.debug("loading point cloud {}".format(instance_name))
            
            # Load multi-sweep point cloud data
            try:
                pc_data = np.load(pc_path)
                point_clouds = pc_data['point_clouds']  # Shape: (6, 256, 3)
                
                # Use first sweep for DeepSDF baseline comparison
                single_sweep = point_clouds[0]  # Shape: (256, 3)
                
                logging.info(f"DeepSDF baseline reconstruction for {instance_name} using single sweep")
                
            except Exception as e:
                logging.error(f"Failed to load point cloud {pc_path}: {e}")
                continue
                
        else:
            # Standard mode: process SDF files
            if "npz" not in npz:
                continue

            full_filename = os.path.join(args.data_source, ws.sdf_samples_subdir, npz)
            instance_name = npz[:-4]

            logging.debug("loading {}".format(npz))

            data_sdf = deep_sdf.data.read_sdf_samples_into_ram(full_filename)

        for k in range(repeat):

            if rerun > 1:
                mesh_filename = os.path.join(
                    reconstruction_meshes_dir, instance_name + "-" + str(k + rerun)
                )
                latent_filename = os.path.join(
                    reconstruction_codes_dir, instance_name + "-" + str(k + rerun) + ".pth"
                )
            else:
                mesh_filename = os.path.join(reconstruction_meshes_dir, instance_name)
                latent_filename = os.path.join(
                    reconstruction_codes_dir, instance_name + ".pth"
                )

            if (
                args.skip
                and os.path.isfile(mesh_filename + ".ply")
                and os.path.isfile(latent_filename)
            ):
                continue

            logging.info("reconstructing {}".format(instance_name))

            start = time.time()
            
            if args.baseline_mode and args.point_cloud_dir:
                # DeepSDF baseline reconstruction from point cloud
                err, latent = reconstruct_from_point_cloud(
                    decoder,
                    single_sweep,
                    int(args.iterations),
                    latent_size,
                    0.01,  # statistical prior
                    0.1,   # clamp distance
                    num_samples=8000,
                    lr=5e-3,
                    l2reg=True,
                )
            else:
                # Standard DeepSDF reconstruction from preprocessed SDF samples
                data_sdf[0] = data_sdf[0][torch.randperm(data_sdf[0].shape[0])]
                data_sdf[1] = data_sdf[1][torch.randperm(data_sdf[1].shape[0])]
                
                err, latent = reconstruct(
                    decoder,
                    int(args.iterations),
                    latent_size,
                    data_sdf,
                    0.01,  # [emp_mean,emp_var],
                    0.1,
                    num_samples=8000,
                    lr=5e-3,
                    l2reg=True,
                )
            logging.debug("reconstruct time: {}".format(time.time() - start))
            err_sum += err
            logging.debug("current_error avg: {}".format((err_sum / (ii + 1))))
            logging.debug(ii)

            logging.debug("latent: {}".format(latent.detach().cpu().numpy()))

            decoder.eval()

            if not os.path.exists(os.path.dirname(mesh_filename)):
                os.makedirs(os.path.dirname(mesh_filename))

            if not save_latvec_only:
                start = time.time()
                with torch.no_grad():
                    deep_sdf.mesh.create_mesh(
                        decoder, latent, mesh_filename, N=256, max_batch=int(2 ** 18)
                    )
                logging.debug("total time: {}".format(time.time() - start))

            if not os.path.exists(os.path.dirname(latent_filename)):
                os.makedirs(os.path.dirname(latent_filename))

            torch.save(latent.unsqueeze(0), latent_filename)
