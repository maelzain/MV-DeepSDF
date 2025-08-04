#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import re
import torch
import torch.utils.data as data_utils
import signal
import sys
import os
import logging
import math
import json
import time
import random
import copy
from itertools import product
from sklearn.model_selection import KFold
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import traceback

import deep_sdf
import deep_sdf.workspace as ws


class LearningRateSchedule:
    def get_learning_rate(self, epoch):
        pass


class ConstantLearningRateSchedule(LearningRateSchedule):
    def __init__(self, value):
        self.value = value

    def get_learning_rate(self, epoch):
        return self.value


class StepLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, interval, factor):
        self.initial = initial
        self.interval = interval
        self.factor = factor

    def get_learning_rate(self, epoch):

        return self.initial * (self.factor ** (epoch // self.interval))


class WarmupLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, warmed_up, length):
        self.initial = initial
        self.warmed_up = warmed_up
        self.length = length

    def get_learning_rate(self, epoch):
        if epoch > self.length:
            return self.warmed_up
        return self.initial + (self.warmed_up - self.initial) * epoch / self.length


def get_learning_rate_schedules(specs):

    schedule_specs = specs["LearningRateSchedule"]

    schedules = []

    for schedule_specs in schedule_specs:

        if schedule_specs["Type"] == "Step":
            schedules.append(
                StepLearningRateSchedule(
                    schedule_specs["Initial"],
                    schedule_specs["Interval"],
                    schedule_specs["Factor"],
                )
            )
        elif schedule_specs["Type"] == "Warmup":
            schedules.append(
                WarmupLearningRateSchedule(
                    schedule_specs["Initial"],
                    schedule_specs["Final"],
                    schedule_specs["Length"],
                )
            )
        elif schedule_specs["Type"] == "Constant":
            schedules.append(ConstantLearningRateSchedule(schedule_specs["Value"]))

        else:
            raise Exception(
                'no known learning rate schedule of type "{}"'.format(
                    schedule_specs["Type"]
                )
            )

    return schedules


def save_model(experiment_directory, filename, decoder, epoch):

    model_params_dir = ws.get_model_params_dir(experiment_directory, True)

    torch.save(
        {"epoch": epoch, "model_state_dict": decoder.state_dict()},
        os.path.join(model_params_dir, filename),
    )


def save_optimizer(experiment_directory, filename, optimizer, epoch):

    optimizer_params_dir = ws.get_optimizer_params_dir(experiment_directory, True)

    torch.save(
        {"epoch": epoch, "optimizer_state_dict": optimizer.state_dict()},
        os.path.join(optimizer_params_dir, filename),
    )


def load_optimizer(experiment_directory, filename, optimizer):

    full_filename = os.path.join(
        ws.get_optimizer_params_dir(experiment_directory), filename
    )

    if not os.path.isfile(full_filename):
        raise Exception(
            'optimizer state dict "{}" does not exist'.format(full_filename)
        )

    data = torch.load(full_filename)

    optimizer.load_state_dict(data["optimizer_state_dict"])

    return data["epoch"]


def save_latent_vectors(experiment_directory, filename, latent_vec, epoch):

    latent_codes_dir = ws.get_latent_codes_dir(experiment_directory, True)

    all_latents = latent_vec.state_dict()

    torch.save(
        {"epoch": epoch, "latent_codes": all_latents},
        os.path.join(latent_codes_dir, filename),
    )


# TODO: duplicated in workspace
def load_latent_vectors(experiment_directory, filename, lat_vecs):

    full_filename = os.path.join(
        ws.get_latent_codes_dir(experiment_directory), filename
    )

    if not os.path.isfile(full_filename):
        raise Exception('latent state file "{}" does not exist'.format(full_filename))

    data = torch.load(full_filename)

    if isinstance(data["latent_codes"], torch.Tensor):

        # for backwards compatibility
        if not lat_vecs.num_embeddings == data["latent_codes"].size()[0]:
            raise Exception(
                "num latent codes mismatched: {} vs {}".format(
                    lat_vecs.num_embeddings, data["latent_codes"].size()[0]
                )
            )

        if not lat_vecs.embedding_dim == data["latent_codes"].size()[2]:
            raise Exception("latent code dimensionality mismatch")

        for i, lat_vec in enumerate(data["latent_codes"]):
            lat_vecs.weight.data[i, :] = lat_vec

    else:
        lat_vecs.load_state_dict(data["latent_codes"])

    return data["epoch"]


def save_logs(
    experiment_directory,
    loss_log,
    lr_log,
    timing_log,
    lat_mag_log,
    param_mag_log,
    epoch,
):

    torch.save(
        {
            "epoch": epoch,
            "loss": loss_log,
            "learning_rate": lr_log,
            "timing": timing_log,
            "latent_magnitude": lat_mag_log,
            "param_magnitude": param_mag_log,
        },
        os.path.join(experiment_directory, ws.logs_filename),
    )


def load_logs(experiment_directory):

    full_filename = os.path.join(experiment_directory, ws.logs_filename)

    if not os.path.isfile(full_filename):
        raise Exception('log file "{}" does not exist'.format(full_filename))

    data = torch.load(full_filename)

    return (
        data["loss"],
        data["learning_rate"],
        data["timing"],
        data["latent_magnitude"],
        data["param_magnitude"],
        data["epoch"],
    )


def clip_logs(loss_log, lr_log, timing_log, lat_mag_log, param_mag_log, epoch):

    iters_per_epoch = len(loss_log) // len(lr_log)

    loss_log = loss_log[: (iters_per_epoch * epoch)]
    lr_log = lr_log[:epoch]
    timing_log = timing_log[:epoch]
    lat_mag_log = lat_mag_log[:epoch]
    for n in param_mag_log:
        param_mag_log[n] = param_mag_log[n][:epoch]

    return (loss_log, lr_log, timing_log, lat_mag_log, param_mag_log)


def get_spec_with_default(specs, key, default):
    try:
        return specs[key]
    except KeyError:
        return default


# ============================================================================
# CROSS-VALIDATION IMPLEMENTATION
# ============================================================================

class CrossValidationError(Exception):
    """Custom exception for cross-validation specific errors"""
    pass


def validate_cv_config(specs: Dict[str, Any]) -> None:
    """
    Validate cross-validation configuration parameters.
    
    Args:
        specs: Experiment specifications dictionary
        
    Raises:
        CrossValidationError: If configuration is invalid
    """
    required_keys = ["cross_validation", "cross_validation_hyperparameter_grid"]
    for key in required_keys:
        if key not in specs:
            raise CrossValidationError(f"Missing required CV config: {key}")
    
    # Validate hyperparameter grid
    grid = specs["cross_validation_hyperparameter_grid"]
    if not isinstance(grid, dict) or not grid:
        raise CrossValidationError("cross_validation_hyperparameter_grid must be a non-empty dictionary")
    
    # Validate numeric parameters
    n_folds = specs.get("cross_validation_folds", 3)
    if not isinstance(n_folds, int) or n_folds < 2:
        raise CrossValidationError("cross_validation_folds must be an integer >= 2")
    
    # Validate file source exists
    cv_file_source = specs.get("cross_validation_file_source", "examples/cars/all_instances_fixed.json")
    if not os.path.exists(cv_file_source):
        raise CrossValidationError(f"Cross-validation file source not found: {cv_file_source}")


def generate_hyperparameter_grid(specs: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generate all combinations of hyperparameters for cross-validation.
    
    Args:
        specs: Experiment specifications dictionary
        
    Returns:
        List of hyperparameter combination dictionaries
        
    Raises:
        CrossValidationError: If grid generation fails
    """
    try:
        if "cross_validation_hyperparameter_grid" not in specs:
            return [{}]

        grid = specs["cross_validation_hyperparameter_grid"]
        param_names = list(grid.keys())
        param_values = list(grid.values())

        # Validate that all parameter values are lists
        for name, values in grid.items():
            if not isinstance(values, list):
                raise CrossValidationError(f"Parameter {name} must have a list of values, got {type(values)}")

        # Generate all combinations
        combinations = list(product(*param_values))

        # Convert to list of dictionaries
        hyperparameter_sets = []
        for combo in combinations:
            param_dict = dict(zip(param_names, combo))
            hyperparameter_sets.append(param_dict)

        if not hyperparameter_sets:
            logging.warning("No hyperparameter combinations generated, using empty set")
            return [{}]

        logging.info(f"Generated {len(hyperparameter_sets)} hyperparameter combinations")
        return hyperparameter_sets
        
    except Exception as e:
        raise CrossValidationError(f"Failed to generate hyperparameter grid: {str(e)}")


def apply_hyperparameters_to_specs(specs: Dict[str, Any], hyperparams: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply hyperparameter set to experiment specifications.
    
    Args:
        specs: Base experiment specifications
        hyperparams: Hyperparameter values to apply
        
    Returns:
        Modified specifications dictionary
        
    Raises:
        CrossValidationError: If parameter application fails
    """
    try:
        modified_specs = copy.deepcopy(specs)

        for param_name, param_value in hyperparams.items():
            if param_name == "CodeRegularizationLambda":
                modified_specs["CodeRegularizationLambda"] = param_value
            elif param_name == "LearningRateSchedule_Decoder_Initial":
                if "LearningRateSchedule" in modified_specs and modified_specs["LearningRateSchedule"]:
                    modified_specs["LearningRateSchedule"][0]["Initial"] = param_value
                else:
                    logging.warning(f"Cannot apply {param_name}: LearningRateSchedule not found or empty")
            elif param_name == "ClampingDistance":
                modified_specs["ClampingDistance"] = param_value
            elif param_name == "SamplesPerScene":
                modified_specs["SamplesPerScene"] = param_value
            else:
                # Generic parameter application - add to specs directly
                modified_specs[param_name] = param_value
                logging.info(f"Applied generic parameter: {param_name} = {param_value}")

        # Set reduced epochs for cross-validation if specified
        if "cross_validation_reduced_epochs" in specs:
            reduced_epochs = specs["cross_validation_reduced_epochs"]
            if isinstance(reduced_epochs, int) and reduced_epochs > 0:
                modified_specs["NumEpochs"] = reduced_epochs
                logging.info(f"Set reduced epochs for CV: {reduced_epochs}")

        return modified_specs
        
    except Exception as e:
        raise CrossValidationError(f"Failed to apply hyperparameters: {str(e)}")


def create_cv_splits(specs: Dict[str, Any]) -> List[Tuple[List[str], List[str]]]:
    """
    Create cross-validation data splits using K-Fold.
    
    Args:
        specs: Experiment specifications dictionary
        
    Returns:
        List of (train_split, val_split) tuples
        
    Raises:
        CrossValidationError: If split creation fails
    """
    try:
        cv_file_source = specs.get("cross_validation_file_source", "examples/cars/all_instances_fixed.json")
        n_folds = specs.get("cross_validation_folds", 3)
        random_seed = specs.get("cross_validation_random_seed", 42)

        # Load all data instances
        with open(cv_file_source, "r") as f:
            data = json.load(f)

        # Handle nested structure: {"ShapeNetV2": {"category": [instances]}}
        if isinstance(data, dict):
            all_instances = []
            if "ShapeNetV2" in data:
                for category, instances in data["ShapeNetV2"].items():
                    if isinstance(instances, list):
                        all_instances.extend(instances)
            else:
                # Flat dictionary structure
                for key, value in data.items():
                    if isinstance(value, list):
                        all_instances.extend(value)
        else:
            # Direct list
            all_instances = data

        if not all_instances:
            raise CrossValidationError(f"No instances found in {cv_file_source}")

        if len(all_instances) < n_folds:
            raise CrossValidationError(f"Not enough instances ({len(all_instances)}) for {n_folds} folds")

        # Set random seed for reproducibility
        random.seed(random_seed)
        random.shuffle(all_instances)

        # Create KFold splits
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
        cv_splits = []

        for fold_idx, (train_indices, val_indices) in enumerate(kfold.split(all_instances)):
            train_instances = [all_instances[i] for i in train_indices]
            val_instances = [all_instances[i] for i in val_indices]
            
            # Convert back to expected format: {"ShapeNetV2": {"02958343": [instances]}}
            train_split = {"ShapeNetV2": {"02958343": train_instances}}
            val_split = {"ShapeNetV2": {"02958343": val_instances}}
            
            cv_splits.append((train_split, val_split))
            
            logging.debug(f"Fold {fold_idx}: {len(train_instances)} train, {len(val_instances)} val instances")

        logging.info(f"Created {len(cv_splits)} cross-validation splits")
        return cv_splits
        
    except FileNotFoundError:
        raise CrossValidationError(f"Cross-validation data file not found: {cv_file_source}")
    except json.JSONDecodeError as e:
        raise CrossValidationError(f"Invalid JSON in {cv_file_source}: {str(e)}")
    except Exception as e:
        raise CrossValidationError(f"Failed to create CV splits: {str(e)}")


def cleanup_fold_directory(fold_dir: str, keep_results: bool = True) -> None:
    """
    Clean up fold directory to save disk space.
    
    Args:
        fold_dir: Path to fold directory
        keep_results: Whether to keep essential result files
    """
    try:
        if not os.path.exists(fold_dir):
            return
            
        if keep_results:
            # Keep only essential files: logs, final model, specs
            essential_files = {'logs.pth', 'specs.json', 'latest.pth'}
            
            for item in os.listdir(fold_dir):
                item_path = os.path.join(fold_dir, item)
                if os.path.isfile(item_path) and item not in essential_files:
                    os.remove(item_path)
                elif os.path.isdir(item_path) and item not in {'Reconstructions'}:
                    shutil.rmtree(item_path)
        else:
            shutil.rmtree(fold_dir)
            
    except Exception as e:
        logging.warning(f"Failed to cleanup fold directory {fold_dir}: {str(e)}")


def get_mean_latent_vector_magnitude(latent_vectors):
    return torch.mean(torch.norm(latent_vectors.weight.data.detach(), dim=1))


def append_parameter_magnitudes(param_mag_log, model):
    for name, param in model.named_parameters():
        if len(name) > 7 and name[:7] == "module.":
            name = name[7:]
        if name not in param_mag_log.keys():
            param_mag_log[name] = []
        param_mag_log[name].append(param.data.norm().item())


def main_function(experiment_directory, continue_from, batch_split, cv_specs=None, 
                 train_split_override=None, cv_fold=None, cv_hyperparams=None):
    """
    Main training function with cross-validation support.
    
    Args:
        experiment_directory: Path to experiment directory
        continue_from: Checkpoint to continue from
        batch_split: Batch split index
        cv_specs: Cross-validation modified specs (optional)
        train_split_override: Override train split for CV (optional)
        cv_fold: Current fold number for logging (optional)
        cv_hyperparams: Current hyperparameters for logging (optional)
    """
    logging.debug("running " + experiment_directory)

    # Load specs - use CV specs if provided
    if cv_specs is not None:
        specs = cv_specs
        logging.info(f"Using CV-modified specs for fold {cv_fold}")
    else:
        specs = ws.load_experiment_specifications(experiment_directory)

    # Apply hyperparameter overrides for cross-validation
    if cv_hyperparams is not None:
        specs = apply_hyperparameters_to_specs(specs, cv_hyperparams)
        logging.info(f"Applied hyperparameters: {cv_hyperparams}")

    logging.info("Experiment description: \n" + "\n".join(specs["Description"]))

    data_source = specs["DataSource"]

    # Use train split override for cross-validation or default from specs
    if train_split_override is not None:
        train_split = train_split_override
        logging.info(f"Using CV train split with {len(train_split)} instances")
    else:
        train_split_file = specs["TrainSplit"]

    arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder"])

    logging.debug(specs["NetworkSpecs"])

    latent_size = specs["CodeLength"]

    checkpoints = list(
        range(
            specs["SnapshotFrequency"],
            specs["NumEpochs"] + 1,
            specs["SnapshotFrequency"],
        )
    )

    for checkpoint in specs["AdditionalSnapshots"]:
        checkpoints.append(checkpoint)
    checkpoints.sort()

    lr_schedules = get_learning_rate_schedules(specs)

    grad_clip = get_spec_with_default(specs, "GradientClipNorm", None)
    if grad_clip is not None:
        logging.debug("clipping gradients to max norm {}".format(grad_clip))

    def save_latest(epoch):

        save_model(experiment_directory, "latest.pth", decoder, epoch)
        save_optimizer(experiment_directory, "latest.pth", optimizer_all, epoch)
        save_latent_vectors(experiment_directory, "latest.pth", lat_vecs, epoch)

    def save_checkpoints(epoch):

        save_model(experiment_directory, str(epoch) + ".pth", decoder, epoch)
        save_optimizer(experiment_directory, str(epoch) + ".pth", optimizer_all, epoch)
        save_latent_vectors(experiment_directory, str(epoch) + ".pth", lat_vecs, epoch)

    def signal_handler(sig, frame):
        logging.info("Stopping early...")
        sys.exit(0)

    def adjust_learning_rate(lr_schedules, optimizer, epoch):
        
        for i, param_group in enumerate(optimizer.param_groups):
            # Use the same schedule for both groups if only one schedule is provided
            schedule_idx = min(i, len(lr_schedules) - 1)
            param_group["lr"] = lr_schedules[schedule_idx].get_learning_rate(epoch)

    def empirical_stat(latent_vecs, indices):
        lat_mat = torch.zeros(0).cuda()
        for ind in indices:
            lat_mat = torch.cat([lat_mat, latent_vecs[ind]], 0)
        mean = torch.mean(lat_mat, 0)
        var = torch.var(lat_mat, 0)
        return mean, var

    signal.signal(signal.SIGINT, signal_handler)

    num_samp_per_scene = specs["SamplesPerScene"]
    scene_per_batch = specs["ScenesPerBatch"]
    clamp_dist = specs["ClampingDistance"]
    minT = -clamp_dist
    maxT = clamp_dist
    enforce_minmax = True

    do_code_regularization = get_spec_with_default(specs, "CodeRegularization", True)
    code_reg_lambda = get_spec_with_default(specs, "CodeRegularizationLambda", 1e-7) #H1e-4

    code_bound = get_spec_with_default(specs, "CodeBound", None)

    decoder = arch.Decoder(latent_size, **specs["NetworkSpecs"]).cuda()

    logging.info("training with {} GPU(s)".format(torch.cuda.device_count()))

    # if torch.cuda.device_count() > 1:
    decoder = torch.nn.DataParallel(decoder)

    num_epochs = specs["NumEpochs"]
    log_frequency = get_spec_with_default(specs, "LogFrequency", 10)

    # Load train split if not provided via CV override
    if train_split_override is None:
        with open(train_split_file, "r") as f:
            train_split = json.load(f)

    sdf_dataset = deep_sdf.data.SDFSamples(
        data_source, train_split, num_samp_per_scene, load_ram=False
    )

    num_data_loader_threads = get_spec_with_default(specs, "DataLoaderThreads", 1)
    logging.debug("loading data with {} threads".format(num_data_loader_threads))

    sdf_loader = data_utils.DataLoader(
        sdf_dataset,
        batch_size=scene_per_batch,
        shuffle=True,
        num_workers=num_data_loader_threads,
        drop_last=True,
    )

    logging.debug("torch num_threads: {}".format(torch.get_num_threads()))

    num_scenes = len(sdf_dataset)

    logging.info("There are {} scenes".format(num_scenes))

    logging.debug(decoder)

    lat_vecs = torch.nn.Embedding(num_scenes, latent_size, max_norm=code_bound)
    torch.nn.init.normal_(
        lat_vecs.weight.data,
        0.0,
        get_spec_with_default(specs, "CodeInitStdDev", 1.0) / math.sqrt(latent_size),
    )

    logging.debug(
        "initialized with mean magnitude {}".format(
            get_mean_latent_vector_magnitude(lat_vecs)
        )
    )

    loss_l1 = torch.nn.L1Loss(reduction="sum")

    # Use the same learning rate for both decoder and latent codes when only one schedule is provided
    if len(lr_schedules) == 1:
        optimizer_all = torch.optim.Adam(
            [
                {
                    "params": decoder.parameters(),
                    "lr": lr_schedules[0].get_learning_rate(0),
                },
                {
                    "params": lat_vecs.parameters(),
                    "lr": lr_schedules[0].get_learning_rate(0),
                },
            ]
        )
    else:
        optimizer_all = torch.optim.Adam(
            [
                {
                    "params": decoder.parameters(),
                    "lr": lr_schedules[0].get_learning_rate(0),
                },
                {
                    "params": lat_vecs.parameters(),
                    "lr": lr_schedules[1].get_learning_rate(0),
                },
            ]
        )

    loss_log = []
    lr_log = []
    lat_mag_log = []
    timing_log = []
    param_mag_log = {}

    start_epoch = 1

    if continue_from is not None:

        logging.info('continuing from "{}"'.format(continue_from))

        lat_epoch = load_latent_vectors(
            experiment_directory, continue_from + ".pth", lat_vecs
        )

        model_epoch = ws.load_model_parameters(
            experiment_directory, continue_from, decoder
        )

        optimizer_epoch = load_optimizer(
            experiment_directory, continue_from + ".pth", optimizer_all
        )

        loss_log, lr_log, timing_log, lat_mag_log, param_mag_log, log_epoch = load_logs(
            experiment_directory
        )

        if not log_epoch == model_epoch:
            loss_log, lr_log, timing_log, lat_mag_log, param_mag_log = clip_logs(
                loss_log, lr_log, timing_log, lat_mag_log, param_mag_log, model_epoch
            )

        if not (model_epoch == optimizer_epoch and model_epoch == lat_epoch):
            raise RuntimeError(
                "epoch mismatch: {} vs {} vs {} vs {}".format(
                    model_epoch, optimizer_epoch, lat_epoch, log_epoch
                )
            )

        start_epoch = model_epoch + 1

        logging.debug("loaded")

    logging.info("starting from epoch {}".format(start_epoch))

    logging.info(
        "Number of decoder parameters: {}".format(
            sum(p.data.nelement() for p in decoder.parameters())
        )
    )
    logging.info(
        "Number of shape code parameters: {} (# codes {}, code dim {})".format(
            lat_vecs.num_embeddings * lat_vecs.embedding_dim,
            lat_vecs.num_embeddings,
            lat_vecs.embedding_dim,
        )
    )

    for epoch in range(start_epoch, num_epochs + 1):

        start = time.time()

        logging.info("epoch {}...".format(epoch))

        decoder.train()

        adjust_learning_rate(lr_schedules, optimizer_all, epoch)

        for sdf_data, indices in sdf_loader:

            # Process the input data
            sdf_data = sdf_data.reshape(-1, 4)

            num_sdf_samples = sdf_data.shape[0]

            sdf_data.requires_grad = False

            xyz = sdf_data[:, 0:3]
            sdf_gt = sdf_data[:, 3].unsqueeze(1)

            if enforce_minmax:
                sdf_gt = torch.clamp(sdf_gt, minT, maxT)

            xyz = torch.chunk(xyz, batch_split)
            indices = torch.chunk(
                indices.unsqueeze(-1).repeat(1, num_samp_per_scene).view(-1),
                batch_split,
            )

            sdf_gt = torch.chunk(sdf_gt, batch_split)

            batch_loss = 0.0

            optimizer_all.zero_grad()

            for i in range(batch_split):

                batch_vecs = lat_vecs(indices[i])

                input = torch.cat([batch_vecs, xyz[i]], dim=1)

                # NN optimization
                pred_sdf = decoder(input)

                if enforce_minmax:
                    pred_sdf = torch.clamp(pred_sdf, minT, maxT)

                chunk_loss = loss_l1(pred_sdf, sdf_gt[i].cuda()) / num_sdf_samples

                if do_code_regularization:
                    l2_size_loss = torch.sum(torch.norm(batch_vecs, dim=1))
                    reg_loss = (
                        code_reg_lambda * min(1, epoch / 100) * l2_size_loss
                    ) / num_sdf_samples

                    chunk_loss = chunk_loss + reg_loss.cuda()

                chunk_loss.backward()

                batch_loss += chunk_loss.item()

            logging.debug("loss = {}".format(batch_loss))

            loss_log.append(batch_loss)

            if grad_clip is not None:

                torch.nn.utils.clip_grad_norm_(decoder.parameters(), grad_clip)

            optimizer_all.step()

        end = time.time()

        seconds_elapsed = end - start
        timing_log.append(seconds_elapsed)

        lr_log.append([schedule.get_learning_rate(epoch) for schedule in lr_schedules])

        lat_mag_log.append(get_mean_latent_vector_magnitude(lat_vecs))

        append_parameter_magnitudes(param_mag_log, decoder)

        if epoch in checkpoints:
            save_checkpoints(epoch)

        if epoch % log_frequency == 0:

            save_latest(epoch)
            save_logs(
                experiment_directory,
                loss_log,
                lr_log,
                timing_log,
                lat_mag_log,
                param_mag_log,
                epoch,
            )


# ============================================================================
# CROSS-VALIDATION PERFORMANCE EVALUATION
# ============================================================================

def evaluate_fold_performance(fold_dir: str, val_split: List[str]) -> float:
    """
    Evaluate performance of a completed fold.
    
    Args:
        fold_dir: Path to fold directory
        val_split: Validation split instances (for future validation evaluation)
        
    Returns:
        Performance metric (lower is better)
    """
    try:
        # Try both possible log file names
        logs_files = [
            os.path.join(fold_dir, "Logs.pth"),
            os.path.join(fold_dir, ws.logs_filename) if hasattr(ws, 'logs_filename') else None
        ]
        
        for logs_file in logs_files:
            if logs_file and os.path.exists(logs_file):
                data = torch.load(logs_file, weights_only=False)
                if "loss" in data and data["loss"] and len(data["loss"]) > 0:
                    # Use final training loss as performance metric
                    # TODO: Implement proper validation loss evaluation
                    final_loss = data["loss"][-1]
                    logging.debug(f"Fold performance: {final_loss}")
                    return final_loss
                else:
                    logging.warning(f"No loss data found in {logs_file}")
        
        logging.warning(f"No valid logs file found in {fold_dir}")
        return float('inf')
    except Exception as e:
        logging.error(f"Error evaluating fold performance: {str(e)}")
        return float('inf')


def run_cross_validation(experiment_directory: str, continue_from: Optional[str], batch_split: int) -> Dict[str, Any]:
    """
    Run cross-validation with hyperparameter grid search.
    
    Args:
        experiment_directory: Path to experiment directory
        continue_from: Checkpoint to continue from (not used in CV)
        batch_split: Batch split index
        
    Returns:
        Dictionary containing best hyperparameters and results
        
    Raises:
        CrossValidationError: If cross-validation fails
    """
    try:
        # Load and validate base specs
        specs = ws.load_experiment_specifications(experiment_directory)
        
        # Check if cross-validation is enabled
        if "cross_validation" not in specs:
            logging.info("Cross-validation not enabled, running standard training")
            main_function(experiment_directory, continue_from, batch_split)
            return {}

        logging.info("=" * 80)
        logging.info(f"STARTING CROSS-VALIDATION: {specs['cross_validation']}")
        logging.info("=" * 80)

        # Validate configuration
        validate_cv_config(specs)

        # Generate hyperparameter grid
        hyperparameter_sets = generate_hyperparameter_grid(specs)
        logging.info(f"Testing {len(hyperparameter_sets)} hyperparameter combinations")

        # Create cross-validation splits
        cv_splits = create_cv_splits(specs)
        n_folds = len(cv_splits)
        logging.info(f"Using {n_folds}-fold cross-validation")

        # Results tracking
        cv_results = []
        best_mean_loss = float('inf')
        best_result = None

        # Main cross-validation loop
        for hp_idx, hyperparams in enumerate(hyperparameter_sets):
            logging.info("=" * 60)
            logging.info(f"HYPERPARAMETER SET {hp_idx + 1}/{len(hyperparameter_sets)}")
            logging.info(f"Parameters: {hyperparams}")
            logging.info("=" * 60)

            fold_losses = []
            fold_dirs = []

            for fold_idx, (train_split, val_split) in enumerate(cv_splits):
                logging.info(f"--- Fold {fold_idx + 1}/{n_folds} ---")

                # Create fold-specific experiment directory
                fold_exp_dir = os.path.join(experiment_directory, f"cv_hp{hp_idx:02d}_fold{fold_idx}")
                fold_dirs.append(fold_exp_dir)
                os.makedirs(fold_exp_dir, exist_ok=True)

                # Apply hyperparameters to specs
                fold_specs = apply_hyperparameters_to_specs(specs, hyperparams)

                # Save modified specs for this fold
                specs_file = os.path.join(fold_exp_dir, "specs.json")
                with open(specs_file, "w") as f:
                    json.dump(fold_specs, f, indent=4)

                try:
                    # Run training for this fold
                    main_function(
                        fold_exp_dir,
                        continue_from,
                        batch_split,
                        cv_specs=fold_specs,
                        train_split_override=train_split,
                        cv_fold=fold_idx,
                        cv_hyperparams=hyperparams
                    )

                    # Evaluate fold performance
                    fold_loss = evaluate_fold_performance(fold_exp_dir, val_split)
                    fold_losses.append(fold_loss)
                    
                    logging.info(f"Fold {fold_idx + 1} completed with loss: {fold_loss}")

                except Exception as e:
                    logging.error(f"Error in fold {fold_idx + 1}: {str(e)}")
                    logging.error(f"Traceback: {traceback.format_exc()}")
                    fold_losses.append(float('inf'))

                # Cleanup intermediate files to save space
                cleanup_fold_directory(fold_exp_dir, keep_results=True)

            # Calculate statistics for this hyperparameter set
            valid_losses = [loss for loss in fold_losses if loss != float('inf')]
            if valid_losses:
                mean_cv_loss = sum(valid_losses) / len(valid_losses)
                std_cv_loss = (sum((x - mean_cv_loss) ** 2 for x in valid_losses) / len(valid_losses)) ** 0.5
            else:
                mean_cv_loss = float('inf')
                std_cv_loss = float('inf')

            result = {
                'hyperparams': hyperparams,
                'fold_losses': fold_losses,
                'mean_cv_loss': mean_cv_loss,
                'std_cv_loss': std_cv_loss,
                'valid_folds': len(valid_losses),
                'fold_dirs': fold_dirs
            }
            cv_results.append(result)

            logging.info(f"Hyperparameter set {hp_idx + 1} completed:")
            logging.info(f"  Mean CV Loss: {mean_cv_loss:.6f} ± {std_cv_loss:.6f}")
            logging.info(f"  Valid folds: {len(valid_losses)}/{n_folds}")

            # Track best result
            if mean_cv_loss < best_mean_loss:
                best_mean_loss = mean_cv_loss
                best_result = result
                logging.info(f"  *** NEW BEST RESULT ***")

        # Save comprehensive cross-validation results
        final_results = {
            'best_hyperparams': best_result['hyperparams'] if best_result else {},
            'best_mean_cv_loss': best_mean_loss,
            'best_std_cv_loss': best_result['std_cv_loss'] if best_result else float('inf'),
            'all_results': cv_results,
            'cv_config': {
                'n_folds': n_folds,
                'hyperparameter_sets': len(hyperparameter_sets),
                'data_source': specs.get("cross_validation_file_source"),
                'random_seed': specs.get("cross_validation_random_seed", 42)
            },
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }

        cv_results_file = os.path.join(experiment_directory, "cv_results.json")
        with open(cv_results_file, "w") as f:
            json.dump(final_results, f, indent=4)

        # Log final summary
        logging.info("=" * 80)
        logging.info("CROSS-VALIDATION COMPLETED")
        logging.info("=" * 80)
        if best_result:
            logging.info(f"Best hyperparameters: {best_result['hyperparams']}")
            logging.info(f"Best mean CV loss: {best_mean_loss:.6f} ± {best_result['std_cv_loss']:.6f}")
            logging.info(f"Valid folds: {best_result['valid_folds']}/{n_folds}")
        else:
            logging.warning("No valid results found")
        logging.info(f"Results saved to: {cv_results_file}")
        logging.info("=" * 80)

        return final_results

    except CrossValidationError as e:
        logging.error(f"Cross-validation configuration error: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error in cross-validation: {str(e)}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        raise CrossValidationError(f"Cross-validation failed: {str(e)}")


if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser(description="Train a DeepSDF autodecoder")
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory. This directory should include "
        + "experiment specifications in 'specs.json', and logging will be "
        + "done in this directory as well.",
    )
    arg_parser.add_argument(
        "--continue",
        "-c",
        dest="continue_from",
        help="A snapshot to continue from. This can be 'latest' to continue"
        + "from the latest running snapshot, or an integer corresponding to "
        + "an epochal snapshot.",
    )
    arg_parser.add_argument(
        "--batch_split",
        dest="batch_split",
        default=1,
        help="This splits the batch into separate subbatches which are "
        + "processed separately, with gradients accumulated across all "
        + "subbatches. This allows for training with large effective batch "
        + "sizes in memory constrained environments.",
    )
    arg_parser.add_argument(
        "--cross_validation",
        "-cv",
        dest="cross_validation",
        action="store_true",
        help="Run cross-validation with hyperparameter grid search instead of regular training. "
        + "Requires 'cross_validation' and 'cross_validation_hyperparameter_grid' in specs.json.",
    )

    deep_sdf.add_common_args(arg_parser)

    args = arg_parser.parse_args()

    deep_sdf.configure_logging(args)

    # Run cross-validation or regular training based on flag
    if args.cross_validation:
        try:
            results = run_cross_validation(args.experiment_directory, args.continue_from, int(args.batch_split))
            logging.info("Cross-validation completed successfully")
        except CrossValidationError as e:
            logging.error(f"Cross-validation failed: {str(e)}")
            sys.exit(1)
        except Exception as e:
            logging.error(f"Unexpected error: {str(e)}")
            sys.exit(1)
    else:
        main_function(args.experiment_directory, args.continue_from, int(args.batch_split))
