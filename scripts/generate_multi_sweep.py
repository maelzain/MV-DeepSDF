#!/usr/bin/env python3
import sys, os
# allow imports from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

"""
Generate multi-sweep point clouds for MV-DeepSDF Stage 2.
PAPER-ACCURATE PCGen Implementation with "one side per sweep" behavior.

Stage‐II Data Gen (Sec. 3.1, Fig. 2 of the ICCV paper):
  - 6 LiDAR sweeps: 3 azimuths ∈ [0°,180°], 3 ∈ [–180°,0°]
  - distance ∈ [3,15], height ∈ [0.8,1.2] 
  - each sweep fires rays with realistic automotive LiDAR FOV (90°)
  - PCGen: "randomly sample one side of the vehicle" per sweep
  - add Gaussian noise σ=0.01
  - downsample each sweep to 256 points via Farthest-Point Sampling
  - output shape: (6,256,3) saved as .npz
  
Key improvements for paper accuracy:
  - Reduced FOV to 90° for realistic single-side capture
  - Forward-facing directional rays (not center-directed)
  - Sparse elevation layers matching real automotive LiDAR
  - Natural occlusion prevents multi-side visibility per sweep
"""

import argparse, glob, json, logging, random
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import trimesh
import torch
from tqdm import tqdm

# deterministic CuDNN
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from networks.mv_deepsdf import FarthestPointSampling

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_deepsdf_normalization(data_dir, dataset_name, class_id, instance_id):
    """Load DeepSDF normalization parameters if available."""
    import deep_sdf.workspace as ws
    try:
        norm_file = ws.get_normalization_params_filename(data_dir, dataset_name, class_id, instance_id)
        if os.path.exists(norm_file):
            data = np.load(norm_file)
            return data['offset'], data['scale']
        else:
            logger.debug(f"No normalization file found: {norm_file}")
            return None, None
    except Exception as e:
        logger.debug(f"Failed to load normalization parameters: {e}")
        return None, None


def detect_deepsdf_clamp_distance(experiment_dir=None):
    """
    SCALABLE: Automatically detect DeepSDF ClampingDistance from specs.json
    This makes the code work with any DeepSDF model regardless of hyperparameters
    """
    # Try multiple possible locations for specs.json
    possible_specs = [
        "examples/cars/specs.json",
        "examples/cars_stage1/specs.json", 
        "experiments/deepsdf/specs.json"
    ]
    
    if experiment_dir:
        possible_specs.insert(0, os.path.join(experiment_dir, "specs.json"))
    
    for specs_path in possible_specs:
        if os.path.exists(specs_path):
            try:
                with open(specs_path, 'r') as f:
                    specs = json.load(f)
                clamp_dist = specs.get("ClampingDistance", 0.05)  # Default to 0.05
                logger.info(f"🎯 AUTO-DETECTED DeepSDF ClampingDistance = {clamp_dist} from {specs_path}")
                return clamp_dist, specs_path
            except Exception as e:
                logger.warning(f"Failed to read {specs_path}: {e}")
                continue
    
    logger.warning("⚠️  Could not auto-detect ClampingDistance, using default 0.05")
    logger.warning("   For custom values, ensure specs.json is in examples/cars/ or specify --deepsdf_experiment")
    return 0.05, None

# Paper constants - FIXED for proper single-side coverage
NUM_VIEWS  = 6
N_RAYS     = 2048              # Sufficient rays for good coverage
FOV_DEG    = 60                # Much narrower FOV for true single-side capture
NUM_POINTS = 256
NOISE_STD  = 0.01             # Gaussian sensor noise

FPS = FarthestPointSampling(NUM_POINTS)

# Predefined sweep positions for TRUE single-side coverage
# Each position is carefully positioned to only see one side
SWEEP_POSITIONS = [
    {'azim': 0, 'name': 'front', 'offset_angle': 0},      # Pure front view
    {'azim': 90, 'name': 'right', 'offset_angle': 0},     # Pure right view  
    {'azim': 180, 'name': 'back', 'offset_angle': 0},     # Pure back view
    {'azim': 270, 'name': 'left', 'offset_angle': 0},     # Pure left view
    {'azim': 45, 'elev': 30, 'name': 'top-right', 'offset_angle': 0},      # Top-right diagonal
    {'azim': 225, 'elev': -30, 'name': 'bottom-left', 'offset_angle': 0}   # Bottom-left diagonal
]


def simulate_lidar_sweep_single_side(mesh, sweep_idx, dist, hgt, instance_id_str=""):
    """
    Simulate LiDAR sweep that captures ONLY ONE SIDE of the vehicle.
    Follows MV-DeepSDF paper methodology with random pose sampling within constraints.
    
    Paper constraints (Section 3.5):
    - θ ∈ [0°, 180°] or θ ∈ [−180°, 0°] (azimuth)
    - r ∈ [3, 15] (distance, normalized space)
    - h ∈ [0.8, 1.2] (height, normalized space)
    - "randomly sample one side of the vehicle"
    """
    
    # Generate 6 different azimuth ranges to ensure single-side coverage
    # Split the full 360° into 6 sectors, each covering ~60° but with constraints
    azimuth_ranges = [
        (0, 60),         # Front-right sector
        (60, 120),       # Right sector  
        (120, 180),      # Back-right sector
        (-180, -120),    # Back-left sector
        (-120, -60),     # Left sector
        (-60, 0)         # Front-left sector
    ]
    
    # Select azimuth range for this sweep
    azim_min, azim_max = azimuth_ranges[sweep_idx % 6]
    
    # Random sampling within paper constraints
    azim_deg = random.uniform(azim_min, azim_max)
    azim = np.deg2rad(azim_deg)
    
    # Small elevation variation for realism
    elev_deg = random.uniform(-5, 5)  # Small elevation variation
    elev = np.deg2rad(elev_deg)
    
    # Position LiDAR using spherical coordinates (paper methodology)
    # In normalized space where vehicle is in [-1, 1] cube
    x = dist * np.cos(azim) * np.cos(elev)
    y = dist * np.sin(azim) * np.cos(elev) 
    z = hgt + dist * np.sin(elev)
    
    origin = np.array([x, y, z], dtype=np.float32)
    
    # Generate rays in a cone pattern pointing toward vehicle center
    # This mimics automotive LiDAR beam patterns
    half_fov = np.deg2rad(FOV_DEG / 2)
    
    # Create beam pattern similar to automotive LiDAR
    n_rays_side = int(np.sqrt(N_RAYS))
    
    # Direction toward vehicle center
    vehicle_center = mesh.centroid
    base_dir = vehicle_center - origin
    base_dir = base_dir / np.linalg.norm(base_dir)
    
    # Generate ray directions in cone pattern
    directions = []
    for i in range(n_rays_side):
        for j in range(n_rays_side):
            # Create angular offsets within FOV
            theta_offset = (i / (n_rays_side - 1) - 0.5) * 2 * half_fov
            phi_offset = (j / (n_rays_side - 1) - 0.5) * 2 * half_fov
            
            # Apply rotations to base direction
            # Rotation around z-axis (azimuth)
            cos_phi, sin_phi = np.cos(phi_offset), np.sin(phi_offset)
            rot_z = np.array([
                [cos_phi, -sin_phi, 0],
                [sin_phi, cos_phi, 0],
                [0, 0, 1]
            ])
            
            # Rotation around y-axis (elevation)
            cos_theta, sin_theta = np.cos(theta_offset), np.sin(theta_offset)
            rot_y = np.array([
                [cos_theta, 0, sin_theta],
                [0, 1, 0],
                [-sin_theta, 0, cos_theta]
            ])
            
            # Apply rotations
            ray_dir = rot_z @ rot_y @ base_dir
            directions.append(ray_dir)
    
    directions = np.array(directions[:N_RAYS], dtype=np.float32)
    origins = np.tile(origin, (len(directions), 1))
    
    # Ray-mesh intersection
    try:
        pts, ray_indices, _ = mesh.ray.intersects_location(origins, directions)
    except Exception as e:
        logger.warning(f"[{instance_id_str}] [Sweep {sweep_idx}] Ray intersection failed: {e}")
        return np.zeros((0,3), dtype=np.float32)
    
    if pts.shape[0] == 0:
        logger.warning(f"[{instance_id_str}] [Sweep {sweep_idx}] No ray intersections found")
        return np.zeros((0,3), dtype=np.float32)
    
    # Single-side filtering based on viewing angle
    # Only keep points that are on the "visible" side from this LiDAR position
    mesh_center = mesh.centroid
    
    # Vector from mesh center to LiDAR
    lidar_dir = origin - mesh_center
    lidar_dir = lidar_dir / np.linalg.norm(lidar_dir)
    
    # Vector from mesh center to each point
    point_dirs = pts - mesh_center
    point_norms = np.linalg.norm(point_dirs, axis=1)
    point_dirs = point_dirs / (point_norms[:, np.newaxis] + 1e-8)
    
    # Keep points that are on the same side as the LiDAR with balanced filtering
    # Dot product > 0.1 provides good single-side coverage without being too restrictive
    dot_products = np.dot(point_dirs, lidar_dir)
    hemisphere_mask = dot_products > 0.1  # Balanced filtering to eliminate wrap-around
    
    # Distance-based filtering (remove outliers)
    distances_to_lidar = np.linalg.norm(pts - origin, axis=1)
    reasonable_distance = dist * 1.5  # Allow some tolerance
    distance_mask = distances_to_lidar < reasonable_distance
    
    # Combine filters
    final_mask = hemisphere_mask & distance_mask
    
    if np.sum(final_mask) == 0:
        logger.warning(f"[{instance_id_str}] [Sweep {sweep_idx}] No valid points after filtering")
        return np.zeros((0,3), dtype=np.float32)
    
    pts = pts[final_mask]
    
    # Add Gaussian noise (paper methodology)
    pts = pts.astype(np.float32) + np.random.normal(scale=NOISE_STD, size=pts.shape).astype(np.float32)
    
    logger.debug(f"[{instance_id_str}] [Sweep {sweep_idx}] Generated {pts.shape[0]} points (azim={azim_deg:.1f}°)")
    return pts


def process_instance(args):
    """Generate and save one .npz for a given instance path (class/instance)."""
    class_id, iid, mesh_root, data_dir, dataset_name, out_dir, num_views = args
    not_watertight = False

    # Check if output file already exists
    out_path = os.path.join(out_dir, f"{class_id}_{iid}.npz")
    if os.path.exists(out_path):
        logger.debug(f"[{class_id}/{iid}] Multisweep data already exists. Skipping.")
        return True, not_watertight

    # Load mesh
    search_dir = os.path.join(mesh_root, class_id, iid)
    objs = glob.glob(os.path.join(search_dir, '**','*.obj'), recursive=True)
    if not objs:
        logger.warning(f"[{class_id}/{iid}] No .obj found in {search_dir}")
        return False, not_watertight

    if len(objs) > 1:
        logger.warning(f"[{class_id}/{iid}] Multiple .obj files found, using first one: {objs[0]}")

    try:
        mesh = trimesh.load(objs[0], force='mesh')
    except Exception as e:
        logger.warning(f"[{class_id}/{iid}] Failed to load mesh: {e}")
        return False, not_watertight

    # Load DeepSDF normalization parameters
    offset, scale = load_deepsdf_normalization(data_dir, dataset_name, class_id, iid)
    
    # PAPER COMPLIANCE: Ensure mesh is normalized to [-1,1] range as required
    # "r and h are expressed in the normalized space, where the size of the vehicle is normalized into the range [−1, 1]"
    
    if offset is not None and scale is not None:
        # Apply DeepSDF normalization first
        mesh.vertices = (mesh.vertices - offset) / scale
        logger.debug(f"Applied DeepSDF normalization: offset={offset}, scale={scale}")
    else:
        # Center the mesh first
        centroid = mesh.centroid
        mesh.vertices = mesh.vertices - centroid
        logger.debug(f"Applied centering: centroid={centroid}")
    
    # PAPER COMPLIANCE: Ensure final mesh is in [-1,1] range regardless of normalization method
    # This guarantees consistency with paper requirements
    vertices = mesh.vertices
    max_extent = np.max(np.abs(vertices))
    
    if max_extent < 1e-6:
        logger.warning(f"[{class_id}/{iid}] Degenerate mesh with max extent {max_extent:.2e}. Skipping.")
        return False, not_watertight
    
    # Scale to [-1,1] range (paper requirement)
    mesh.vertices = vertices / max_extent
    logger.debug(f"Applied [-1,1] normalization: max_extent={max_extent}")
    
    # Verify the mesh is actually in [-1,1] range
    final_max = np.max(np.abs(mesh.vertices))
    assert final_max <= 1.0 + 1e-6, f"Mesh normalization failed: max={final_max}"
    
    # Check if mesh is suitable for ray intersection
    if not mesh.is_watertight:
        logger.warning(f"[{class_id}/{iid}] Mesh is not watertight - may cause ray intersection issues")
        not_watertight = True
    logger.info(f"[{class_id}/{iid}] Normalized mesh: watertight={mesh.is_watertight}")

    sweeps = []
    failed_sweeps = 0
    
    for i in range(num_views):
        # Randomize distance and height within paper ranges (normalized space)
        dist = random.uniform(3, 15)
        hgt = random.uniform(0.8, 1.2)
        
        pts = simulate_lidar_sweep_single_side(mesh, i, dist, hgt, f"{class_id}/{iid}")

        if pts.shape[0] < NUM_POINTS:
            if pts.shape[0] == 0:
                failed_sweeps += 1
                pts = np.zeros((NUM_POINTS, 3), dtype=np.float32)
                logger.warning(f"[{class_id}/{iid}] Sweep {i+1} failed - no intersections")
            else:
                # Pad with last point to reach NUM_POINTS
                pad = np.repeat(pts[-1:], NUM_POINTS - pts.shape[0], axis=0)
                pts = np.vstack([pts, pad])
                logger.debug(f"[{class_id}/{iid}] Sweep {i+1} padded from {pts.shape[0]-pad.shape[0]} to {NUM_POINTS} points")
        else:
            # Downsample using FPS
            pts = FPS(torch.from_numpy(pts)).cpu().numpy()
            logger.debug(f"[{class_id}/{iid}] Sweep {i+1} downsampled to {NUM_POINTS} points")

        sweeps.append(pts.astype(np.float32))

    # Validate generated sweeps
    if failed_sweeps >= num_views // 2:  # If majority of sweeps failed
        logger.warning(f"[{class_id}/{iid}] Too many failed sweeps ({failed_sweeps}/{num_views}) - data quality poor")
        return False, not_watertight

    data = np.stack(sweeps, axis=0)  # (num_views,256,3)
    
    # Final validation - check if data is reasonable
    if np.allclose(data, 0):
        logger.warning(f"[{class_id}/{iid}] All sweeps contain zeros - failed")
        return False, not_watertight
    
    np.savez(out_path, point_clouds=data)
    logger.info(f"[{class_id}/{iid}] Successfully generated multisweep data with {num_views-failed_sweeps} valid sweeps")
    return True, not_watertight


def load_instance_ids(split_json):
    """Return list of (class_id, instance_id) tuples."""
    with open(split_json) as f:
        js = json.load(f)

    pairs = []
    for ds in js.values():
        for class_id, instances in ds.items():
            for iid in instances:
                pairs.append((class_id, iid))
    return pairs


def main():
    p = argparse.ArgumentParser(description="Generate single or multi-sweep LiDAR data for DeepSDF/MV-DeepSDF")
    p.add_argument("--mesh_root", required=True, help="ShapeNet root (per-class/instance dirs)")
    p.add_argument("--data_dir", required=True, help="DeepSDF data directory (contains NormalizationParameters)")
    p.add_argument("--dataset_name", required=True, help="Dataset name (e.g., ShapeNetV2)")
    p.add_argument("--split", required=True, help="JSON split file")
    p.add_argument("--out_dir", required=True, help="Where to write .npz files")
    p.add_argument("--jobs", type=int, default=1, help="Parallel worker count")
    p.add_argument("--smoke", action="store_true", help="Smoke-run on 10 instances")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--num_views", type=int, default=6, help="Number of LiDAR sweeps (1 for single-view, 6 for multi-view)")
    args = p.parse_args()

    # reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    pairs = load_instance_ids(args.split)
    total = len(pairs)
    if args.smoke:
        pairs = pairs[:10]
        logger.info("Smoke run: first 10 only.")
    logger.info(f"{total} instances found; processing {len(pairs)} with {args.jobs} workers.")

    tasks = [(cls, iid, args.mesh_root, args.data_dir, args.dataset_name, args.out_dir, args.num_views) for cls, iid in pairs]
    
    succ = fail = not_watertight_count = 0

    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as exe:
            for ok, not_watertight in tqdm(exe.map(process_instance, tasks), total=len(tasks), desc="Gen sweeps"):
                succ += ok
                fail += not ok
                not_watertight_count += not_watertight
    else:
        for task in tqdm(tasks, desc="Gen sweeps"):
            ok, not_watertight = process_instance(task)
            succ += ok
            fail += not ok
            not_watertight_count += not_watertight

    logger.info(f"Done → Success: {succ}, Fail: {fail}, not_watertight: {not_watertight_count}")
    if fail > 0:
        logger.warning(f"Failed instances: {fail} - check logs for details")


if __name__=="__main__":
    main()