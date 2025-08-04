import numpy as np
from pathlib import Path
import argparse

def save_point_cloud_to_ply(points: np.ndarray, file_path: str):
    num_points = points.shape[0]
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {num_points}",
        "property float x",
        "property float y",
        "property float z",
        "end_header"
    ]
    with open(file_path, 'w') as f:
        f.write('\n'.join(header) + '\n')
        np.savetxt(f, points, fmt='%.6f')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Overlay all 6 partial views for a specific instance into one .ply file.")
    parser.add_argument('--data_root', required=True, help='Directory containing the .npz files (e.g., dataa/MultiSweepPointClouds)')
    parser.add_argument('--class_id', required=True, help='Class ID (e.g., 02958343)')
    parser.add_argument('--instance_id', required=True, help='Instance ID (e.g., 100715345ee54d7ae38b52b4ee9d36a3)')
    parser.add_argument('--output_ply', required=True, help='Output .ply file path')
    args = parser.parse_args()

    npz_path = Path(args.data_root) / f"{args.class_id}_{args.instance_id}.npz"
    data = np.load(npz_path)
    key = 'point_cloud' if 'point_cloud' in data else 'point_clouds'
    point_clouds = data[key]  # shape (6, N, 3)
    all_points = point_clouds.reshape(-1, 3)  # shape (6*N, 3)
    save_point_cloud_to_ply(all_points, args.output_ply)
    print(f"Saved combined point cloud to {args.output_ply}") 