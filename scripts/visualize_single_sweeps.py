#!/usr/bin/env python3
"""
Visualize single-sweep point clouds for the report.
Creates publication-quality visualizations showing the realistic random LiDAR viewpoints.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
from pathlib import Path

def save_point_cloud_as_ply(point_cloud, save_path, instance_id=""):
    """Save point cloud as PLY file for MeshLab visualization."""
    # PLY header
    header = f"""ply
format ascii 1.0
comment Single-sweep LiDAR point cloud: {instance_id}
comment Realistic random street viewpoint - Uniform red color
element vertex {len(point_cloud)}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
    
    # Use uniform red color (high visibility and contrast for research visualization)
    # Red: RGB(255, 0, 0) - high contrast, easily visible
    red_r, red_g, red_b = 255, 0, 0
    
    with open(save_path, 'w') as f:
        f.write(header)
        for i in range(len(point_cloud)):
            x, y, z = point_cloud[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {red_r} {red_g} {red_b}\n")
    
    print(f"Saved PLY file: {save_path}")

def main():
    # Load test instances
    with open('configs/mvdeepsdf_stage2_test.json', 'r') as f:
        test_instances = json.load(f)['ShapeNetV2']['02958343']
    
    single_sweep_dir = Path('experiments/single_sweep_data')
    vis_dir = Path('visualizations/single_sweeps_ply')
    vis_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"Creating PLY files of single-sweep point clouds for MeshLab visualization...")
    
    # Generate PLY files for all test instances
    sample_instances = test_instances  # All 506 test instances
    
    for i, instance_id in enumerate(sample_instances):
        file_path = single_sweep_dir / f"02958343_{instance_id}.npz"
        
        if file_path.exists():
            try:
                # Load single-sweep data
                data = np.load(file_path)
                point_clouds = data['point_clouds']  # Shape: (1, 256, 3)
                single_sweep = point_clouds[0]       # Shape: (256, 3)
                
                # Create PLY file for MeshLab
                ply_path = vis_dir / f"single_sweep_{i+1:02d}_{instance_id[:8]}.ply"
                save_point_cloud_as_ply(single_sweep, ply_path, f"Instance {i+1} - {instance_id}")
                
                # Print statistics
                print(f"Instance {i+1:2d} ({instance_id[:8]}):")
                print(f"  Points: {single_sweep.shape[0]}")
                print(f"  Range X: [{single_sweep[:, 0].min():.3f}, {single_sweep[:, 0].max():.3f}]")
                print(f"  Range Y: [{single_sweep[:, 1].min():.3f}, {single_sweep[:, 1].max():.3f}]")
                print(f"  Range Z: [{single_sweep[:, 2].min():.3f}, {single_sweep[:, 2].max():.3f}]")
                print()
                
            except Exception as e:
                print(f"Error processing {instance_id}: {e}")
        else:
            print(f"File not found: {file_path}")
    
    print(f"\n🎉 PLY files created successfully!")
    print(f"   Output directory: {vis_dir}")
    print(f"   Files created:")
    for file in vis_dir.glob("*.ply"):
        print(f"     {file.name}")
    print(f"\n📋 Instructions for MeshLab:")
    print(f"   1. Open MeshLab")
    print(f"   2. File > Import Mesh > Select any .ply file from {vis_dir}")
    print(f"   3. The point cloud will show in uniform red color")
    print(f"   4. Use mouse to rotate and examine the realistic random LiDAR viewpoints")
    print(f"   5. Each file represents a single LiDAR sweep (not multiple sweeps)")

if __name__ == "__main__":
    main()