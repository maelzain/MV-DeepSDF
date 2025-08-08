#!/usr/bin/env python3
"""
Extract single sweeps from existing multi-sweep data for fair DeepSDF comparison.
This maintains the realistic random LiDAR positioning while ensuring we have valid data.
"""

import json
import numpy as np
import os
from pathlib import Path

def main():
    # Load test instances
    with open('configs/mvdeepsdf_stage2_test.json', 'r') as f:
        test_instances = json.load(f)['ShapeNetV2']['02958343']
    
    multisweep_dir = Path('experiments/multisweep_data_final')
    output_dir = Path('experiments/single_sweep_data')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    successful = 0
    missing = 0
    
    print(f"Extracting single sweeps for {len(test_instances)} test instances...")
    
    for instance_id in test_instances:
        multisweep_file = multisweep_dir / f"02958343_{instance_id}.npz"
        output_file = output_dir / f"02958343_{instance_id}.npz"
        
        if multisweep_file.exists():
            try:
                # Load multi-sweep data
                data = np.load(multisweep_file)
                if 'point_clouds' in data:
                    sweeps = data['point_clouds']  # Shape: (6, 256, 3)
                    
                    # Extract first sweep (realistic random viewpoint)
                    single_sweep = sweeps[0:1]  # Shape: (1, 256, 3)
                    
                    # Save as single-sweep data
                    np.savez_compressed(output_file, point_clouds=single_sweep)
                    successful += 1
                    print(f"✅ {instance_id}")
                else:
                    print(f"❌ {instance_id} - no point_clouds in file")
                    missing += 1
            except Exception as e:
                print(f"❌ {instance_id} - error: {e}")
                missing += 1
        else:
            print(f"❌ {instance_id} - file not found")
            missing += 1
    
    print(f"\n🎉 Extraction complete!")
    print(f"   Successful: {successful}")
    print(f"   Missing: {missing}")
    print(f"   Output directory: {output_dir}")
    
    if successful > 0:
        print(f"\n✅ Ready for fair DeepSDF vs MV-DeepSDF comparison on {successful} instances!")

if __name__ == "__main__":
    main()