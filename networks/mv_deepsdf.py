import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple

# Try to import fpsample first, then torch_cluster, fallback to CPU implementation
try:
    import fpsample
    FPSAMPLE_AVAILABLE = True
except ImportError:
    FPSAMPLE_AVAILABLE = False
    try:
        from torch_cluster import fps
        TORCH_CLUSTER_AVAILABLE = True
    except ImportError:
        TORCH_CLUSTER_AVAILABLE = False

class FarthestPointSampling:
    """Farthest Point Sampling for point cloud preprocessing."""
    def __init__(self, num_points: int = 256):
        self.num_points = num_points

    def __call__(self, points: torch.Tensor) -> torch.Tensor:
        if points.ndim == 3:  # Batched input (B, N, 3)
            # Use a list comprehension which is often clearer and more efficient
            return torch.stack([self(p) for p in points], dim=0)

        # The rest of the logic operates on a single point cloud (N, 3)
        if points.ndim != 2:
            raise ValueError(f"FarthestPointSampling expects a 2D or 3D tensor, but got {points.ndim}D.")

        N = points.shape[0]
        if N <= self.num_points:
            if N < self.num_points:
                # This logic is now safe because 'points' is guaranteed to be 2D
                repeat = (self.num_points + N - 1) // N
                points = points.repeat(repeat, 1)[:self.num_points]
            return points

        if FPSAMPLE_AVAILABLE:
            return self._fps_fpsample(points)
        elif TORCH_CLUSTER_AVAILABLE:
            return self._fps_torch_cluster(points)
        else:
            return self._fps_cpu(points)

    def _fps_fpsample(self, points: torch.Tensor) -> torch.Tensor:
        device = points.device
        pts = points.cpu().numpy()
        idx = fpsample.bucket_fps_kdline_sampling(pts, self.num_points, h=3)
        return torch.from_numpy(pts[idx]).to(device)

    def _fps_torch_cluster(self, points: torch.Tensor) -> torch.Tensor:
        device = points.device
        batch = torch.zeros(points.shape[0], dtype=torch.long, device=device)
        idx = fps(points, batch, ratio=self.num_points / points.shape[0])
        return points[idx]

    def _fps_cpu(self, points: torch.Tensor) -> torch.Tensor:
        device = points.device
        pts = points.cpu().numpy()
        N = pts.shape[0]
        inds = np.zeros(self.num_points, dtype=np.int64)
        inds[0] = np.random.randint(N)
        dist = np.sum((pts - pts[inds[0]])**2, axis=1)
        for i in range(1, self.num_points):
            idx = np.argmax(dist)
            inds[i] = idx
            d2 = np.sum((pts - pts[idx])**2, axis=1)
            dist = np.minimum(dist, d2)
        return torch.from_numpy(pts[inds]).to(device)

class GlobalFeatureExtractor(nn.Module):
    """EXACT Yellow Block per specification"""
    def __init__(self):
        super().__init__()
        # Step 2: Shared MLP 3→128
        self.fc1 = nn.Linear(3, 128)
        # Step 3: BatchNorm + ReLU
        self.bn1 = nn.BatchNorm1d(128)
        # Step 4: Shared MLP 128→256
        self.fc2 = nn.Linear(128, 256)
        # Step 8: Shared MLP 512→512
        self.fc3 = nn.Linear(512, 512)
        # Step 9: BatchNorm + ReLU
        self.bn3 = nn.BatchNorm1d(512)
        # Step 10: Shared MLP 512→1024
        self.fc4 = nn.Linear(512, 1024)
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Yellow block EXACT implementation:
        Input: [B, N, 3] → Output: [B, 1024]
        """
        B, N, _ = x.shape
        
        # Step 1: Input [B, N, 3]
        # Reshape for shared MLP: (B*N, 3)
        x = x.view(B * N, 3)
        
        # Step 2: Shared MLP → [B, N, 128]
        x = self.fc1(x)  # (B*N, 128)
        
        # Step 3: BatchNorm + ReLU
        x = self.bn1(x)
        x = self.relu(x)
        
        # Step 4: Shared MLP → [B, N, 256]
        x = self.fc2(x)  # (B*N, 256)
        
        # Reshape back: (B, N, 256)
        x = x.view(B, N, 256)
        
        # Step 5: Max-pool over N → [B, 1, 256]
        global_256 = x.max(dim=1, keepdim=True)[0]  # (B, 1, 256)
        
        # Step 6: Repeat N times → [B, N, 256]
        repeated_global = global_256.expand(-1, N, -1)  # (B, N, 256)
        
        # Step 7: Concatenate → [B, N, 512]
        concat_features = torch.cat([x, repeated_global], dim=2)  # (B, N, 512)
        
        # Reshape for shared MLP: (B*N, 512)
        concat_features = concat_features.view(B * N, 512)
        
        # Step 8: Shared MLP → [B, N, 512]
        x = self.fc3(concat_features)  # (B*N, 512)
        
        # Step 9: BatchNorm + ReLU
        x = self.bn3(x)
        x = self.relu(x)
        
        # Step 10: Shared MLP → [B, N, 1024]
        x = self.fc4(x)  # (B*N, 1024)
        
        # Reshape back: (B, N, 1024)
        x = x.view(B, N, 1024)
        
        # Step 11: Max-pool over N → [B, 1024]
        final_global = x.max(dim=1)[0]  # (B, 1024)
        
        return final_global

# REMOVED: ElementToSetFeatureExtractor - NOT in diagram!
# The diagram shows direct concatenation + pooling, no intermediate FC layers

class RedBlockAggregator(nn.Module):
    """EXACT Red Block per specification - pools across sweep axis (dim=0)"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Red Block Step 4: Average-pool across sweep axis (dim=0)
        # Input: [B, 1280] → Output: [1, 1280]
        return x.mean(dim=0, keepdim=True)  # Keep dimension for [1, 1280]

class LatentCodePredictor(nn.Module):
    """EXACT Red Block Step 5: FC layer 1280→256, NO BN/ReLU"""
    def __init__(self, input_dim: int = 1280, output_dim: int = 256):
        super().__init__()
        # Red Block Step 5: Fully-connected layer (Linear 1280→256, NO BN/ReLU)
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Red Block Step 5: [1, 1280] → [1, 256] (predicted latent)
        return self.fc(x)

class MVDeepSDF(nn.Module):
    """100% EXACT implementation per specification"""
    def __init__(self,
                 num_points: int = 256,
                 global_feature_dim: int = 1024,
                 latent_dim: int = 256,
                 element_feature_dim: int = 128,  # Kept for backward compatibility but not used
                 num_views: int = 6):
        super().__init__()
        self.num_views = num_views  # This is B in the specification
        self.global_feature_dim = global_feature_dim
        self.latent_dim = latent_dim
        
        # Components:
        self.fps = FarthestPointSampling(num_points)
        self.global_extractor = GlobalFeatureExtractor()  # Yellow Block
        self.aggregator = RedBlockAggregator()            # Red Block Step 4
        self.predictor = LatentCodePredictor(             # Red Block Step 5
            input_dim=global_feature_dim + latent_dim,    # 1024 + 256 = 1280
            output_dim=latent_dim                         # 256
        )

    def forward(self, point_clouds: torch.Tensor, latent_codes: torch.Tensor) -> torch.Tensor:
        """
        EXACT implementation per specification
        
        For SINGLE EXAMPLE processing (Red Block specification):
        - point_clouds: [B, N, 3] where B=6 sweeps
        - latent_codes: [B, 256] where B=6 sweeps
        - Returns: [1, 256] predicted latent
        """
        # Handle batching: process each example separately per Red Block spec
        if point_clouds.dim() == 4:  # (batch, B, N, 3)
            batch_size = point_clouds.shape[0]
            results = []
            for i in range(batch_size):
                single_result = self._forward_single_example(
                    point_clouds[i],    # [B, N, 3]
                    latent_codes[i]     # [B, 256]
                )
                results.append(single_result)
            return torch.cat(results, dim=0)  # [batch, 256]
        else:
            # Single example: [B, N, 3] and [B, 256]
            return self._forward_single_example(point_clouds, latent_codes)
    
    def _forward_single_example(self, point_clouds: torch.Tensor, latent_codes: torch.Tensor) -> torch.Tensor:
        """
        Process single example per Red Block specification
        
        Args:
            point_clouds: [B, N, 3] where B=6 sweeps
            latent_codes: [B, 256] where B=6 sweeps
            
        Returns:
            predicted_latent: [1, 256]
        """
        B, N, _ = point_clouds.shape
        assert B == self.num_views, f"Expected {self.num_views} sweeps, got {B}"
        
        # Apply FPS to each sweep
        sampled_sweeps = []
        for sweep_idx in range(B):
            pts = point_clouds[sweep_idx:sweep_idx+1]  # [1, N, 3] - keep batch dim for FPS
            sampled = self.fps(pts)                    # [1, num_points, 3]
            sampled_sweeps.append(sampled.squeeze(0))  # [num_points, 3]
        
        stacked_sweeps = torch.stack(sampled_sweeps, dim=0)  # [B, num_points, 3]
        
        # Yellow Block: Extract global features per sweep
        # Process each sweep through GlobalFeatureExtractor
        global_features = []
        for sweep_idx in range(B):
            sweep_pc = stacked_sweeps[sweep_idx:sweep_idx+1]  # [1, num_points, 3]
            global_feat = self.global_extractor(sweep_pc)     # [1, 1024]
            global_features.append(global_feat.squeeze(0))   # [1024]
        
        global_features = torch.stack(global_features, dim=0)  # [B, 1024]
        
        # Red Block implementation per specification:
        
        # Step 1: Per-sweep DeepSDF latents [B, 256] ✓ (already provided)
        # Step 2: Per-sweep global features [B, 1024] ✓ (extracted above)
        
        # Step 3: Concatenate along feature axis → [B, 1280]
        concat_features = torch.cat([global_features, latent_codes], dim=1)  # [B, 1280]
        
        # Step 4: Average-pool across sweep axis (dim=0) → [1, 1280]
        pooled_features = self.aggregator(concat_features)  # [1, 1280]
        
        # Step 5: FC layer 1280→256, NO BN/ReLU → [1, 256]
        predicted_latent = self.predictor(pooled_features)  # [1, 256]
        
        return predicted_latent

if __name__ == "__main__":
    # smoke‐test
    B, V, N = 2, 6, 1024
    pcs = torch.randn(B, V, N, 3)
    lat = torch.randn(B, V, 256)
    m = MVDeepSDF()
    out = m(pcs, lat)
    print("Output shape:", out.shape)  # should be (2,256)
