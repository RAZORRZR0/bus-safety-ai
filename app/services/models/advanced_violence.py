"""
Implementation of advanced violence detection architectures as proposed in:
- mathematics-13-02665.pdf: Int.2D-3D-CNN
- sensors-24-08102.pdf: Temporal Pyramid Network (TPN)
"""

import torch
import torch.nn as nn
from torchvision import models

class TemporalPyramidNetwork(nn.Module):
    """
    Simplified Temporal Pyramid Network (TPN) for action recognition.
    Fuses multi-scale temporal features extracted from a 2D CNN backbone.
    """
    def __init__(self, num_classes=1, backbone_name='resnet18'):
        super().__init__()
        
        # Load a 2D CNN backbone (e.g., ResNet18)
        if backbone_name == 'resnet18':
            backbone = models.resnet18(weights=None)
            self.feature_dim = 512
        elif backbone_name == 'mobilenet_v2':
            backbone = models.mobilenet_v2(weights=None)
            self.feature_dim = 1280
            backbone.fc = nn.Identity() # MobileNetV2 uses classifier, handled differently, but just for consistency
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
            
        if hasattr(backbone, 'fc'):
            backbone.fc = nn.Identity()
        elif hasattr(backbone, 'classifier'):
            backbone.classifier = nn.Identity()
            
        self.backbone = backbone
        
        # Temporal Pyramid pooling (e.g., bin sizes 1, 2, 4)
        self.temporal_scales = [1, 2, 4]
        
        # Final classification head
        total_features = self.feature_dim * sum(self.temporal_scales)
        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(total_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        """
        x shape: (Batch, Time, Channels, Height, Width)
        """
        B, T, C, H, W = x.shape
        # Fold time into batch for 2D backbone
        x = x.view(B * T, C, H, W)
        
        # Extract features: (B*T, Feature_Dim)
        features = self.backbone(x)
        if len(features.shape) > 2:
            features = nn.functional.adaptive_avg_pool2d(features, (1, 1))
            features = features.view(B * T, -1)
            
        # Reshape to (Batch, Feature_Dim, Time) for temporal pooling
        features = features.view(B, T, -1).transpose(1, 2)
        
        # Temporal Pyramid Pooling
        pooled_features = []
        for scale in self.temporal_scales:
            # Adaptive pooling over time dimension
            pool = nn.functional.adaptive_avg_pool1d(features, scale)
            # Flatten pooled features for this scale
            pooled_features.append(pool.reshape(B, -1))
            
        # Concatenate all scales
        concat_features = torch.cat(pooled_features, dim=1)
        
        # Classification
        logits = self.head(concat_features)
        
        if logits.shape[-1] == 1:
            return logits.squeeze(-1)
        return logits


class Int2D3DCNN(nn.Module):
    """
    Int.2D-3D-CNN architecture for violence recognition.
    Extracts spatial features via standard 2D CNN, combined with 3D CNN
    to capture spatiotemporal dynamics.
    """
    def __init__(self, num_classes=1):
        super().__init__()
        
        # 2D Spatial Branch
        spatial_backbone = models.mobilenet_v3_small(weights=None)
        self.spatial_features = spatial_backbone.features
        self.spatial_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 3D Temporal Branch
        # A simple 3D CNN (C3D style)
        self.temporal_branch = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        
        # The MobileNetV3 small outputs 576 channels
        spatial_dim = 576 
        temporal_dim = 64
        
        self.head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(spatial_dim + temporal_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        """
        x shape: (Batch, Time, Channels, Height, Width)
        """
        B, T, C, H, W = x.shape
        
        # --- 2D Branch ---
        # Take the middle frame or average across frames
        x_2d = x.view(B * T, C, H, W)
        feat_2d = self.spatial_features(x_2d)
        feat_2d = self.spatial_pool(feat_2d).view(B, T, -1)
        # Pool across time (e.g., mean)
        feat_2d = feat_2d.mean(dim=1)
        
        # --- 3D Branch ---
        # Permute to (Batch, Channels, Time, Height, Width) for 3D CNN
        x_3d = x.transpose(1, 2)
        feat_3d = self.temporal_branch(x_3d).view(B, -1)
        
        # Fusion
        fused = torch.cat((feat_2d, feat_3d), dim=1)
        
        logits = self.head(fused)
        if logits.shape[-1] == 1:
            return logits.squeeze(-1)
        return logits

