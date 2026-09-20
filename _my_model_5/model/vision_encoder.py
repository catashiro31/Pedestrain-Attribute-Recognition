# vision_encoder.py
# Khối Vision Hybrid (ResNet18 + Attention + VMamba) cho CLIMP-PAR

import sys
import os
import torch
import torch.nn as nn
import torchvision.models as models

# Thêm đường dẫn VMamba vào sys.path để import VSSBlock
_VMAMBA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'VMamba')
if _VMAMBA_DIR not in sys.path:
    sys.path.insert(0, _VMAMBA_DIR)

from vmamba import VSSBlock

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Đảm bảo in_planes // ratio không nhỏ hơn 1
        mid_channels = max(1, in_planes // ratio)
        self.fc1   = nn.Conv2d(in_planes, mid_channels, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(mid_channels, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, channels, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(channels, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

class HybridVisionEncoder(nn.Module):
    def __init__(self, embed_dim=768, pretrained=True, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        
        # 1. ResNet18 Backbone (2 Stages đầu)
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        
        # Stem: Conv1 -> BN1 -> ReLU -> MaxPool
        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        
        # Stage 1: layer1 (Output channels: 64)
        self.layer1 = resnet.layer1
        
        # Stage 2: layer2 (Output channels: 128)
        self.layer2 = resnet.layer2
        
        # 2. Fusion Module
        # Nén layer1 (64) xuống không gian của layer2 (stride 2) để cùng shape 56x56
        self.downsample_l1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Layer kết hợp: Nhận 128 (từ downsampled L1) + 128 (từ L2) = 256
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # 3. Attention Module (CBAM)
        self.attention = CBAM(channels=256, ratio=16)
        
        # 4. Vision Mamba (4 VSS Blocks)
        # VMamba VSSBlock mặc định dùng shape (B, H, W, C) nếu channel_first=False
        self.vss_blocks = nn.ModuleList([
            VSSBlock(hidden_dim=256, channel_first=False, forward_type="v0") for _ in range(4)
        ])
        
        # 5. Output Projection
        # Adaptive pooling để ép về 8x4 patch
        self.pool = nn.AdaptiveAvgPool2d((8, 4))
        # Projection Linear
        self.proj = nn.Linear(256, embed_dim)

    def forward(self, x):
        """
        Input: x shape (B, 3, 448, 448)
        Output: (B, 32, 768)
        """
        B = x.shape[0]
        
        # 1. Đi qua ResNet18
        x_stem = self.stem(x)       # (B, 64, 112, 112)
        f1 = self.layer1(x_stem)    # (B, 64, 112, 112)
        f2 = self.layer2(f1)        # (B, 128, 56, 56)
        
        # 2. Fusion
        f1_down = self.downsample_l1(f1) # (B, 128, 56, 56)
        fused = torch.cat([f1_down, f2], dim=1) # (B, 256, 56, 56)
        fused = self.fusion_conv(fused)  # (B, 256, 56, 56)
        
        # 3. Attention
        attn_out = self.attention(fused) # (B, 256, 56, 56)
        
        # 4. VSS Blocks (VMamba)
        # VSSBlock yêu cầu (B, H, W, C)
        x_vss = attn_out.permute(0, 2, 3, 1).contiguous() # (B, 56, 56, 256)
        
        for block in self.vss_blocks:
            x_vss = block(x_vss)
            
        # Đưa về lại (B, C, H, W)
        x_vss = x_vss.permute(0, 3, 1, 2).contiguous() # (B, 256, 56, 56)
        
        # 5. Pooling & Projection
        x_pool = self.pool(x_vss) # (B, 256, 8, 4)
        
        # Flatten không gian: (B, 256, 8, 4) -> (B, 256, 32)
        x_flat = x_pool.view(B, 256, -1)
        
        # Transpose: -> (B, 32, 256)
        x_flat = x_flat.transpose(1, 2)
        
        # Projection: -> (B, 32, 768)
        out = self.proj(x_flat)
        
        return out
