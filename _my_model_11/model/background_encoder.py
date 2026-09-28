import os
import torch
import torch.nn as nn
from torchvision.models import resnet18

class BackgroundEncoder(nn.Module):
    """
    Background Encoder trích xuất domain context từ ảnh background.
    - Backbone: ResNet18 (frozen), tải checkpoint resnet18_background_best.pth
    - Head: MLP (Linear 512 -> 768)
    """
    def __init__(self, embed_dim=768, checkpoint_path='resnet18_background_best.pth'):
        super().__init__()
        
        # ResNet18 backbone
        self.backbone = resnet18(num_classes=2) # Giả sử pretrained binary classification
        self.backbone.fc = nn.Identity() # Bỏ classification head, giữ lại (B, 512)
        
        # Load weights
        if os.path.exists(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
                
            # Xóa các key của fc layer nếu có
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith('fc.')}
            
            self.backbone.load_state_dict(state_dict, strict=False)
            print(f"[BackgroundEncoder] Loaded weights từ {checkpoint_path}")
        else:
            print(f"[BackgroundEncoder] Cảnh báo: Không tìm thấy {checkpoint_path}")
            
        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # MLP Projection
        self.mlp = nn.Sequential(
            nn.Linear(512, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )
        
    def forward(self, bg_images):
        """
        bg_images: (B, 3, H, W)
        Returns: (B, embed_dim)
        """
        with torch.no_grad():
            features = self.backbone(bg_images) # (B, 512)
            
        out = self.mlp(features) # (B, embed_dim)
        return out
