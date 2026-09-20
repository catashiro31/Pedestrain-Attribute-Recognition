# vision_encoder.py
# VMamba Vision Encoder wrapper cho CLIMP-PAR
# Dựa trên kiến trúc CLIMP (arXiv:2601.06891):
#   - VMamba làm vision backbone (thay thế ViT)
#   - Không dùng positional encoding (VMamba tự mã hóa vị trí qua SS2D scanning)
#   - Hỗ trợ đa phân giải tự nhiên (variable resolution)

import sys
import os
import torch
import torch.nn as nn

# Thêm đường dẫn VMamba vào sys.path để import VSSM
_VMAMBA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'VMamba')
if _VMAMBA_DIR not in sys.path:
    sys.path.insert(0, _VMAMBA_DIR)

from vmamba import VSSM


# ============================================================================
# Cấu hình VMamba theo từng variant (Tiny / Small / Base)
# Tham khảo: vmamba.py factory functions
# ============================================================================
VMAMBA_CONFIGS = {
    'tiny': dict(
        depths=[2, 2, 9, 2],
        dims=96,
        drop_path_rate=0.2,
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        ssm_d_state=16,
        ssm_ratio=2.0,
        ssm_dt_rank="auto",
        ssm_act_layer="silu",
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0.0,
        ssm_init="v0",
        forward_type="v0",
        mlp_ratio=0.0,
        mlp_act_layer="gelu",
        mlp_drop_rate=0.0,
        gmlp=False,
        patch_norm=True,
        norm_layer="ln",
        downsample_version="v1",
        patchembed_version="v1",
        use_checkpoint=False,
        posembed=False,
        imgsize=224,
    ),
    'small': dict(
        depths=[2, 2, 27, 2],
        dims=96,
        drop_path_rate=0.3,
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        ssm_d_state=16,
        ssm_ratio=2.0,
        ssm_dt_rank="auto",
        ssm_act_layer="silu",
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0.0,
        ssm_init="v0",
        forward_type="v0",
        mlp_ratio=0.0,
        mlp_act_layer="gelu",
        mlp_drop_rate=0.0,
        gmlp=False,
        patch_norm=True,
        norm_layer="ln",
        downsample_version="v1",
        patchembed_version="v1",
        use_checkpoint=False,
        posembed=False,
        imgsize=224,
    ),
    'base': dict(
        depths=[2, 2, 27, 2],
        dims=128,
        drop_path_rate=0.6,
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        ssm_d_state=16,
        ssm_ratio=2.0,
        ssm_dt_rank="auto",
        ssm_act_layer="silu",
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0.0,
        ssm_init="v0",
        forward_type="v0",
        mlp_ratio=0.0,
        mlp_act_layer="gelu",
        mlp_drop_rate=0.0,
        gmlp=False,
        patch_norm=True,
        norm_layer="ln",
        downsample_version="v1",
        patchembed_version="v1",
        use_checkpoint=False,
        posembed=False,
        imgsize=224,
    ),
}

# Pretrained checkpoint URLs cho từng variant (tự động tải từ GitHub)
VMAMBA_PRETRAINED_URLS = {
    'tiny': "https://github.com/MzeroMiko/VMamba/releases/download/%23v0cls/vssmtiny_dp01_ckpt_epoch_292.pth",
    'small': "https://github.com/MzeroMiko/VMamba/releases/download/%23v0cls/vssmsmall_dp03_ckpt_epoch_238.pth",
    'base': "https://github.com/MzeroMiko/VMamba/releases/download/%23v0cls/vssmbase_dp06_ckpt_epoch_241.pth",
}

# Kích thước feature cuối cùng (num_features) của từng variant
VMAMBA_FEATURE_DIMS = {
    'tiny': 768,    # dims=96, 4 stages → 96*8=768
    'small': 768,   # dims=96, 4 stages → 96*8=768
    'base': 1024,   # dims=128, 4 stages → 128*8=1024
}


class VMambaVisionEncoder(nn.Module):
    """
    Vision Encoder dựa trên VMamba cho CLIMP-PAR.
    
    Theo bài báo CLIMP (Section 3.2 - Vision Encoder):
    - Ảnh đầu vào I ∈ R^(H×W×3) được chia thành non-overlapping patches P×P
    - Đưa qua VSS blocks với SS2D cross-scan mechanism (4 hướng quét)
    - Hierarchical structure giảm dần kích thước feature map qua patch merging
    - Feature cuối cùng được chiếu sang shared embedding space qua projection W_v
    
    Ưu điểm chính:
    - Không cần positional encoding → hỗ trợ đa phân giải tự nhiên
    - Spatial inductive bias qua scanning patterns → robustness tốt hơn ViT
    - Sub-quadratic complexity → tiết kiệm bộ nhớ ở phân giải cao
    
    Args:
        variant (str): Phiên bản VMamba ('tiny', 'small', 'base'). Mặc định: 'tiny'
        embed_dim (int): Chiều của shared embedding space. Mặc định: 768
        pretrained (bool): Có tải pretrained ImageNet-1K weights không. Mặc định: True
        pretrained_path (str, optional): Đường dẫn checkpoint local. 
            Nếu None, tự động tải từ GitHub.
    """
    
    def __init__(self, variant='tiny', embed_dim=768, pretrained=True, pretrained_path=None):
        super().__init__()
        
        assert variant in VMAMBA_CONFIGS, f"Variant '{variant}' không hợp lệ. Chọn: {list(VMAMBA_CONFIGS.keys())}"
        
        self.variant = variant
        self.embed_dim = embed_dim
        self.feature_dim = VMAMBA_FEATURE_DIMS[variant]
        
        # 1. Khởi tạo VMamba backbone
        cfg = VMAMBA_CONFIGS[variant].copy()
        self.backbone = VSSM(**cfg)
        
        # 2. Loại bỏ classification head ban đầu (classifier.head)
        # Giữ lại: norm, permute, avgpool
        # Bỏ: flatten, head (Linear → 1000 classes)
        
        # 3. Tạo projection head: feature_dim → embed_dim
        # Theo CLIMP: "final features mapped to the shared embedding space via a learned projection W_v"
        self.norm = nn.LayerNorm(self.feature_dim)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Linear(self.feature_dim, embed_dim)
        
        # 4. Tải pretrained weights nếu yêu cầu
        if pretrained:
            self._load_pretrained(pretrained_path)
    
    def _load_pretrained(self, pretrained_path=None):
        """
        Tải pretrained weights cho VMamba backbone.
        Nếu không có file local, tự động tải từ GitHub releases.
        """
        if pretrained_path is not None and os.path.exists(pretrained_path):
            ckpt_path = pretrained_path
        else:
            # Tự động tải từ GitHub
            url = VMAMBA_PRETRAINED_URLS[self.variant]
            cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'checkpoints')
            os.makedirs(cache_dir, exist_ok=True)
            
            filename = f"vmamba_{self.variant}_pretrained.pth"
            ckpt_path = os.path.join(cache_dir, filename)
            
            if not os.path.exists(ckpt_path):
                print(f"[VMamba] Đang tải pretrained weights cho {self.variant} từ GitHub...")
                try:
                    state_dict = torch.hub.load_state_dict_from_url(
                        url, 
                        model_dir=cache_dir,
                        map_location='cpu',
                        check_hash=False,
                        file_name=filename
                    )
                    print(f"[VMamba] Tải thành công! Lưu tại: {ckpt_path}")
                    # torch.hub.load_state_dict_from_url đã tải và lưu xong
                    self._apply_pretrained(state_dict)
                    return
                except Exception as e:
                    print(f"[VMamba] Cảnh báo: Không thể tải pretrained weights: {e}")
                    print(f"[VMamba] Tiếp tục với random initialization.")
                    return

        # Tải từ file local
        try:
            ckpt = torch.load(ckpt_path, map_location='cpu')
            self._apply_pretrained(ckpt)
        except Exception as e:
            print(f"[VMamba] Lỗi khi tải checkpoint {ckpt_path}: {e}")
            print(f"[VMamba] Tiếp tục với random initialization.")
    
    def _apply_pretrained(self, ckpt):
        """Áp dụng pretrained state_dict vào backbone."""
        if isinstance(ckpt, dict):
            # Checkpoint có thể chứa key 'model' hoặc là state_dict trực tiếp
            state_dict = ckpt.get('model', ckpt)
        else:
            state_dict = ckpt
            
        # Load với strict=False vì chúng ta đã bỏ classification head
        incompatible = self.backbone.load_state_dict(state_dict, strict=False)
        
        missing = [k for k in incompatible.missing_keys if 'classifier.head' not in k]
        if missing:
            print(f"[VMamba] Missing keys (ngoài classifier head): {missing[:5]}...")
        if incompatible.unexpected_keys:
            print(f"[VMamba] Unexpected keys: {incompatible.unexpected_keys[:5]}...")
        
        print(f"[VMamba] Đã tải pretrained weights cho {self.variant} thành công!")
    
    def forward(self, x):
        """
        Forward pass: Ảnh → Image Embedding
        
        Args:
            x (Tensor): Ảnh đầu vào, shape (B, 3, H, W)
            
        Returns:
            image_features (Tensor): Image embedding, shape (B, embed_dim)
        """
        # 1. Patch embedding + VSS blocks (hierarchical)
        x = self.backbone.patch_embed(x)
        
        # Thêm positional embedding nếu có (VMamba mặc định không dùng)
        if self.backbone.pos_embed is not None:
            channel_first = self.backbone.channel_first
            pos_embed = self.backbone.pos_embed.permute(0, 2, 3, 1) if not channel_first else self.backbone.pos_embed
            x = x + pos_embed
        
        # 2. Đi qua tất cả các VSS block stages
        for layer in self.backbone.layers:
            x = layer(x)
        
        # 3. x hiện tại có shape (B, H', W', C) hoặc (B, C, H', W') tùy channel_first
        # Chuẩn hóa về (B, C, H', W') cho AdaptiveAvgPool2d
        if not self.backbone.channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()  # (B, H', W', C) → (B, C, H', W')
        
        # 4. Flatten không gian: (B, C, H', W') -> (B, C, L)
        B, C, H_prime, W_prime = x.shape
        x = x.view(B, C, -1)
        
        # 5. Chuyển sang (B, L, C)
        x = x.transpose(1, 2)
        
        # 6. Layer Norm và Projection
        x = self.norm(x)
        image_features = self.projection(x) # (B, L, 768)
        
        return image_features
    
    def get_feature_dim(self):
        """Trả về chiều feature trước projection."""
        return self.feature_dim
    
    def get_embed_dim(self):
        """Trả về chiều embedding sau projection."""
        return self.embed_dim


# ============================================================================
# Tiện ích: Tạo nhanh encoder theo variant
# ============================================================================
def create_vision_encoder(variant='tiny', embed_dim=768, pretrained=True, **kwargs):
    """
    Factory function tạo VMamba Vision Encoder.
    
    Args:
        variant: 'tiny', 'small', hoặc 'base'
        embed_dim: Chiều embedding đầu ra
        pretrained: Có tải pretrained weights không
        
    Returns:
        VMambaVisionEncoder instance
    """
    return VMambaVisionEncoder(
        variant=variant,
        embed_dim=embed_dim,
        pretrained=pretrained,
        **kwargs
    )
