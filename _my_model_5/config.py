# config.py
# Cấu hình hyperparameters cho CLIMP-PAR
# Dựa trên bài báo CLIMP (arXiv:2601.06891) + tùy chỉnh cho PAR
# Tối ưu cho Kaggle 2× T4 GPU (16GB VRAM mỗi GPU)

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CLIMPConfig:
    """
    Cấu hình toàn bộ mô hình CLIMP-PAR.
    
    Tham khảo CLIMP paper Section 4 (Experimental Setup):
    - Training: AdamW, cosine LR schedule, peak LR 5e-5
    - Projection dim: 768
    
    Tối ưu cho Kaggle 2× T4:
    - T4 VRAM: 16GB mỗi GPU
    - T4 KHÔNG hỗ trợ bf16, chỉ hỗ trợ fp16
    - Dùng DataParallel (DP) hoặc DistributedDataParallel (DDP)
    """
    
    # ========================
    # Dataset
    # ========================
    dataset_name: str = 'MSP60k'
    pkl_path: str = ''   # Sẽ được thiết lập trong __post_init__ hoặc từ bên ngoài
    img_dir: str = ''    # Sẽ được thiết lập trong __post_init__ hoặc từ bên ngoài
    
    # ========================
    # Vision Encoder (Hybrid ResNet18 + Attention + VMamba)
    # ========================
    resnet_pretrained: bool = True        # Tải ImageNet-1K pretrained weights cho ResNet18
    
    # ========================
    # Text Encoder (Mamba LLM)
    # ========================
    mamba_model: str = 'mamba-130m'       # Model pretrained trên HuggingFace
    mamba_freeze: bool = False            # Đóng băng backbone khi huấn luyện
    max_text_length: int = 0             # 0 = không giới hạn (pad theo batch dài nhất)
    
    # ========================
    # Shared Embedding Space
    # ========================
    embed_dim: int = 768                  # Chiều embedding chung (projection dimension)
    
    # ========================
    # Image Processing
    # ========================
    img_height: int = 448                 # Phân giải ảnh đầu vào (cho collate/padding batch)
    img_width: int = 448                  # Khung chứa ảnh cố định, giữ nguyên pixel
    
    # ========================
    # Training — tối ưu cho 2× T4 (16GB mỗi GPU)
    # ========================
    batch_size: int = 8                  # Batch size MỖI GPU (effective batch = 16×2 = 32)
    num_workers: int = 2                  # Kaggle thường có 4 CPU cores
    epochs: int = 30
    lr: float = 5e-5                      # Peak learning rate (AdamW)
    weight_decay: float = 0.05
    warmup_epochs: int = 2
    grad_accum_steps: int = 2             # Gradient accumulation → effective batch = 16×2×2 = 64
    
    # ========================
    # Mixed Precision & Memory
    # ========================
    use_amp: bool = True                  # Automatic Mixed Precision (fp16 cho T4)
    gradient_checkpointing: bool = True   # Tiết kiệm VRAM bằng recomputation
    
    # ========================
    # Multi-GPU
    # ========================
    use_ddp: bool = True                  # DistributedDataParallel cho 2× T4
    
    # ========================
    # Contrastive
    # ========================
    temperature: float = 0.07             # Temperature ban đầu (learnable)
    
    # ========================
    # Loss
    # ========================
    loss_type: str = 'bce'                # 'bce' (Binary Cross Entropy) hoặc 'asl' (Asymmetric Loss)
    
    # ========================
    # Misc
    # ========================
    seed: int = 42
    device: str = 'cuda'
    output_dir: str = ''                  # Sẽ được thiết lập
    log_interval: int = 50                # In log mỗi N iterations
    save_interval: int = 5                # Lưu checkpoint mỗi N epochs
    
    def __post_init__(self):
        """Tự động thiết lập đường dẫn mặc định nếu chưa có."""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        if not self.pkl_path:
            self.pkl_path = os.path.join(base_dir, '..', 'MSP60k', 'SUBMIT', 'dataset_random.pkl')
        if not self.img_dir:
            self.img_dir = os.path.join(base_dir, '..', 'MSP60k', 'SUBMIT', 'images')
        if not self.output_dir:
            self.output_dir = os.path.join(base_dir, 'output')
            
        os.makedirs(self.output_dir, exist_ok=True)


# ============================================================================
# Cấu hình mặc định
# ============================================================================
def get_default_config(**kwargs):
    """Tạo cấu hình mặc định, có thể ghi đè bằng kwargs."""
    return CLIMPConfig(**kwargs)


def get_kaggle_config(**kwargs):
    """
    Cấu hình cho Kaggle 2× T4 GPU.
    T4: 16GB VRAM, hỗ trợ fp16 (KHÔNG hỗ trợ bf16).
    """
    defaults = dict(
        dataset_name='PETA',
        img_dir='/kaggle/input/peta-dataset/PETA/images',
        resnet_pretrained=True,
        mamba_model='mamba-130m',
        embed_dim=768,
        img_height=448,
        img_width=448,
        batch_size=8,          # Mỗi GPU (do ảnh 448x448 khá nặng)
        num_workers=2,
        grad_accum_steps=4,     # Effective batch = 8×2×4 = 64
        use_amp=True,
        gradient_checkpointing=True,
        use_ddp=True,
        lr=5e-5,
    )
    defaults.update(kwargs)
    return CLIMPConfig(**defaults)


def get_msp60k_config(**kwargs):
    """Alias cho get_kaggle_config."""
    return get_kaggle_config(**kwargs)
