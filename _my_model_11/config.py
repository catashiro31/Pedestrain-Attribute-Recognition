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
    
    Tối ưu cho máy cá nhân / server 2× RTX 3090:
    - VRAM: 24GB mỗi GPU (tổng 48GB)
    - Kiến trúc Ampere: Hỗ trợ mạnh mẽ TF32, FP16, BF16
    - Dùng DistributedDataParallel (DDP)
    """
    
    # ========================
    # Dataset
    # ========================
    dataset_name: str = 'MSP60k'
    pkl_path: str = ''   # Sẽ được thiết lập trong __post_init__ hoặc từ bên ngoài
    img_dir: str = ''    # Sẽ được thiết lập trong __post_init__ hoặc từ bên ngoài
    bg_img_dir: str = '' # Thư mục chứa ảnh background
    bg_checkpoint: str = 'resnet18_background_best.pth' # Checkpoint cho BackgroundEncoder
    
    # ========================
    # Vision Encoder (VMamba)
    # ========================
    vmamba_variant: str = 'tiny'          # 'tiny', 'small', 'base'
    vmamba_pretrained: bool = True        # Tải ImageNet-1K pretrained weights
    vmamba_pretrained_path: Optional[str] = None  # Path local (None = tự tải từ GitHub)
    
    # ========================
    # Text Encoder (Mamba LLM)
    # ========================
    mamba_model: str = 'mamba-130m'       # Model pretrained trên HuggingFace
    mamba_freeze: bool = True             # Đóng băng backbone Mamba-130M (theo architecture doc §2.6)
    max_text_length: int = 0             # 0 = không giới hạn (pad theo batch dài nhất)
    
    # ========================
    # Shared Embedding Space
    # ========================
    embed_dim: int = 768                  # Chiều embedding chung (projection dimension)
    
    # ========================
    # Image Processing
    # ========================
    img_height: int = 256               # Phân giải ảnh đầu vào (cho collate/padding batch)
    img_width: int = 128                  # Khung chứa ảnh cố định, giữ nguyên pixel
    
    # ========================
    # Training — tối ưu cho 2× RTX 3060 (12GB mỗi GPU)
    # ========================
    batch_size: int = 16                  # Batch size MỖI GPU 
    num_workers: int = 8                  # Số lượng process nạp data
    cache_in_memory: bool = True         # Tải toàn bộ dataset vào RAM nếu dư thừa bộ nhớ
    epochs: int = 30
    lr: float = 5e-5                      # Peak learning rate (AdamW)
    weight_decay: float = 0.05
    warmup_epochs: int = 2
    grad_accum_steps: int = 2             # Effective batch = 16 (batch) × 2 (GPU) × 2 (accum) = 64
    
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
    loss_type: str = 'hybrid_asl_bce'              # 'hybrid_asl_bce', 'hybrid_loss', 'asl', 'bce'
    asl_gamma_neg: float = 4.0
    asl_gamma_pos: float = 1.0
    asl_clip: float = 0.05
    
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
        if not self.bg_img_dir:
            self.bg_img_dir = os.path.join(base_dir, '..', 'MSP60k', 'SUBMIT', 'domain_noper_images_inpainted_refined')
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
    Cấu hình cho máy 2× RTX 3090 GPU (24GB VRAM).
    (Hàm giữ nguyên tên get_kaggle_config để tương thích với các file cũ)
    - Hỗ trợ TF32 / FP16 cực tốt (Ampere architecture).
    """
    defaults = dict(
        dataset_name='MSP60k',
        vmamba_variant='tiny',
        mamba_model='mamba-130m',
        embed_dim=768,
        img_height=256,
        img_width=128,
        batch_size=16,          # 16 sample / 12GB VRAM
        num_workers=4,          # 4 workers cho 2x RTX 3060
        cache_in_memory=True,  # Tắt để tránh nhân bản cache khi multiprocessing fork
        grad_accum_steps=1,     # Effective batch = 16 × 2(GPU) × 2(accum) = 64
        use_amp=True,
        gradient_checkpointing=True,
        use_ddp=True,
        lr=5e-5,
        mamba_freeze=True,      # Freeze Mamba-130M backbone theo architecture doc
    )
    defaults.update(kwargs)
    return CLIMPConfig(**defaults)


def get_msp60k_config(**kwargs):
    """Alias cho config."""
    return get_kaggle_config(**kwargs)
