# config.py
# Cấu hình hyperparameters cho CLIMP-PAR v8
# Pipeline mới: 3-branch (Text + Vision + Cross-Modal Mamba)
# Tối ưu cho Kaggle 2× T4 GPU (16GB VRAM mỗi GPU)

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CLIMPConfig:
    """
    Cấu hình toàn bộ mô hình CLIMP-PAR v8.
    
    Pipeline v8.1 (Spatial Cross-Attention):
    - Text Branch: style / attribute / full prompts qua Mamba-130M (đóng băng backbone)
    - Vision Branch: ảnh → VMamba-Small (đóng băng backbone) → vision features
    - Spatial Cross-Attention: vision tokens attend vào text attribute features
    - Cross Branch: CrossModalMambaBlock fuses 2 nhánh
    
    Tối ưu cho Kaggle 2× T4:
    - T4 VRAM: 16GB mỗi GPU
    - T4 KHÔNG hỗ trợ bf16, chỉ hỗ trợ fp16
    """
    
    # ========================
    # Dataset
    # ========================
    dataset_name: str = 'MSP60k'
    pkl_path: str = ''
    img_dir: str = ''
    
    # ========================
    # Vision Encoder (VMamba-Small, đóng băng)
    # ========================
    vmamba_variant: str = 'small'
    vmamba_pretrained: bool = True
    vmamba_pretrained_path: Optional[str] = None
    
    # ========================
    # Text Encoder (Mamba-130M, đóng băng)
    # ========================
    mamba_model: str = 'mamba-130m'
    mamba_freeze: bool = True  # Luôn đóng băng text encoder
    max_text_length: int = 0
    
    # ========================
    # Shared Embedding Space
    # ========================
    embed_dim: int = 768
    
    # ========================
    # Domain Tokens
    # ========================
    domain_tokens_path: str = ''  # Đường dẫn tới domain_tokens.pt
    
    # ========================
    # Image Processing
    # ========================
    img_height: int = 448
    img_width: int = 448
    
    # ========================
    # Training
    # ========================
    batch_size: int = 32     # Batch size MỖI GPU (nhỏ hơn vì model phức tạp hơn)
    num_workers: int = 2
    epochs: int = 30
    lr: float = 5e-5
    weight_decay: float = 0.05
    warmup_epochs: int = 2
    grad_accum_steps: int = 1     # Gradient accumulation → effective batch lớn hơn (1 = không dùng accumulation)
    
    # ========================
    # Mixed Precision & Memory
    # ========================
    use_amp: bool = True
    gradient_checkpointing: bool = True
    
    # ========================
    # Multi-GPU
    # ========================
    use_ddp: bool = True
    
    # ========================
    # Loss
    # ========================
    loss_type: str = 'weighted_bce_asl'  # 'bce', 'weighted_bce', 'asl', hoặc 'weighted_bce_asl'
    global_contrastive_weight: float = 0.1
    finegrained_contrastive_weight: float = 0.1
    temperature_init: float = 0.07
    
    # ========================
    # Misc
    # ========================
    seed: int = 42
    device: str = 'cuda'
    output_dir: str = ''
    log_interval: int = 20
    save_interval: int = 5
    
    def __post_init__(self):
        """Tự động thiết lập đường dẫn mặc định nếu chưa có."""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        if not self.pkl_path:
            self.pkl_path = os.path.join(base_dir, '..', 'MSP60k', 'SUBMIT', 'dataset_random.pkl')
        if not self.img_dir:
            self.img_dir = os.path.join(base_dir, '..', 'MSP60k', 'SUBMIT', 'images')
        if not self.domain_tokens_path:
            self.domain_tokens_path = '/media/catashiro31/DATA/Nghiên cứu khoa học/Pedestrain Attribute Recognition/_my_model_8/domain_tokens/domain_tokens.pt'
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
    Batch size nhỏ hơn do pipeline mới có Spatial Cross-Attention.
    """
    defaults = dict(
        dataset_name='MSP60k',
        vmamba_variant='small',
        mamba_model='mamba-130m',
        mamba_freeze=True,
        embed_dim=768,
        img_height=448,
        img_width=448,
        batch_size=4,
        num_workers=2,
        grad_accum_steps=4,     # Effective batch = 4×2×4 = 32
        use_amp=True,
        gradient_checkpointing=True,
        use_ddp=True,
        lr=5e-5,
        loss_type='weighted_bce_asl',
    )
    defaults.update(kwargs)
    return CLIMPConfig(**defaults)
