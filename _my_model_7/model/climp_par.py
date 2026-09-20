# climp_par.py
# Mô hình CLIMP-PAR v7: Bổ sung AttributeGCN sau Text Encoder
# Kế thừa từ v4, thêm GCN để khai thác mối quan hệ giữa các thuộc tính

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vision_encoder import VMambaVisionEncoder
from .text_encoder import MambaTextEncoder
from .cross_mamba import CrossModalMambaBlock
from .attribute_gcn import AttributeGCN

class CLIMPPAR(nn.Module):
    """
    CLIMP-PAR v7: Thêm AttributeGCN khai thác tương quan thuộc tính.
    
    Luồng dữ liệu:
    - Vision: (B, 32, 768)
    - Text: (B, 57, 768) → GCN → (B, 57, 768) → nén thành (B, 32, 768)
    - Cross-Modal Mamba: Fusion 2 nhánh
    - Output: Concat → GAP → MLP → 57 logits
    
    So với v4: Thêm AttributeGCN giữa text_encoder và text_compress.
    GCN hoạt động trên đồ thị 57 nodes (mỗi node = 1 thuộc tính),
    cho phép mỗi attribute embedding trao đổi thông tin với các 
    thuộc tính liên quan trước khi fusion với vision.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 1. Vision Encoder (VMamba-Tiny)
        self.vision_encoder = VMambaVisionEncoder(
            embed_dim=config.embed_dim,
            pretrained_path=config.vmamba_pretrained_path
        )
        
        # 2. Text Encoder (Mamba LLM)
        self.text_encoder = MambaTextEncoder(
            model_name=config.mamba_model,
            embed_dim=config.embed_dim,
            freeze_backbone=config.mamba_freeze
        )
        
        # 3. Attribute GCN (MỚI — khai thác quan hệ giữa 57 thuộc tính)
        self.attr_gcn = AttributeGCN(
            num_attrs=57,
            d_model=config.embed_dim,
            num_layers=config.gcn_num_layers,
        )
        
        # 4. Text Compression (Nén 57 token xuống 32 token để khớp với ảnh)
        self.text_compress = nn.Linear(57, 32)
        
        # 5. Cross-Modal Mamba Block
        self.cross_mamba = CrossModalMambaBlock(d_model=config.embed_dim)
        
        # 6. MLP Fusion Head (Đầu ra 57 logits)
        self.mlp_head = nn.Sequential(
            nn.Linear(config.embed_dim * 2, config.embed_dim),
            nn.LayerNorm(config.embed_dim),
            nn.GELU(),
            nn.Linear(config.embed_dim, 57)
        )

    def forward(self, images, text_input_ids=None, text_attention_mask=None, texts=None, cached_text_features=None):
        """
        Forward pass tính toán logits.
        """
        # 1. Trích xuất đặc trưng hình ảnh
        # image_features: (B, 32, 768)
        image_features = self.vision_encoder(images)
        B = image_features.shape[0]
        
        # 2. Trích xuất đặc trưng văn bản
        if cached_text_features is not None:
            text_features = cached_text_features # (57, 768)
        elif text_input_ids is not None:
            text_features = self.text_encoder(text_input_ids, text_attention_mask)
        elif texts is not None:
            text_features = self.text_encoder.encode_prompts(texts, device=images.device)
        else:
            raise ValueError("Phải cung cấp text_input_ids, texts, hoặc cached_text_features")
            
        # Mở rộng batch dimension cho text_features tĩnh
        if text_features.dim() == 2:
            text_features = text_features.unsqueeze(0).expand(B, -1, -1) # (B, 57, 768)
        
        # 3. AttributeGCN — khai thác quan hệ giữa 57 thuộc tính
        # Input: (B, 57, 768) → Output: (B, 57, 768)
        text_features = self.attr_gcn(text_features)
            
        # 4. Nén Text từ 57 → 32
        text_features = text_features.transpose(1, 2) # (B, 768, 57)
        text_features = self.text_compress(text_features) # (B, 768, 32)
        text_features = text_features.transpose(1, 2) # (B, 32, 768)
        
        # 5. Cross-Modal Mamba
        z_out, t_out = self.cross_mamba(image_features, text_features)
        
        # 6. Fusion & MLP Head
        # Concat
        fused = torch.cat([z_out, t_out], dim=-1) # (B, 32, 1536)
        
        # Global Average Pooling theo chiều sequence
        fused_pooled = fused.mean(dim=1) # (B, 1536)
        
        # MLP → Logits
        logits = self.mlp_head(fused_pooled) # (B, 57)
        
        return logits, None

    def predict_attributes(self, images, attribute_prompts):
        """
        Hàm dự đoán nhãn trực tiếp (Inference).
        attribute_prompts: danh sách prompt tương ứng với các thuộc tính (N,)
        Trả về xác suất dự đoán (B, N) dùng sigmoid.
        """
        self.eval()
        with torch.no_grad():
            logits_per_image, _ = self.forward(images=images, texts=attribute_prompts)
            probs = torch.sigmoid(logits_per_image) # Đưa về [0, 1]
        return probs
