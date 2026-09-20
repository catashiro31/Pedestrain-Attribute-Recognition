# climp_par.py
# Mô hình CLIMP-PAR hoàn chỉnh kết hợp Vision Encoder và Text Encoder

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vision_encoder import HybridVisionEncoder
from .text_encoder import MambaTextEncoder
from .cross_mamba import CrossModalMambaBlock

class CLIMPPAR(nn.Module):
    """
    CLIMP-PAR v3: Áp dụng Cross-Modal Mamba Block và MLP Fusion.
    Luồng dữ liệu:
    - Vision: (B, 32, 768)
    - Text: (B, 57, 768) -> nén thành (B, 32, 768)
    - Cross-Modal Mamba: Fusion 2 nhánh
    - Output: Concat -> GAP -> MLP -> 57 logits
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 1. Vision Encoder (Hybrid ResNet18 + Attention + VMamba)
        self.vision_encoder = HybridVisionEncoder(
            embed_dim=config.embed_dim,
            pretrained=config.resnet_pretrained
        )
        
        # 2. Text Encoder (Mamba LLM)
        self.text_encoder = MambaTextEncoder(
            model_name=config.mamba_model,
            embed_dim=config.embed_dim,
            freeze_backbone=config.mamba_freeze
        )
        
        # 3. Text Compression (Nén 57 token xuống 32 token để khớp với ảnh)
        self.text_compress = nn.Linear(57, 32)
        
        # 4. Cross-Modal Mamba Block
        self.cross_mamba = CrossModalMambaBlock(d_model=config.embed_dim)
        
        # 5. MLP Fusion Head (Đầu ra 57 logits)
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
            
        # 3. Nén Text từ 57 -> 32
        text_features = text_features.transpose(1, 2) # (B, 768, 57)
        text_features = self.text_compress(text_features) # (B, 768, 32)
        text_features = text_features.transpose(1, 2) # (B, 32, 768)
        
        # 4. Cross-Modal Mamba
        z_out, t_out = self.cross_mamba(image_features, text_features)
        
        # 5. Fusion & MLP Head
        # Concat
        fused = torch.cat([z_out, t_out], dim=-1) # (B, 32, 1536)
        
        # Global Average Pooling theo chiều sequence
        fused_pooled = fused.mean(dim=1) # (B, 1536)
        
        # MLP -> Logits
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
