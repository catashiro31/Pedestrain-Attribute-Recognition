# climp_par.py
# Mô hình CLIMP-PAR hoàn chỉnh kết hợp Vision Encoder và Text Encoder

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vision_encoder import VMambaVisionEncoder
from .text_encoder import MambaTextEncoder
from .gcn import SemanticGCN

class CLIMPPAR(nn.Module):
    """
    Mô hình CLIMP-PAR cho nhận dạng thuộc tính người đi bộ (PAR).
    Kết hợp VMamba (Vision) và Mamba (Text) với một đầu fusion cơ bản.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 1. Vision Encoder (VMamba)
        self.vision_encoder = VMambaVisionEncoder(
            variant=config.vmamba_variant,
            embed_dim=config.embed_dim,
            pretrained=config.vmamba_pretrained,
            pretrained_path=config.vmamba_pretrained_path
        )
        
        # 2. Text Encoder (Mamba LLM)
        self.text_encoder = MambaTextEncoder(
            model_name=config.mamba_model,
            embed_dim=config.embed_dim,
            freeze_backbone=config.mamba_freeze
        )
        
        # 3. GCN Module để học mối quan hệ giữa các Text Embeddings
        self.gcn = SemanticGCN(embed_dim=config.embed_dim)
        
        # 4. Learnable temperature parameter cho Contrastive/Similarity fusion (tương tự CLIP)
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / config.temperature)))
        
        # 4. (Optional) Basic Fusion Head / Classifier
        # Nếu muốn dùng concatenate + MLPs thay vì dot product (contrastive)
        # self.fusion_head = nn.Sequential(
        #     nn.Linear(config.embed_dim * 2, config.embed_dim),
        #     nn.ReLU(),
        #     nn.Linear(config.embed_dim, 1) # Logit cho mỗi thuộc tính
        # )

    def forward(self, images, text_input_ids=None, text_attention_mask=None, texts=None, cached_text_features=None):
        """
        Forward pass tính toán logits.
        
        Args:
            images (Tensor): Batch ảnh (B, 3, H, W)
            text_input_ids (Tensor): Batch token ids cho text prompt (N, L)
            text_attention_mask (Tensor): Mask tương ứng (N, L)
            texts (list[str]): Hoặc danh sách câu prompt (N,)
            cached_text_features (Tensor): Đặc trưng văn bản đã tính sẵn (N, embed_dim)
            
        Returns:
            logits_per_image (Tensor): (B, N)
            logits_per_text (Tensor): (N, B)
        """
        # 1. Trích xuất đặc trưng ảnh
        image_features = self.vision_encoder(images)  # (B, embed_dim)
        
        # 2. Trích xuất đặc trưng văn bản
        if cached_text_features is not None:
            text_features = cached_text_features
        elif text_input_ids is not None:
            text_features = self.text_encoder(input_ids=text_input_ids, attention_mask=text_attention_mask)
        elif texts is not None:
            text_features = self.text_encoder(texts=texts, max_length=self.config.max_text_length)
        else:
            raise ValueError("Phải cung cấp text_input_ids, texts, hoặc cached_text_features")
            
        # 3. Đưa đặc trưng văn bản qua GCN để khai thác mối quan hệ giữa các thuộc tính
        # Lưu ý: GCN sẽ tự động tính ma trận kề dựa trên độ tương đồng Cosine của text_features
        text_features = self.gcn(text_features)
            
        # 4. Chuẩn hóa (L2 Normalize) - Quan trọng cho Contrastive Learning
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        # 5. Fusion cơ bản: Dot product similarity (Cosine Similarity có scale)
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.T  # (B, N)
        logits_per_text = logits_per_image.T                               # (N, B)
        
        return logits_per_image, logits_per_text

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
