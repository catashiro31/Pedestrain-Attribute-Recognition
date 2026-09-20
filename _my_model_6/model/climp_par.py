# climp_par.py
# Mô hình CLIMP-PAR v6 hoàn chỉnh kết hợp:
#   - Vision Encoder (VMamba-Tiny)
#   - Text Encoder (Mamba LLM)
#   - Conditional Prompt Learning (CoCoOp) cho nhánh text
#   - Cross-Modal Mamba Block
#   - MLP Fusion Head
#
# Khác biệt chính so với v4:
#   Text features KHÔNG còn là tĩnh (cache 1 lần mỗi epoch).
#   Thay vào đó, mỗi batch sẽ sinh prompt embeddings riêng biệt 
#   phụ thuộc vào ảnh đầu vào (instance-conditional), rồi chạy qua text encoder.
#   Chỉ word embeddings thô mới có thể cache.

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vision_encoder import VMambaVisionEncoder
from .text_encoder import MambaTextEncoder
from .cross_mamba import CrossModalMambaBlock
from .conditional_prompt import ConditionalPromptLearner


class CLIMPPAR(nn.Module):
    """
    CLIMP-PAR v6: Bổ sung Conditional Prompt Learning (CoCoOp) cho nhánh Text.
    
    Luồng dữ liệu mới:
    1. Vision Encoder: Ảnh → (B, 32, 768) vision tokens
    2. Global Pooling: (B, 32, 768) → (B, 768) global feature cho Meta-Net
    3. ConditionalPromptLearner:
       - Meta-Net(global_feature) → bias (B, 768)
       - ctx_shifted = learnable_ctx + bias → (B, n_ctx, 768)
       - prompt_embs = concat(ctx_shifted, word_embeddings) → (B*57, n_ctx+L_w, 768)
    4. Text Encoder: SSM blocks → last-token pooling → projection → (B, 57, 768)
    5. Text Compression: 57→32 tokens
    6. Cross-Modal Mamba: Fusion 2 nhánh
    7. Output: Concat → GAP → MLP → 57 logits
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 1. Vision Encoder (VMamba-Tiny) — giữ nguyên từ v4
        self.vision_encoder = VMambaVisionEncoder(
            embed_dim=config.embed_dim,
            pretrained_path=config.vmamba_pretrained_path
        )
        
        # 2. Text Encoder (Mamba LLM) — freeze backbone theo CoCoOp
        self.text_encoder = MambaTextEncoder(
            model_name=config.mamba_model,
            embed_dim=config.embed_dim,
            freeze_backbone=config.mamba_freeze
        )
        
        # 3. Conditional Prompt Learner (CoCoOp) — MODULE MỚI
        self.prompt_learner = ConditionalPromptLearner(
            n_ctx=config.n_ctx,
            embed_dim=self.text_encoder.d_model,  # d_model của Mamba backbone (768 cho mamba-130m)
            n_attrs=57,
            meta_net_hidden_dim=config.meta_net_hidden_dim,
        )
        
        # 4. Projection: d_model → embed_dim cho vision global feature
        # (Vì Meta-Net nhận embed_dim nhưng vision projection đã output embed_dim,
        #  nên ta dùng trực tiếp nếu embed_dim == d_model. Nếu khác, cần thêm projection.)
        # Ở đây embed_dim = 768 = d_model nên không cần thêm projection.
        
        # 5. Text Compression (Nén 57 token xuống 32 token để khớp với ảnh) — giữ nguyên
        self.text_compress = nn.Linear(57, 32)
        
        # 6. Cross-Modal Mamba Block — giữ nguyên
        self.cross_mamba = CrossModalMambaBlock(d_model=config.embed_dim)
        
        # 7. MLP Fusion Head (Đầu ra 57 logits) — giữ nguyên
        self.mlp_head = nn.Sequential(
            nn.Linear(config.embed_dim * 2, config.embed_dim),
            nn.LayerNorm(config.embed_dim),
            nn.GELU(),
            nn.Linear(config.embed_dim, 57)
        )
        
        # 8. Gradient Checkpointing
        if getattr(config, 'gradient_checkpointing', False):
            self.vision_encoder.enable_gradient_checkpointing()
            self.text_encoder.enable_gradient_checkpointing()
        
        # 9. Text forward batch size (chunked processing để giảm peak VRAM)
        self.text_forward_batch_size = getattr(config, 'text_forward_batch_size', 0)

    def forward(self, images, cached_word_embeddings=None, cached_word_mask=None,
                text_input_ids=None, text_attention_mask=None, texts=None,
                cached_text_features=None):
        """
        Forward pass tính toán logits.
        
        Hỗ trợ 2 chế độ:
        
        Chế độ 1 — CoCoOp (MỚI, khuyến nghị):
            Truyền cached_word_embeddings + cached_word_mask.
            → Sử dụng ConditionalPromptLearner để sinh prompt embeddings phụ thuộc ảnh.
            
        Chế độ 2 — Legacy (tương thích ngược với v4):
            Truyền cached_text_features hoặc text_input_ids/texts.
            → Bỏ qua ConditionalPromptLearner, dùng text features tĩnh.
        
        Args:
            images: (B, 3, H, W)
            cached_word_embeddings: (N_attr, L_w, d_model) — word embeddings đã cache
            cached_word_mask: (N_attr, L_w) — attention mask cho word embeddings
            text_input_ids, text_attention_mask, texts: Legacy inputs (v4 compatible)
            cached_text_features: (N_attr, embed_dim) — Legacy cached text features
        """
        # 1. Trích xuất đặc trưng hình ảnh
        # image_features: (B, 32, 768)
        image_features = self.vision_encoder(images)
        B = image_features.shape[0]
        
        # ================================================================
        # CHẾ ĐỘ 1: CoCoOp — Conditional Prompt Learning
        # ================================================================
        if cached_word_embeddings is not None:
            # 2a. Global Average Pooling để lấy global vision feature
            # (B, 32, 768) → (B, 768)
            image_global = image_features.mean(dim=1)
            
            # 2b. Sinh prompt embeddings có điều kiện
            # prompt_embs: (B * N_attr, n_ctx + L_w, d_model)
            # prompt_mask: (B * N_attr, n_ctx + L_w)
            prompt_embs, prompt_mask = self.prompt_learner(
                image_global, cached_word_embeddings, cached_word_mask
            )
            
            # 2c. Chạy qua text encoder (SSM blocks + last-token pooling + projection)
            # text_features_flat: (B * N_attr, embed_dim)
            total_seqs = prompt_embs.shape[0]  # B * N_attr
            if self.text_forward_batch_size > 0 and total_seqs > self.text_forward_batch_size:
                # Chunked processing: chia nhỏ để giảm peak VRAM
                chunks = prompt_embs.split(self.text_forward_batch_size, dim=0)
                mask_chunks = prompt_mask.split(self.text_forward_batch_size, dim=0)
                text_features_flat = torch.cat([
                    self.text_encoder.forward_with_prompt_embeddings(c, m)
                    for c, m in zip(chunks, mask_chunks)
                ], dim=0)
            else:
                text_features_flat = self.text_encoder.forward_with_prompt_embeddings(
                    prompt_embs, prompt_mask
                )
            
            # 2d. Reshape: (B * N_attr, embed_dim) → (B, N_attr, embed_dim) = (B, 57, 768)
            N_attr = cached_word_embeddings.shape[0]
            text_features = text_features_flat.view(B, N_attr, -1)  # (B, 57, 768)
        
        # ================================================================
        # CHẾ ĐỘ 2: Legacy — Text features tĩnh (tương thích v4)
        # ================================================================
        elif cached_text_features is not None:
            text_features = cached_text_features  # (57, 768)
            if text_features.dim() == 2:
                text_features = text_features.unsqueeze(0).expand(B, -1, -1)  # (B, 57, 768)
        elif text_input_ids is not None:
            text_features = self.text_encoder(text_input_ids, text_attention_mask)
            if text_features.dim() == 2:
                text_features = text_features.unsqueeze(0).expand(B, -1, -1)
        elif texts is not None:
            text_features = self.text_encoder.encode_prompts(texts, device=images.device)
            if text_features.dim() == 2:
                text_features = text_features.unsqueeze(0).expand(B, -1, -1)
        else:
            raise ValueError(
                "Phải cung cấp cached_word_embeddings (CoCoOp mode), "
                "cached_text_features, text_input_ids, hoặc texts"
            )
            
        # 3. Nén Text từ 57 → 32
        text_features = text_features.transpose(1, 2)   # (B, 768, 57)
        text_features = self.text_compress(text_features) # (B, 768, 32)
        text_features = text_features.transpose(1, 2)    # (B, 32, 768)
        
        # 4. Cross-Modal Mamba
        z_out, t_out = self.cross_mamba(image_features, text_features)
        
        # 5. Fusion & MLP Head
        # Concat
        fused = torch.cat([z_out, t_out], dim=-1)  # (B, 32, 1536)
        
        # Global Average Pooling theo chiều sequence
        fused_pooled = fused.mean(dim=1)  # (B, 1536)
        
        # MLP → Logits
        logits = self.mlp_head(fused_pooled)  # (B, 57)
        
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
            probs = torch.sigmoid(logits_per_image)  # Đưa về [0, 1]
        return probs
