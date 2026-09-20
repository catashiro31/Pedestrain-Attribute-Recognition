# climp_par.py
# Mô hình CLIMP-PAR hoàn chỉnh kết hợp Vision Encoder và Text Encoder
# v8.1: Thay thế Upsampling Block bằng Spatial Cross-Attention

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .vision_encoder import VMambaVisionEncoder
from .text_encoder import MambaTextEncoder
from .cross_mamba import CrossModalMambaBlock


class SpatialCrossAttention(nn.Module):
    """
    Spatial Cross-Attention: Vision tokens attend vào attribute text features.
    
    Thay thế Upsampling Block cũ (lossy 768→3→448×448) bằng cross-attention trực tiếp
    trên embedding space, giữ nguyên toàn bộ 768-dim semantic information.
    
    Q = vision_features (B, L_v, D) — mỗi vision token query
    K, V = diff_features (B, L_t, D) — 57 attribute embeddings đã trừ style
    
    Kiến trúc: Pre-Norm Multi-Head Cross-Attention + FFN (standard Transformer block)
    """
    def __init__(self, d_model, num_heads=8, dropout=0.1, ffn_ratio=4):
        super().__init__()
        assert d_model % num_heads == 0, f"d_model ({d_model}) phải chia hết cho num_heads ({num_heads})"
        
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Pre-Norm layers
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        
        # Multi-Head Cross-Attention projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.attn_dropout = nn.Dropout(dropout)
        
        # FFN (Feed-Forward Network) sau cross-attention
        self.norm_ffn = nn.LayerNorm(d_model)
        ffn_hidden = int(d_model * ffn_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, d_model),
            nn.Dropout(dropout),
        )
        
        # Khởi tạo weights theo Xavier
        self._init_weights()
    
    def _init_weights(self):
        """Xavier uniform initialization cho projection layers."""
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
        for module in self.ffn:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        
    def forward(self, query, key_value):
        """
        Args:
            query: (B, L_v, D) — vision features, ví dụ (B, 32, 768)
            key_value: (B, L_t, D) — text attribute features, ví dụ (B, 57, 768)
            
        Returns:
            out: (B, L_v, D) — vision features đã được conditioned bởi text attributes
        """
        B, L_v, D = query.shape
        L_t = key_value.shape[1]
        
        # === Cross-Attention Block ===
        # 1. Pre-Norm
        q = self.norm_q(query)
        kv = self.norm_kv(key_value)
        
        # 2. Multi-Head Projection: (B, L, D) → (B, num_heads, L, head_dim)
        q = self.q_proj(q).view(B, L_v, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv).view(B, L_t, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv).view(B, L_t, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 3. Scaled Dot-Product Attention
        attn_weights = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, L_v, L_t)
        attn_weights = attn_weights.softmax(dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # 4. Apply attention and project back
        attn_out = (attn_weights @ v)  # (B, H, L_v, head_dim)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L_v, D)
        attn_out = self.out_proj(attn_out)
        
        # 5. Residual connection
        x = query + attn_out
        
        # === FFN Block ===
        x = x + self.ffn(self.norm_ffn(x))
        
        return x


class CLIMPPAR(nn.Module):
    """
    CLIMP-PAR v8.1: Spatial Cross-Attention thay thế Upsampling Block.
    
    Luồng dữ liệu mới:
    - Text Branch: domain_tokens + hidden_words → Mamba-130M → diff_features (B, 57, 768) + full_features (B, 768)
    - Vision Branch: ảnh → VMamba-Small → vision_features (B, 32, 768)
    - Spatial Cross-Attention: vision_features (Q) attend vào diff_features (K,V) → conditioned_vision (B, 32, 768)
    - Cross-Modal Mamba: Fusion conditioned_vision + full_features
    - Output: Concat → GAP → MLP → 57 logits
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 1. Vision Encoder (VMamba-Tiny/Small)
        self.vision_encoder = VMambaVisionEncoder(
            variant=getattr(config, 'vmamba_variant', 'tiny'),
            embed_dim=config.embed_dim,
            pretrained_path=config.vmamba_pretrained_path,
            use_checkpoint=getattr(config, 'gradient_checkpointing', False)
        )
        
        # 2. Text Encoder (Mamba LLM)
        self.text_encoder = MambaTextEncoder(
            model_name=config.mamba_model,
            embed_dim=config.embed_dim,
            freeze_backbone=config.mamba_freeze
        )
        
        # 3. Soft Prompts (Hidden Words)
        self.num_hidden_words = 16
        self.hidden_words = nn.Parameter(torch.randn(self.num_hidden_words, config.embed_dim))
        
        # 4. Spatial Cross-Attention (THAY THẾ Upsampling Block)
        # Vision tokens (Q) attend vào text attribute features (K,V)
        # Giữ nguyên semantic information trong 768-dim space
        self.spatial_cross_attn = SpatialCrossAttention(
            d_model=config.embed_dim,
            num_heads=8,
            dropout=0.1,
            ffn_ratio=4
        )
        
        # 5. Cross-Modal Mamba Block
        self.cross_mamba = CrossModalMambaBlock(d_model=config.embed_dim)
        
        # 6. MLP Fusion Head (Đầu ra 57 logits)
        self.mlp_head = nn.Sequential(
            nn.Linear(config.embed_dim * 2, config.embed_dim),
            nn.LayerNorm(config.embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.embed_dim, 57)
        )
        
        # 7. Đóng băng BACKBONE của Vision & Text Encoder
        # Chỉ freeze backbone layers, giữ projection heads trainable
        self._freeze_backbones()

    def _freeze_backbones(self):
        """
        Đóng băng backbone layers nhưng giữ projection heads trainable.
        - Vision: freeze backbone (VSSM), giữ norm + projection trainable
        - Text: freeze backbone (MambaLMHeadModel), giữ norm + projection trainable
        """
        # Vision Encoder: freeze backbone, unfreeze projection
        for param in self.vision_encoder.backbone.parameters():
            param.requires_grad = False
        # Giữ vision_encoder.norm, vision_encoder.projection trainable
        for param in self.vision_encoder.norm.parameters():
            param.requires_grad = True
        for param in self.vision_encoder.projection.parameters():
            param.requires_grad = True
            
        # Text Encoder: backbone đã được freeze trong __init__ nếu mamba_freeze=True
        # Giữ text_encoder.norm, text_encoder.projection trainable
        for param in self.text_encoder.norm.parameters():
            param.requires_grad = True
        for param in self.text_encoder.projection.parameters():
            param.requires_grad = True

    def forward(self, images, domain_tokens, attribute_names=None):
        """
        Forward pass tính toán logits.
        images: (B, 3, 448, 448)
        domain_tokens: (B, 32, 768)
        attribute_names: List[str] len 57
        """
        B = images.shape[0]
        device = images.device
        
        assert attribute_names is not None, "Phải cung cấp attribute_names"
        num_attrs = len(attribute_names) # 57
        
        # ---------------- 1. TEXT BRANCH ----------------
        
        # [OPTIMIZATION]
        # Thường trong 1 batch, tất cả các ảnh đều có chung 1 domain_token (vì load từ 1 dataset).
        # Nếu ta không gộp, Text Branch phải chạy Mamba (B * 57) lần = RẤT CHẬM VÀ TỐN VRAM.
        # Ta lọc ra các domain_token duy nhất, chạy Text Encoder, sau đó ánh xạ ngược lại cho batch.
        unique_domain_tokens, inverse_indices = torch.unique(domain_tokens, dim=0, return_inverse=True)
        B_u = unique_domain_tokens.shape[0]
        
        hw = self.hidden_words.unsqueeze(0).expand(B_u, -1, -1) # (B_u, M, 768)
        
        # -- Style embeddings (domain + hidden)
        style_embeds_u = torch.cat([unique_domain_tokens, hw], dim=1) # (B_u, 32+M, 768)
        style_features_u = self.text_encoder(inputs_embeds=style_embeds_u) # (B_u, 768)
        
        # -- Attribute embeddings (57 times)
        tokens = self.text_encoder.tokenize(attribute_names, device=device)
        attr_input_ids = tokens['input_ids'] # (57, L)
        attr_mask = tokens['attention_mask'] # (57, L)
        
        word_embeds = self.text_encoder.backbone.backbone.embedding(attr_input_ids) # (57, L, 768)
        
        style_embeds_exp_u = style_embeds_u.unsqueeze(1).expand(-1, num_attrs, -1, -1) # (B_u, 57, 32+M, 768)
        word_embeds_exp_u = word_embeds.unsqueeze(0).expand(B_u, -1, -1, -1) # (B_u, 57, L, 768)
        
        attr_inputs_embeds_u = torch.cat([style_embeds_exp_u, word_embeds_exp_u], dim=2) # (B_u, 57, 32+M+L, 768)
        
        style_mask_u = torch.ones(B_u, num_attrs, style_embeds_u.shape[1], device=device, dtype=torch.long)
        attr_mask_exp_u = attr_mask.unsqueeze(0).expand(B_u, -1, -1) # (B_u, 57, L)
        full_attr_mask_u = torch.cat([style_mask_u, attr_mask_exp_u], dim=2) # (B_u, 57, 32+M+L)
        
        # Flatten and compute
        attr_inputs_embeds_flat = attr_inputs_embeds_u.view(B_u * num_attrs, -1, self.config.embed_dim)
        full_attr_mask_flat = full_attr_mask_u.view(B_u * num_attrs, -1)
        
        attr_features_flat = self.text_encoder(inputs_embeds=attr_inputs_embeds_flat, attention_mask=full_attr_mask_flat)
        attr_features_u = attr_features_flat.view(B_u, num_attrs, self.config.embed_dim) # (B_u, 57, 768)
        
        # Trừ style từ attribute
        diff_features_u = attr_features_u - style_features_u.unsqueeze(1) # (B_u, 57, 768)
        
        # -- Full embeddings (all attributes combined)
        full_text = ' '.join(attribute_names)
        full_tokens = self.text_encoder.tokenize([full_text], device=device)
        full_ids = full_tokens['input_ids'] # (1, L_full)
        full_mask = full_tokens['attention_mask'] # (1, L_full)
        
        full_word_embeds = self.text_encoder.backbone.backbone.embedding(full_ids) # (1, L_full, 768)
        full_word_embeds_exp_u = full_word_embeds.expand(B_u, -1, -1) # (B_u, L_full, 768)
        
        full_inputs_embeds_u = torch.cat([style_embeds_u, full_word_embeds_exp_u], dim=1) # (B_u, 32+M+L_full, 768)
        
        style_mask_full_u = torch.ones(B_u, style_embeds_u.shape[1], device=device, dtype=torch.long)
        full_mask_exp_u = full_mask.expand(B_u, -1)
        full_attention_mask_u = torch.cat([style_mask_full_u, full_mask_exp_u], dim=1)
        
        full_features_u = self.text_encoder(inputs_embeds=full_inputs_embeds_u, attention_mask=full_attention_mask_u) # (B_u, 768)
        
        # Ánh xạ ngược lại kích thước B
        diff_features = diff_features_u[inverse_indices] # (B, 57, 768)
        full_features = full_features_u[inverse_indices] # (B, 768)
        
        # ---------------- 2. VISION BRANCH ----------------
        # [MỚI] Ảnh đi trực tiếp qua VMamba, KHÔNG concat z nữa
        image_features = self.vision_encoder(images) # (B, 32, 768)
        
        # ---------------- 3. SPATIAL CROSS-ATTENTION ----------------
        # [MỚI] Vision tokens attend vào text attribute features
        # Q = image_features (B, 32, 768), K/V = diff_features (B, 57, 768)
        # → Mỗi vision token tìm kiếm thông tin thuộc tính liên quan từ text
        conditioned_vision = self.spatial_cross_attn(image_features, diff_features) # (B, 32, 768)
        
        # ---------------- 4. CROSS-MODAL & FUSION ----------------
        
        # Expand full_features to match vision length
        text_features_exp = full_features.unsqueeze(1).expand(-1, 32, -1) # (B, 32, 768)
        
        # Cross-Modal Mamba
        z_out, t_out = self.cross_mamba(conditioned_vision, text_features_exp)
        
        # Fusion & MLP Head
        fused = torch.cat([z_out, t_out], dim=-1) # (B, 32, 1536)
        fused_pooled = fused.mean(dim=1) # (B, 1536)
        
        logits = self.mlp_head(fused_pooled) # (B, 57)
        features = {
            'image_features': image_features,
            'full_features': full_features,
            'diff_features': diff_features
        }
        return logits, features

    def predict_attributes(self, images, domain_tokens, attribute_prompts):
        """
        Hàm dự đoán nhãn trực tiếp (Inference).
        attribute_prompts: danh sách prompt tương ứng với các thuộc tính (N,)
        Trả về xác suất dự đoán (B, N) dùng sigmoid.
        """
        self.eval()
        with torch.no_grad():
            logits_per_image, _ = self.forward(images=images, domain_tokens=domain_tokens, attribute_names=attribute_prompts)
            probs = torch.sigmoid(logits_per_image) # Đưa về [0, 1]
        return probs
