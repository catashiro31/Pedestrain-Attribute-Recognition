# conditional_prompt.py
# Module Conditional Prompt Learning theo cơ chế CoCoOp (Conditional Context Optimization)
# Tham khảo: Zhou et al., "Conditional Prompt Learning for Vision-Language Models", CVPR 2022
#
# Ý tưởng cốt lõi:
#   - Thay vì dùng prompt tĩnh (cố định cho mọi ảnh như CoOp),
#     CoCoOp sinh "instance-conditional prompt vectors" — mỗi ảnh đầu vào 
#     sẽ tạo ra một bộ prompt riêng biệt, giúp mô hình tổng quát hóa tốt hơn.
#   - Meta-Net: MLP nhỏ nhận global vision feature, sinh conditional bias vector.
#   - Conditional contexts = learnable_ctx + meta_net(image_feature)
#   - Prompt embedding = [conditional_ctx] + [word_embeddings_of_attribute]

import torch
import torch.nn as nn


class MetaNet(nn.Module):
    """
    Meta-Net theo CoCoOp: MLP nhỏ sinh instance-conditional bias vector 
    từ global vision feature.
    
    Kiến trúc:
        Linear(embed_dim, hidden_dim) → ReLU → Linear(hidden_dim, embed_dim)
    
    Bias vector này sẽ được cộng vào learnable context vectors,
    khiến prompt text thay đổi theo từng ảnh cụ thể.
    
    Args:
        embed_dim (int): Chiều embedding (768)
        hidden_dim (int): Chiều hidden layer (mặc định: embed_dim // 16 = 48)
    """
    def __init__(self, embed_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim // 16  # 768 // 16 = 48
        
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embed_dim),
        )
    
    def forward(self, x):
        """
        Args:
            x: Global vision feature, shape (B, embed_dim)
        Returns:
            bias: Conditional bias vector, shape (B, embed_dim)
        """
        return self.net(x)


class ConditionalPromptLearner(nn.Module):
    """
    Conditional Prompt Learner theo CoCoOp cho CLIMP-PAR.
    
    Quản lý:
    1. Learnable context vectors (n_ctx, embed_dim) — chia sẻ cho mọi ảnh
    2. Meta-Net: sinh conditional bias từ vision feature
    3. Kết hợp: ctx_shifted = ctx + meta_net(image_global)
    4. Ghép nối: [ctx_shifted | word_embeddings] → prompt embedding hoàn chỉnh
    
    Luồng dữ liệu:
        image_global (B, D) ──► Meta-Net ──► bias (B, D)
                                                 │
        learnable ctx (n_ctx, D) ──────► ctx + bias ──► (B, n_ctx, D)
                                                            │
        word_embs (N_attr, L_w, D) ────────────────► Concat ──► (B, N_attr, n_ctx+L_w, D)
    
    Args:
        n_ctx (int): Số lượng learnable context tokens. Mặc định: 4
        embed_dim (int): Chiều embedding. Mặc định: 768
        n_attrs (int): Số thuộc tính (57 cho MSP60k). Mặc định: 57
        meta_net_hidden_dim (int): Hidden dim của Meta-Net. Mặc định: 48
    """
    def __init__(self, n_ctx=4, embed_dim=768, n_attrs=57, meta_net_hidden_dim=48):
        super().__init__()
        
        self.n_ctx = n_ctx
        self.embed_dim = embed_dim
        self.n_attrs = n_attrs
        
        # 1. Learnable context vectors: (n_ctx, embed_dim)
        # Khởi tạo random từ phân phối chuẩn, scale nhỏ
        ctx_vectors = torch.empty(n_ctx, embed_dim)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)  # (n_ctx, embed_dim)
        
        # 2. Meta-Net: sinh conditional bias từ vision feature
        self.meta_net = MetaNet(
            embed_dim=embed_dim, 
            hidden_dim=meta_net_hidden_dim
        )
        
        print(f"[ConditionalPromptLearner] n_ctx={n_ctx}, embed_dim={embed_dim}, "
              f"n_attrs={n_attrs}, meta_hidden={meta_net_hidden_dim}")
    
    def forward(self, image_global, word_embeddings, word_mask=None):
        """
        Tạo prompt embeddings có điều kiện theo ảnh đầu vào.
        
        Args:
            image_global (Tensor): Global vision feature, shape (B, embed_dim)
                Đây là kết quả global average pooling của vision tokens.
            word_embeddings (Tensor): Word embeddings đã cache, shape (N_attr, L_w, embed_dim)
                L_w = max sequence length của từng attribute prompt sau khi tokenize.
            word_mask (Tensor, optional): Attention mask cho word embeddings, 
                shape (N_attr, L_w). 1=valid, 0=padding.
        
        Returns:
            prompt_embeddings (Tensor): shape (B * N_attr, n_ctx + L_w, embed_dim)
                Prompt embedding hoàn chỉnh cho từng ảnh × từng thuộc tính.
            prompt_mask (Tensor): Attention mask tương ứng, 
                shape (B * N_attr, n_ctx + L_w). 1=valid, 0=padding.
        """
        B = image_global.shape[0]
        N_attr = word_embeddings.shape[0]
        L_w = word_embeddings.shape[1]
        
        # 1. Sinh conditional bias từ vision feature
        # bias: (B, embed_dim)
        bias = self.meta_net(image_global)
        
        # 2. Tạo instance-conditional context
        # ctx: (n_ctx, embed_dim) → expand thành (B, n_ctx, embed_dim)
        # bias: (B, embed_dim) → (B, 1, embed_dim) để broadcast
        ctx = self.ctx.unsqueeze(0).expand(B, -1, -1)          # (B, n_ctx, D)
        bias = bias.unsqueeze(1)                                # (B, 1, D)
        ctx_shifted = ctx + bias                                # (B, n_ctx, D)
        
        # 3. Mở rộng ctx_shifted cho mỗi thuộc tính
        # (B, n_ctx, D) → (B, 1, n_ctx, D) → (B, N_attr, n_ctx, D)
        ctx_shifted = ctx_shifted.unsqueeze(1).expand(-1, N_attr, -1, -1)
        
        # 4. Mở rộng word_embeddings cho mỗi ảnh
        # (N_attr, L_w, D) → (1, N_attr, L_w, D) → (B, N_attr, L_w, D)
        word_embs = word_embeddings.unsqueeze(0).expand(B, -1, -1, -1)
        
        # 5. Ghép nối: [ctx_shifted | word_embeddings]
        # (B, N_attr, n_ctx + L_w, D)
        prompt_embeddings = torch.cat([ctx_shifted, word_embs], dim=2)
        
        # 6. Reshape thành (B * N_attr, n_ctx + L_w, D) để đưa vào text encoder
        prompt_embeddings = prompt_embeddings.view(B * N_attr, self.n_ctx + L_w, self.embed_dim)
        
        # 7. Tạo attention mask
        if word_mask is not None:
            # Context tokens luôn valid (mask = 1)
            ctx_mask = torch.ones(N_attr, self.n_ctx, device=word_mask.device, dtype=word_mask.dtype)
            # Ghép: (N_attr, n_ctx + L_w)
            full_mask = torch.cat([ctx_mask, word_mask], dim=1)
            # Expand cho batch: (1, N_attr, n_ctx + L_w) → (B, N_attr, n_ctx + L_w)
            prompt_mask = full_mask.unsqueeze(0).expand(B, -1, -1)
            # Reshape: (B * N_attr, n_ctx + L_w)
            prompt_mask = prompt_mask.reshape(B * N_attr, self.n_ctx + L_w)
        else:
            # Không có mask → tất cả đều valid
            prompt_mask = torch.ones(
                B * N_attr, self.n_ctx + L_w, 
                device=prompt_embeddings.device, dtype=torch.long
            )
        
        return prompt_embeddings, prompt_mask
