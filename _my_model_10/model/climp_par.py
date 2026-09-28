import torch
import torch.nn as nn
import torch.nn.functional as F

from .vision_encoder import VMambaVisionEncoder
from .text_encoder import MambaTextEncoder
from .cross_mamba import CrossModalMambaBlock
from .background_encoder import BackgroundEncoder


from .hidden_words import HiddenWords
from .prompt_generator import PromptGenerator

class CLIMPPAR(nn.Module):
    """
    CLIMP-PAR v4: Áp dụng Background Extraction, Hidden Words, Disentanglement.
    Luồng dữ liệu:
    - Vision: (B, 32, 768)
    - Background: bg_images -> (B, 768)
    - Text: PromptGenerator tạo Style Prompt và Attr Prompts
    - TextEncoder: Mã hóa prompts -> style_emb, attr_embs
    - Disentanglement: attr_embs - style_emb -> disentangled_embs (B, 57, 768)
    - Text Compression: nén thành (B, 32, 768)
    - Cross-Modal Mamba: Fusion 2 nhánh
    - Output: Concat -> GAP -> MLP -> 57 logits
    """
    def __init__(self, config, attributes):
        super().__init__()
        self.config = config
        
        # 1. Vision Encoder (VMamba-Tiny)
        self.vision_encoder = VMambaVisionEncoder(
            embed_dim=config.embed_dim,
            pretrained_path=config.vmamba_pretrained_path,
            use_checkpoint=getattr(config, 'gradient_checkpointing', False)
        )
        
        # # Đóng băng 2 block đầu (layers.0 và layers.1) của VMamba để tiết kiệm VRAM
        # for name, param in self.vision_encoder.backbone.named_parameters():
        #     if "layers.0" in name or "layers.1" in name:
        #         param.requires_grad = False

        
        # 2. Background Encoder
        checkpoint_path = getattr(config, 'bg_checkpoint', 'resnet18_background_best.pth')
        self.background_encoder = BackgroundEncoder(
            embed_dim=config.embed_dim, 
            checkpoint_path=checkpoint_path
        )
        
        # 3. Hidden Words
        self.hidden_words = HiddenWords(
            embed_dim=config.embed_dim,
            num_groups=12,
            tokens_per_group=4
        )
        
        # 4. Text Encoder (Mamba LLM)
        self.text_encoder = MambaTextEncoder(
            model_name=config.mamba_model,
            embed_dim=config.embed_dim,
            freeze_backbone=config.mamba_freeze
        )
        
        # 5. Prompt Generator
        self.prompt_generator = PromptGenerator(
            hidden_words=self.hidden_words,
            text_encoder=self.text_encoder,
            attributes=attributes
        )
        
        # Khởi tạo CoOp token bằng word embeddings thực tế
        device = torch.device(config.device if hasattr(config, 'device') else 'cpu')
        self.hidden_words.initialize_with_words(self.text_encoder, device=device)
        
        # 6. Semantic Grouping (Tận dụng mảng phân nhóm có sẵn từ PromptGenerator)
        # self.prompt_generator.attr_to_hw_group là list (len 57) chứa group ID (từ 1 đến 11)
        matrix = torch.zeros(len(attributes), 11)
        for i, group_id in enumerate(self.prompt_generator.attr_to_hw_group):
            # group_id chạy từ 1->11, matrix index chạy từ 0->10
            matrix[i, group_id - 1] = 1.0
            
        # Chuẩn hóa để tính trung bình cộng (Mean Pooling)
        col_sums = matrix.sum(dim=0, keepdim=True)
        col_sums[col_sums == 0] = 1.0
        pool_matrix = matrix / col_sums
        
        self.register_buffer('pool_matrix', pool_matrix)
        
        # Nén Image từ 32 patches xuống 11 vùng ngữ nghĩa để khớp với Text
        self.img_compress = nn.Linear(32, 11)
        
        # 7. Cross-Modal Mamba Block
        self.cross_mamba = CrossModalMambaBlock(d_model=config.embed_dim)
        
        # 8. MLP Fusion Head (Đầu ra 57 logits)
        self.mlp_head = nn.Sequential(
            nn.Linear(config.embed_dim * 2, config.embed_dim),
            nn.LayerNorm(config.embed_dim),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(config.embed_dim, len(attributes))
        )

    def forward(self, images, bg_images, return_features=False):
        """
        Forward pass tính toán logits v4.
        """
        B = images.shape[0]
        
        # 1. Trích xuất đặc trưng hình ảnh
        image_features = self.vision_encoder(images) # (B, 32, 768)
        
        # 2. Trích xuất background context
        bg_context = self.background_encoder(bg_images) # (B, 768)
        
        # 3. Tạo Prompt Embeddings
        style_embeds, style_mask, attr_embeds, attr_masks = self.prompt_generator(bg_context)
        # style_embeds: (B, L_style, 768)
        # attr_embeds: (B, 57, L_attr, 768)
        
        # 4. Mã hóa qua Text Encoder
        # Style
        style_emb = self.text_encoder.encode_from_embeddings(style_embeds, style_mask) # (B, 768)
        
        # Attribute
        # Gộp batch và số lượng thuộc tính để qua text encoder một lần
        num_attrs = attr_embeds.size(1)
        L_attr = attr_embeds.size(2)
        
        attr_embeds_flat = attr_embeds.view(B * num_attrs, L_attr, -1)
        attr_masks_flat = attr_masks.view(B * num_attrs, L_attr)
        
        attr_embs_flat = self.text_encoder.encode_from_embeddings(attr_embeds_flat, attr_masks_flat) # (B*57, 768)
        attr_embs = attr_embs_flat.view(B, num_attrs, -1) # (B, 57, 768)
        
        # 5. Disentanglement
        # Trừ đi Style Embedding (loại bỏ bias domain)
        disentangled = attr_embs - style_emb.unsqueeze(1) # (B, 57, 768)
        
        # 6. Semantic Grouping & Compression
        # Image: (B, 32, 768) -> (B, 11, 768)
        image_features = self.img_compress(image_features.transpose(1, 2)).transpose(1, 2)
        
        # Text Pooling: (B, 57, 768) -> (B, 11, 768)
        text_groups = torch.matmul(disentangled.transpose(1, 2), self.pool_matrix) # (B, 768, 11)
        text_groups = text_groups.transpose(1, 2) # (B, 11, 768)
        
        # 7. Cross-Modal Mamba
        z_out, t_out = self.cross_mamba(image_features, text_groups)
        
        # 8. Fusion & MLP Head
        fused = torch.cat([z_out, t_out], dim=-1) # (B, 11, 1536)
        fused_pooled = fused.mean(dim=1) # (B, 1536)
        
        logits = self.mlp_head(fused_pooled) # (B, 57)
        
        if return_features:
            return logits, (disentangled, style_emb, bg_context)
            
        return logits, None
