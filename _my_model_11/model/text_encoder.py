# text_encoder.py
# Mamba Text Encoder wrapper cho CLIMP-PAR
# Dựa trên kiến trúc CLIMP (arXiv:2601.06891):
#   - Mamba LLM pretrained làm text backbone (thay thế Transformer text encoder)
#   - Last-token pooling: lấy hidden state ở vị trí token cuối (causal → full context)
#   - Projection W_t chiếu sang shared embedding space
#   - Hỗ trợ văn bản dài hơn 77 token (vượt giới hạn CLIP gốc)

import os
import sys
import torch
import torch.nn as nn

# Thêm đường dẫn mamba local vào sys.path
_MAMBA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'mamba')
if _MAMBA_DIR not in sys.path:
    sys.path.insert(0, _MAMBA_DIR)

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel


# ============================================================================
# Cấu hình Mamba Text Encoder
# ============================================================================
MAMBA_PRETRAINED_MODELS = {
    # Mamba-1 variants (trên Hugging Face: state-spaces/mamba-*)
    'mamba-130m': {
        'hf_name': 'state-spaces/mamba-130m-hf',
        'd_model': 768,
        'n_layer': 24,
    },
    'mamba-370m': {
        'hf_name': 'state-spaces/mamba-370m-hf',
        'd_model': 1024,
        'n_layer': 48,
    },
    'mamba-790m': {
        'hf_name': 'state-spaces/mamba-790m-hf',
        'd_model': 1536,
        'n_layer': 48,
    },
    'mamba-1.4b': {
        'hf_name': 'state-spaces/mamba-1.4b-hf',
        'd_model': 2048,
        'n_layer': 48,
    },
    # Mamba-2 variants
    'mamba2-130m': {
        'hf_name': 'state-spaces/mamba2-130m',
        'd_model': 768,
        'n_layer': 24,
    },
    'mamba2-370m': {
        'hf_name': 'state-spaces/mamba2-370m',
        'd_model': 1024,
        'n_layer': 48,
    },
    'mamba2-780m': {
        'hf_name': 'state-spaces/mamba2-780m',
        'd_model': 1536,
        'n_layer': 48,
    },
    'mamba2-1.3b': {
        'hf_name': 'state-spaces/mamba2-1.3b',
        'd_model': 2048,
        'n_layer': 48,
    },
}


class MambaTextEncoder(nn.Module):
    """
    Text Encoder dựa trên Mamba LLM pretrained cho CLIMP-PAR.
    
    Theo bài báo CLIMP (Section 3.2 - Text Encoder):
    - Sử dụng pretrained Mamba LLM (Mamba-1 hoặc Mamba-2) làm text backbone
    - Last-token pooling: trích hidden state tại vị trí token cuối cùng (non-padding)
      → Đây là vị trí duy nhất có đầy đủ ngữ cảnh do tính causal của Mamba
    - Projection W_t chiếu representation sang shared embedding space
    - Vượt giới hạn 77-token của CLIP nhờ cơ chế autoregressive
    
    Công thức (Eq. từ CLIMP):
        H = [h1, h2, ..., hL] = MambaBackbone(T)
        text_repr = H[last_non_pad_idx]
        text_embedding = W_t @ text_repr
    
    Args:
        model_name (str): Tên model Mamba pretrained. Mặc định: 'mamba-130m'
        embed_dim (int): Chiều shared embedding space. Mặc định: 768
        cache_dir (str, optional): Thư mục cache lưu model tải về.
        freeze_backbone (bool): Đóng băng backbone khi huấn luyện. Mặc định: False
    """
    
    def __init__(self, model_name='mamba-130m', embed_dim=768, cache_dir=None, freeze_backbone=False):
        super().__init__()
        
        self.model_name = model_name
        self.embed_dim = embed_dim
        
        # Lấy cấu hình
        if model_name in MAMBA_PRETRAINED_MODELS:
            config = MAMBA_PRETRAINED_MODELS[model_name]
            self.d_model = config['d_model']
            self.hf_name = config['hf_name']
        else:
            # Fallback: coi model_name là Hugging Face model name trực tiếp
            self.hf_name = model_name
            self.d_model = None  # Sẽ xác định sau khi load model
        
        # Cache directory mặc định
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'checkpoints', 'mamba_text')
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_dir = cache_dir
        
        # 1. Load Mamba backbone + tokenizer từ Hugging Face
        self.backbone, self.tokenizer = self._load_model_and_tokenizer()
        
        # 2. Xác định d_model nếu chưa biết
        if self.d_model is None:
            self.d_model = self.backbone.config.d_model
        
        # 3. Freeze backbone nếu yêu cầu
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print(f"[MambaText] Backbone đã được đóng băng (freeze).")
        
        # 4. Projection head: d_model → embed_dim
        self.projection = nn.Linear(self.d_model, embed_dim)
        
        # 5. Layer norm trước projection (stabilize training)
        self.norm = nn.LayerNorm(self.d_model)
        
        print(f"[MambaText] Khởi tạo thành công: {model_name}")
        print(f"[MambaText]   d_model={self.d_model}, embed_dim={embed_dim}")
    
    def _load_model_and_tokenizer(self):
        """
        Tải Mamba model và tokenizer.
        Xử lý:
        1. config.json từ HF chứa key thừa → lọc trước khi tạo MambaConfig
        2. HF mới dùng safetensors thay vì pytorch_model.bin → hỗ trợ cả 2
        """
        try:
            from transformers import AutoTokenizer
            from dataclasses import fields as dataclass_fields
            from huggingface_hub import hf_hub_download
            
            print(f"[MambaText] Đang tải model '{self.hf_name}'...")
            
            # 1. Tải tokenizer — Mamba-130m dùng GPT-NeoX tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                "EleutherAI/gpt-neox-20b",
                trust_remote_code=True
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # 2. Tải config.json
            config_path = hf_hub_download(self.hf_name, "config.json")
            import json
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            # 3. Lọc bỏ key không thuộc MambaConfig dataclass
            from mamba_ssm.models.config_mamba import MambaConfig
            valid_keys = {f.name for f in dataclass_fields(MambaConfig)}
            filtered_config = {k: v for k, v in config_data.items() if k in valid_keys}
            removed_keys = set(config_data.keys()) - valid_keys
            if removed_keys:
                print(f"[MambaText] Bỏ qua config keys: {removed_keys}")
            
            # 4. Tạo model
            config = MambaConfig(**filtered_config)
            model = MambaLMHeadModel(config, device="cpu", dtype=torch.float32)
            
            # 5. Tải weights — thử safetensors trước, fallback sang pytorch_model.bin
            try:
                weights_path = hf_hub_download(self.hf_name, "model.safetensors")
                from safetensors.torch import load_file
                state_dict = load_file(weights_path, device="cpu")
                print(f"[MambaText] Tải weights từ safetensors thành công!")
            except Exception:
                try:
                    weights_path = hf_hub_download(self.hf_name, "pytorch_model.bin")
                    state_dict = torch.load(weights_path, map_location="cpu")
                    print(f"[MambaText] Tải weights từ pytorch_model.bin thành công!")
                except Exception as e2:
                    raise RuntimeError(f"Không tìm thấy weights (safetensors hoặc bin): {e2}")
            
            # Đổi tên key cho khớp với local MambaLMHeadModel
            if "backbone.embeddings.weight" in state_dict:
                state_dict["backbone.embedding.weight"] = state_dict.pop("backbone.embeddings.weight")
            
            # Dùng strict=False vì lm_head.weight được tied với embedding, có thể thiếu trong state_dict
            model.load_state_dict(state_dict, strict=False)
            
            print(f"[MambaText] Tải model thành công!")
            return model, tokenizer
            
        except ImportError as e:
            raise ImportError(
                f"Cần cài đặt thư viện: {e}\n"
                "Chạy: pip install transformers huggingface_hub safetensors"
            )
        except Exception as e:
            raise RuntimeError(f"Không thể tải model Mamba '{self.hf_name}': {e}")
    
    def _get_hidden_states(self, input_ids, attention_mask=None):
        """
        Chạy backbone Mamba và trả về hidden states của layer cuối.
        
        Args:
            input_ids (Tensor): Token IDs, shape (B, L)
            attention_mask (Tensor, optional): Mask, shape (B, L). 1=valid, 0=padding
            
        Returns:
            hidden_states (Tensor): shape (B, L, d_model)
        """
        # Lấy hidden states từ local mamba_ssm
        # MambaLMHeadModel trả về CausalLMOutput(logits=lm_logits), ta cần trích xuất hidden states.
        # Hoặc dùng trực tiếp model.backbone
        
        hidden_states = self.backbone.backbone(input_ids, inference_params=None)
        
        return hidden_states
    
    def _last_token_pooling(self, hidden_states, attention_mask):
        """
        Last-token pooling theo CLIMP.
        
        Theo bài báo: "We extract the hidden state at the last non-padding token 
        as the text representation"
        
        Lý do: Mamba là mô hình causal (autoregressive), mỗi h_t chỉ chứa
        thông tin từ t1 đến t_t. Do đó, token cuối cùng (non-padding) là vị trí
        duy nhất có TOÀN BỘ ngữ cảnh của câu.
        
        Args:
            hidden_states (Tensor): shape (B, L, d_model)
            attention_mask (Tensor): shape (B, L), 1=valid token, 0=padding
            
        Returns:
            pooled (Tensor): shape (B, d_model)
        """
        if attention_mask is None:
            # Không có mask → lấy token cuối cùng
            return hidden_states[:, -1, :]
        
        # Tìm vị trí token cuối cùng (non-padding) cho mỗi sample trong batch
        # attention_mask.sum(1) - 1 = index của token cuối
        sequence_lengths = attention_mask.sum(dim=1) - 1  # (B,)
        sequence_lengths = sequence_lengths.long()
        
        # Gather hidden state tại vị trí cuối
        batch_size = hidden_states.shape[0]
        pooled = hidden_states[torch.arange(batch_size, device=hidden_states.device), sequence_lengths]
        
        return pooled
    
    def tokenize(self, texts, max_length=None, device=None):
        """
        Tokenize danh sách câu văn bản thành input_ids và attention_mask.
        
        Args:
            texts (list[str]): Danh sách câu prompt
            max_length (int): Độ dài tối đa để padding/batching. Nếu None sẽ lấy theo câu dài nhất.
            device: Device đích cho tensor output
            
        Returns:
            dict với 'input_ids' (B, L) và 'attention_mask' (B, L)
        """
        tokens = self.tokenizer(
            texts,
            padding='max_length' if max_length else True,
            truncation=True if max_length else False,
            max_length=max_length,
            return_tensors='pt'
        )
        
        if device is not None:
            tokens = {k: v.to(device) for k, v in tokens.items()}
        
        return tokens
    
    def forward(self, input_ids=None, attention_mask=None, texts=None, max_length=None):
        """
        Forward pass: Text → Text Embedding
        
        Có thể truyền vào input_ids + attention_mask (đã tokenize sẵn)
        hoặc truyền texts (list[str]) để tự động tokenize.
        
        Args:
            input_ids (Tensor, optional): Token IDs, shape (B, L)
            attention_mask (Tensor, optional): Mask, shape (B, L)
            texts (list[str], optional): Danh sách câu văn bản
            max_length (int): Độ dài tối đa khi tự tokenize. (Không giới hạn nếu truyền None)
            
        Returns:
            text_features (Tensor): Text embedding, shape (B, embed_dim)
        """
        # Tự động tokenize nếu truyền texts
        if texts is not None:
            device = next(self.parameters()).device
            tokens = self.tokenize(texts, max_length=max_length, device=device)
            input_ids = tokens['input_ids']
            attention_mask = tokens['attention_mask']
        
        assert input_ids is not None, "Phải truyền input_ids hoặc texts"
        
        # 1. Lấy hidden states từ Mamba backbone
        hidden_states = self._get_hidden_states(input_ids, attention_mask)
        
        # 2. Last-token pooling → (B, d_model)
        pooled = self._last_token_pooling(hidden_states, attention_mask)
        
        # 3. Layer Norm
        pooled = self.norm(pooled.float())  # Đảm bảo float32
        
        # 4. Projection → shared embedding space
        text_features = self.projection(pooled)
        
        return text_features
        
    def encode_from_embeddings(self, inputs_embeds, attention_mask=None):
        """
        Forward pass sử dụng trực tiếp embeddings đã được tổng hợp (PromptGenerator).
        Args:
            inputs_embeds (Tensor): (B, L, d_model)
            attention_mask (Tensor, optional): (B, L)
        """
        # [QUAN TRỌNG] Tắt AMP autocast và ép float32 cho toàn bộ Mamba backbone
        # Mamba SSM có các phép tính recurrent (torch.exp, tích lũy trạng thái h)
        # rất dễ tràn số NaN/Inf khi chạy ở FP16.
        with torch.amp.autocast('cuda', enabled=False):
            inputs_embeds = inputs_embeds.float()
            
            mixer_model = self.backbone.backbone
            hidden_states = inputs_embeds
            residual = None
            
            for layer in mixer_model.layers:
                hidden_states, residual = layer(hidden_states, residual, inference_params=None)
                
            if not mixer_model.fused_add_norm:
                residual = (hidden_states + residual) if residual is not None else hidden_states
                hidden_states = mixer_model.norm_f(residual.to(dtype=mixer_model.norm_f.weight.dtype))
            else:
                import mamba_ssm.models.mixer_seq_simple as mixer_utils
                hidden_states = mixer_utils.layer_norm_fn(
                    hidden_states,
                    mixer_model.norm_f.weight,
                    mixer_model.norm_f.bias,
                    eps=mixer_model.norm_f.eps,
                    residual=residual,
                    prenorm=False,
                    residual_in_fp32=mixer_model.residual_in_fp32,
                    is_rms_norm=isinstance(mixer_model.norm_f, mixer_utils.RMSNorm)
                )
            
            pooled = self._last_token_pooling(hidden_states, attention_mask)
            pooled = self.norm(pooled.float())
            text_features = self.projection(pooled)
        
        return text_features
    
    
    def encode_prompts(self, prompt_list, max_length=None, batch_size=32):
        """
        Mã hóa danh sách prompt thành embeddings (hỗ trợ batch lớn).
        Hữu ích cho việc cache text features của tất cả thuộc tính.
        
        Args:
            prompt_list (list[str]): Danh sách tất cả prompts
            max_length (int): Độ dài padding batching. None là tự động padding.
            batch_size (int): Batch size khi encode
            
        Returns:
            all_features (Tensor): shape (num_prompts, embed_dim)
        """
        all_features = []
        
        for i in range(0, len(prompt_list), batch_size):
            batch_texts = prompt_list[i:i + batch_size]
            
            with torch.no_grad():
                features = self.forward(texts=batch_texts, max_length=max_length)
                all_features.append(features)
        
        return torch.cat(all_features, dim=0)
    
    def get_d_model(self):
        """Trả về chiều hidden state của backbone Mamba."""
        return self.d_model
    
    def get_embed_dim(self):
        """Trả về chiều embedding sau projection."""
        return self.embed_dim
        
    def get_word_embeddings(self, input_ids):
        """
        Lấy word embeddings từ layer embedding của Mamba.
        """
        # Tuỳ thuộc vào phiên bản Mamba, tên layer có thể là embedding hoặc embeddings
        if hasattr(self.backbone.backbone, 'embedding'):
            return self.backbone.backbone.embedding(input_ids)
        elif hasattr(self.backbone.backbone, 'embeddings'):
            return self.backbone.backbone.embeddings(input_ids)
        else:
            raise AttributeError("Không tìm thấy layer embedding trong Mamba backbone.")


# ============================================================================
# Factory function
# ============================================================================
def create_text_encoder(model_name='mamba-130m', embed_dim=768, **kwargs):
    """
    Factory function tạo Mamba Text Encoder.
    
    Args:
        model_name: Tên model ('mamba-130m', 'mamba-370m', 'mamba2-130m', ...)
        embed_dim: Chiều embedding đầu ra
        
    Returns:
        MambaTextEncoder instance
    """
    return MambaTextEncoder(
        model_name=model_name,
        embed_dim=embed_dim,
        **kwargs
    )
