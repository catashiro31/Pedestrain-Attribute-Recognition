import torch
import torch.nn as nn
import torch.nn.functional as F

class HiddenWords(nn.Module):
    """
    Hidden Words: 12 Nhóm Learnable Prompt Tokens
    - Group 0: Dành riêng background (Style HW)
    - Group 1-11: Ánh xạ theo nhóm ngữ nghĩa của thuộc tính (Attribute HWs)
    - Mỗi nhóm gồm 4 tokens (chiều embed_dim = 768)
    """
    def __init__(self, embed_dim=768, num_groups=12, tokens_per_group=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_groups = num_groups
        self.tokens_per_group = tokens_per_group
        
        # 12 x 4 x 768
        self.tokens = nn.Parameter(torch.empty(num_groups, tokens_per_group, embed_dim))
        nn.init.normal_(self.tokens, std=0.02)
        
    def get_style_hw(self):
        """
        Lấy token cho Style Prompt (Group 0).
        Returns: (tokens_per_group, embed_dim)
        """
        tokens = self.tokens[0]
        return F.dropout(tokens, p=0.1, training=self.training)
        
    def get_attr_hw(self, group_idx):
        """
        Lấy token cho Attribute Prompt của nhóm group_idx (từ 1 đến 11).
        Returns: (tokens_per_group, embed_dim)
        """
        assert 1 <= group_idx < self.num_groups, f"group_idx={group_idx} không hợp lệ"
        tokens = self.tokens[group_idx]
        return F.dropout(tokens, p=0.1, training=self.training)
        
    def initialize_with_words(self, text_encoder, device):
        """
        Khởi tạo các Hidden Words (CoOp) bằng embedding của các từ có ý nghĩa
        để tăng tốc độ hội tụ thay vì random noise ban đầu.
        """
        # 12 nhóm ngữ nghĩa ứng với 11 Semantic Groups + 1 Background Group
        # Sử dụng các cụm từ ngữ cảnh (context phrases) mang tính dẫn dắt cho thuộc tính
        init_words = [
            ["a", "photo", "of", "a"],                                    # Group 0: Background
            ["the", "person", "is", "a"],                                 # Group 1: Gender
            ["the", "person", "age", "is"],                               # Group 2: Age
            ["the", "person", "body", "is"],                              # Group 3: Body Size
            ["the", "person", "viewed", "from"],                          # Group 4: Viewpoint
            ["the", "person", "has", "on"],                               # Group 5: Head
            ["the", "person", "wearing", "upper"],                        # Group 6: Upper Body
            ["the", "person", "wearing", "lower"],                        # Group 7: Lower Body
            ["the", "person", "wearing", "shoes"],                        # Group 8: Shoes
            ["the", "person", "carrying", "a"],                           # Group 9: Bag
            ["the", "person", "is", "doing"],                             # Group 10: Activity
            ["the", "person", "is", "in"]                                 # Group 11: Posture
        ]
        
        with torch.no_grad():
            # Xác định device thực tế của text_encoder (lúc này thường là CPU)
            enc_device = next(text_encoder.parameters()).device
            my_device = self.tokens.device
            
            for i, words in enumerate(init_words):
                # Đảm bảo đúng số lượng token
                words = (words * self.tokens_per_group)[:self.tokens_per_group]
                
                group_embeds = []
                for w in words:
                    tokens = text_encoder.tokenize([w], device=enc_device)
                    input_ids = tokens['input_ids']
                    # Lấy word embeddings (1, L, 768)
                    embeds = text_encoder.get_word_embeddings(input_ids)
                    
                    # Tránh lấy nhầm BOS/EOS token nếu có, ta lấy trung bình các sub-words của từ đó
                    word_embed = embeds.mean(dim=1).squeeze(0)
                    group_embeds.append(word_embed.to(my_device))
                
                group_embeds = torch.stack(group_embeds, dim=0) # (tokens_per_group, 768)
                self.tokens.data[i] = group_embeds.clone()
        print("[HiddenWords] Đã khởi tạo các tokens bằng word embeddings có ý nghĩa thay vì noise.")
