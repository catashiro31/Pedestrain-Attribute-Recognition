import torch
import torch.nn as nn

class PromptGenerator(nn.Module):
    """
    Prompt Generator: Tạo chuỗi embeddings (thay vì tokens rời rạc) để đưa vào TextEncoder.
    - Style Prompt: [background4prompt] + [Style HW]
    - Attribute Prompts: [background4prompt] + [Attribute HW_k] + [Attribute Word Embeddings]
    """
    def __init__(self, hidden_words, text_encoder, attributes):
        super().__init__()
        self.hidden_words = hidden_words
        self.text_encoder = text_encoder
        self.attributes = attributes
        self.num_attrs = len(attributes)
        
        # Định nghĩa 11 nhóm theo Semantic Groups (Bảng phân chia)
        group_keywords = {
            1: ["female"],
            2: ["child", "adult", "elderly"],
            3: ["fat", "normal", "thin"],
            4: ["front", "back", "side"],
            5: ["bald", "long hair", "black hair", "hat", "glasses", "mask", "helmet", "scarf", "gloves"],
            6: ["short sleeves", "long sleeves", "shirt", "jacket", "suit", "vest", "cotton coat", "coat", "graduation gown", "chef uniform"],
            7: ["trousers", "shorts", "jeans", "long skirt", "short skirt", "dress"],
            8: ["leather shoes", "casual shoes", "boots", "sandals", "other shoes"],
            9: ["backpack", "shoulder bag", "hand bag", "plastic bag", "paper bag", "suitcase", "others", "bag"],
            10: ["calling", "smoking", "hands back", "arms crossed"],
            11: ["walking", "running", "standing", "bicycle", "scooter", "skateboard"]
        }
        
        # Ánh xạ thuộc tính vào 11 nhóm HW (từ group 1 đến 11)
        self.attr_to_hw_group = []
        for attr in attributes:
            attr_lower = attr.lower().strip()
            assigned_group = 1 # Mặc định
            
            found = False
            # Pass 1: Ưu tiên khớp chính xác (exact match)
            for group_id, keywords in group_keywords.items():
                if attr_lower in keywords:
                    assigned_group = group_id
                    found = True
                    break
                    
            # Pass 2: Nếu không khớp chính xác, khớp một phần (substring match)
            if not found:
                for group_id, keywords in group_keywords.items():
                    for kw in keywords:
                        if kw in attr_lower or attr_lower in kw:
                            assigned_group = group_id
                            found = True
                            break
                    if found:
                        break
                        
            self.attr_to_hw_group.append(assigned_group)
        
    def _get_attr_word_embeds(self, device):
        # Tokenize attributes
        tokens = self.text_encoder.tokenize(self.attributes, device=device)
        input_ids = tokens['input_ids']
        attention_mask = tokens['attention_mask']
        
        # Chuyển qua embedding layer một cách an toàn
        word_embeds = self.text_encoder.get_word_embeddings(input_ids) # (num_attrs, L, 768)
        
        return word_embeds, attention_mask
        
    def forward(self, bg_context):
        """
        bg_context: (B, 768) từ BackgroundEncoder
        
        Returns:
            style_embeds: (B, L_style, 768)
            style_mask: (B, L_style)
            attr_embeds: (B, num_attrs, L_attr, 768)
            attr_masks: (B, num_attrs, L_attr)
        """
        B = bg_context.size(0)
        device = bg_context.device
        
        bg_expanded = bg_context.unsqueeze(1) # (B, 1, 768)
        
        # --- 1. Tạo Style Prompt ---
        hw0 = self.hidden_words.get_style_hw() # (4, 768)
        hw0_expanded = hw0.unsqueeze(0).expand(B, -1, -1) # (B, 4, 768)
        
        style_embeds = torch.cat([bg_expanded, hw0_expanded], dim=1) # (B, 5, 768)
        style_mask = torch.ones((B, style_embeds.size(1)), dtype=torch.long, device=device)
        
        # --- 2. Tạo Attribute Prompts (Vectorized bằng Broadcasting) ---
        attr_word_embeds, attr_mask = self._get_attr_word_embeds(device) 
        # attr_word_embeds: (num_attrs, max_len, 768)
        # attr_mask: (num_attrs, max_len)
        
        max_len = attr_word_embeds.size(1)
        
        # 2.1 Chuẩn bị Background Context
        # (B, 1, 768) -> (B, num_attrs, 1, 768)
        bg_broadcast = bg_expanded.unsqueeze(1).expand(-1, self.num_attrs, -1, -1)
        
        # 2.2 Chuẩn bị Hidden Words cho từng thuộc tính
        # Lấy trước danh sách HW tương ứng với 57 thuộc tính -> (num_attrs, 4, 768)
        hw_list = [self.hidden_words.get_attr_hw(self.attr_to_hw_group[i]) for i in range(self.num_attrs)]
        hw_tensors = torch.stack(hw_list) # (num_attrs, 4, 768)
        # (num_attrs, 4, 768) -> (B, num_attrs, 4, 768)
        hw_broadcast = hw_tensors.unsqueeze(0).expand(B, -1, -1, -1)
        
        # 2.3 Chuẩn bị Word Embeddings
        # (num_attrs, max_len, 768) -> (B, num_attrs, max_len, 768)
        words_broadcast = attr_word_embeds.unsqueeze(0).expand(B, -1, -1, -1)
        
        # Gộp tất cả lại (Concat theo chiều sequence length: dim=2)
        attr_embeds = torch.cat([bg_broadcast, hw_broadcast, words_broadcast], dim=2) 
        # (B, num_attrs, 1 + 4 + max_len, 768)
        
        # 2.4 Xử lý Masks tương ứng
        # Background và HW luôn là token hợp lệ (không phải padding) -> mask = 1
        ones_bg_hw = torch.ones((B, self.num_attrs, 5), dtype=torch.long, device=device)
        
        # Mask của từ (num_attrs, max_len) -> (B, num_attrs, max_len)
        attr_mask_broadcast = attr_mask.unsqueeze(0).expand(B, -1, -1)
        
        attr_masks = torch.cat([ones_bg_hw, attr_mask_broadcast], dim=2) 
        # (B, num_attrs, 1 + 4 + max_len)
        
        return style_embeds, style_mask, attr_embeds, attr_masks
