# prompt_templates.py
# Cấu hình lưu trữ các mẫu prompt của thuộc tính người đi bộ phục vụ cho mô hình CLIP.

# Bản đồ ghi đè các prompt đặc thù giúp câu văn tự nhiên hơn
# Cấu trúc: 'attribute_name': (neg_prompt, pos_prompt)
ATTRIBUTE_PROMPT_MAP = {}

def generate_attribute_prompts(attribute_name: str):
    """
    Sinh cặp prompt (phủ định, khẳng định) tự động cho một thuộc tính.
    Nếu thuộc tính có trong bản đồ ghi đè ATTRIBUTE_PROMPT_MAP thì dùng bản đồ đó,
    ngược lại sinh theo mẫu mặc định.
    """
    # Làm sạch tên thuộc tính (chuyển sang chữ thường, xóa khoảng trắng thừa)
    clean_name = attribute_name.strip()
    
    # Tìm kiếm trong bản đồ ghi đè (không phân biệt hoa thường)
    for key, value in ATTRIBUTE_PROMPT_MAP.items():
        if key.lower() == clean_name.lower():
            return value
            
    # Mẫu mặc định nếu không khớp ghi đè
    neg_prompt = f"a photo of a pedestrian without {clean_name}"
    pos_prompt = f"a photo of a pedestrian with {clean_name}"
    
    return neg_prompt, pos_prompt

def get_dataset_prompts(attribute_list):
    """
    Nhận vào danh sách tên thuộc tính của dataset và sinh cặp prompt tương ứng.
    Trả về:
        neg_prompts: list câu prompt phủ định
        pos_prompts: list câu prompt khẳng định
    """
    neg_prompts = []
    pos_prompts = []
    
    for attr in attribute_list:
        neg, pos = generate_attribute_prompts(attr)
        neg_prompts.append(neg)
        pos_prompts.append(pos)
        
    return neg_prompts, pos_prompts
