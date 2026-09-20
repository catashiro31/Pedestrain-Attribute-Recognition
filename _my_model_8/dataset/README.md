# Hướng dẫn Sử dụng Bộ đọc Dữ liệu CLIP-PAR

Thư mục này chứa mã nguồn tùy biến dùng để nạp và tiền xử lý các tập dữ liệu nhận dạng thuộc tính người đi bộ (PETA, PA-100K, RAP,...) tối ưu riêng cho mô hình **CLIP**.

## 1. Cấu trúc Thư mục Đề xuất

Để bộ nạp hoạt động chính xác, bạn nên sắp xếp cấu trúc các tập dữ liệu theo định dạng sau:

```
dataset/
├── PETA/
│   ├── images/              # Thư mục chứa tất cả ảnh gốc của PETA (.png/.jpg)
│   └── dataset.pkl          # File pickle chứa thông tin nhãn gốc
├── PA100k/
│   ├── images/              # Thư mục chứa ảnh gốc của PA-100k
│   └── dataset.pkl          # File pickle chứa thông tin nhãn gốc
├── clip_dataset.py          # Lớp CLIPPARDataset chính
├── prompt_templates.py      # Bản đồ cấu hình prompt của từng thuộc tính
├── test_dataset.py          # Script kiểm thử tự động
└── README.md                # Tài liệu hướng dẫn này
```

---

## 2. Cách Sử dụng trong Mã nguồn (PyTorch)

Dưới đây là ví dụ nhanh cách tích hợp `CLIPPARDataset` vào vòng lặp huấn luyện PyTorch:

```python
import torch
from torch.utils.data import DataLoader
from dataset.clip_dataset import CLIPPARDataset, get_clip_transforms

# 1. Khởi tạo Transforms chuẩn hóa của CLIP (mặc định kích thước 224x224)
train_transform, val_transform = get_clip_transforms(height=224, width=224)

# 2. Khởi tạo Dataset
train_dataset = CLIPPARDataset(
    pkl_path="dataset/PETA/dataset.pkl",
    img_dir="dataset/PETA/images",  # Hoặc để None nếu đường dẫn ảnh lưu đúng trong file pickle
    split="train",
    transform=train_transform
)

# 3. Khởi tạo DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

# 4. Trích xuất danh sách Prompts của các thuộc tính
# Trả về: (list_negative_prompts, list_positive_prompts)
neg_prompts, pos_prompts = train_dataset.get_prompts()

# Khuyên nghị: Mã hóa prompt thành text tokens của CLIP một lần trước khi huấn luyện (caching)
import clip
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _ = clip.load("ViT-B/32", device=device)

# Mã hóa câu prompt thành tensor token [attr_num, 77]
pos_tokens = clip.tokenize(pos_prompts).to(device)
neg_tokens = clip.tokenize(neg_prompts).to(device)

with torch.no_grad():
    # Tính toán sẵn đặc trưng văn bản của CLIP [attr_num, D]
    pos_text_features = model.encode_text(pos_tokens)
    neg_text_features = model.encode_text(neg_tokens)
    
    # Chuẩn hóa đặc trưng văn bản về độ dài đơn vị (unit vector)
    pos_text_features /= pos_text_features.norm(dim=-1, keepdim=True)
    neg_text_features /= neg_text_features.norm(dim=-1, keepdim=True)

# 5. Huấn luyện/Đánh giá
for images, labels, img_names in train_loader:
    images = images.to(device)
    labels = labels.to(device) # Shape: [batch_size, attr_num]
    
    # Trích xuất đặc trưng ảnh [batch_size, D]
    image_features = model.encode_image(images)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    
    # Tính độ tương đồng cosine với prompt khẳng định và phủ định
    # Tích vô hướng thu được shape: [batch_size, attr_num]
    logits_pos = image_features @ pos_text_features.T
    logits_neg = image_features @ neg_text_features.T
    
    # Dự đoán (ví dụ dùng Softmax hoặc Sigmoid phân loại thuộc tính)
    # ...
```

---

## 3. Các đặc điểm nâng cao của bộ nạp này

* **PadToSquare**: Người đi bộ có ảnh chụp đứng (tỷ lệ cao, hẹp). Phép biến đổi này tự động thêm viền đen vào hai bên để đưa ảnh về hình vuông trước khi resize. Việc này giúp ảnh không bị biến dạng tỷ lệ cơ thể, cải thiện độ chính xác phân loại của CLIP Visual Encoder.
* **CLIP Normalization**: Chuẩn hóa màu sắc sử dụng đúng tham số màu của CLIP (`mean=[0.4814, 0.4578, 0.4082]`, `std=[0.2686, 0.2613, 0.2758]`) thay vì ImageNet chuẩn.
* **Prompt Auto-Generation**: Hàm `get_prompts` tự sinh cặp câu miêu tả phủ định và khẳng định dựa vào tên thuộc tính của dataset (ví dụ: nhãn `backpack` sẽ ánh xạ sang `"a photo of a pedestrian carrying a backpack"` và phủ định là `"a photo of a pedestrian without a backpack"`). Bạn có thể cấu hình thêm hoặc ghi đè các prompt trong `prompt_templates.py`.
