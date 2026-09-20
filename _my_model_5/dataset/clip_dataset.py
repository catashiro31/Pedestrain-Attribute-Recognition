# clip_dataset.py
# Cài đặt bộ đọc dữ liệu tùy chỉnh tối ưu cho mô hình CLIP

import os
import pickle
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms import functional as F

try:
    from .prompt_templates import get_dataset_prompts
except ImportError:
    from prompt_templates import get_dataset_prompts


class PadToFixedSize:
    """
    Thêm padding (viền đen) để đưa ảnh về kích thước cố định (ví dụ 448x448) 
    mà không làm thay đổi số pixel thực của ảnh bên trong (giữ nguyên tỷ lệ và kích thước thật).
    """
    def __init__(self, height, width, fill=0):
        self.height = height
        self.width = width
        self.fill = fill

    def __call__(self, img):
        w, h = img.size
        
        # Nếu ảnh to hơn khung, dùng thumbnail thu nhỏ (giữ nguyên tỉ lệ) để lọt vừa khung
        if w > self.width or h > self.height:
            img.thumbnail((self.width, self.height), Image.Resampling.LANCZOS)
            w, h = img.size

        pad_left = max(0, (self.width - w) // 2)
        pad_right = max(0, self.width - w - pad_left)
        pad_top = max(0, (self.height - h) // 2)
        pad_bottom = max(0, self.height - h - pad_top)
        
        # Chèn viền đen xung quanh ảnh
        return F.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=self.fill)


def get_clip_transforms(height=224, width=224):
    """
    Khởi tạo pipeline biến đổi ảnh chuẩn cho CLIP.
    Sử dụng mean và std mặc định của OpenAI CLIP.
    """
    # Mean và std chính xác của OpenAI CLIP
    clip_mean = (0.48145466, 0.4578275, 0.40821073)
    clip_std = (0.26862954, 0.26130258, 0.27577711)
    
    normalize = T.Normalize(mean=clip_mean, std=clip_std)
    
    train_transform = T.Compose([
        PadToFixedSize(height, width, fill=0),  # Tấm đệm về kích thước đích (ví dụ 448x448)
        T.Pad(10),                              # Pad thêm lề ngẫu nhiên
        T.RandomCrop((height, width)),          # Crop ngẫu nhiên về lại kích thước đích
        T.RandomHorizontalFlip(),               # Lật ngang ảnh ngẫu nhiên
        T.ToTensor(),                           # Chuyển thành Tensor
        normalize,                              # Chuẩn hóa theo phân phối CLIP
    ])

    valid_transform = T.Compose([
        PadToFixedSize(height, width, fill=0),
        T.ToTensor(),
        normalize
    ])

    return train_transform, valid_transform


class PARDataset(data.Dataset):
    """
    Lớp Dataset đọc dữ liệu thuộc tính người đi bộ tùy chỉnh cho CLIP.
    """
    def __init__(self, pkl_path, img_dir=None, split='train', transform=None, target_transform=None):
        """
        Tham số:
            pkl_path (str): Đường dẫn đến file .pkl chứa thông tin nhãn và cấu trúc dữ liệu.
            img_dir (str, optional): Đường dẫn thư mục chứa ảnh gốc. Nếu không truyền, sẽ tự động
                                     đọc từ trường 'root' lưu trong file pickle.
            split (str): Tập dữ liệu phân chia ('train', 'val', 'test').
            transform (callable, optional): Các phép biến đổi ảnh (ví dụ: sinh từ get_clip_transforms).
            target_transform (callable, optional): Biến đổi nhãn.
        """
        assert os.path.exists(pkl_path), f"File pickle nhãn không tồn tại tại: {pkl_path}"
        
        with open(pkl_path, 'rb') as f:
            try:
                # Một số file pkl lưu ở dạng nhị phân pickle chuẩn
                dataset_info = pickle.load(f)
            except Exception as e:
                # Fallback nếu cần dùng encoding Latin1
                f.seek(0)
                dataset_info = pickle.load(f, encoding='latin1')

        # 1. Trích xuất thuộc tính nhãn (hỗ trợ nhiều định dạng đặt tên trường trong OpenPAR)
        img_ids = dataset_info.image_name
        attr_labels = dataset_info.label

        # 2. Kiểm tra tập phân chia dữ liệu (split)
        assert split in dataset_info.partition.keys(), f"Phân chia '{split}' không tồn tại trong dataset"
        self.img_idx = dataset_info.partition[split]
        if isinstance(self.img_idx, list):
            self.img_idx = self.img_idx[0]
            
        self.img_num = self.img_idx.shape[0]
        self.img_ids = [img_ids[i] for i in self.img_idx]
        self.labels = attr_labels[self.img_idx]

        # 3. Trích xuất thông tin thuộc tính
        if hasattr(dataset_info, 'attributes'):
            self.attributes = dataset_info.attributes
        elif hasattr(dataset_info, 'attr_name'):
            self.attributes = dataset_info.attr_name
        elif hasattr(dataset_info, 'attr_words'):
            self.attributes = dataset_info.attr_words
        else:
            raise AttributeError("Không tìm thấy thuộc tính chứa tên nhãn trong file pickle.")
            
        self.attr_num = len(self.attributes)

        # 4. Xác định thư mục chứa ảnh
        self.root_path = img_dir if img_dir is not None else getattr(dataset_info, 'root', '')
        
        self.transform = transform
        self.target_transform = target_transform

        # 5. Tự động sinh danh sách các câu prompt CLIP khẳng định và phủ định cho toàn bộ tập dữ liệu
        self.neg_prompts, self.pos_prompts = get_dataset_prompts(self.attributes)

    def __getitem__(self, index):
        """
        Trả về:
            img_tensor (Tensor): Ảnh đã qua tiền xử lý chuẩn của CLIP.
            gt_label (Tensor): Nhãn nhị phân dạng float32 (0 hoặc 1 cho từng thuộc tính).
            imgname (str): Tên file ảnh gốc.
        """
        imgname = self.img_ids[index]
        gt_label = self.labels[index]
        
        imgpath = os.path.join(self.root_path, imgname)
        
        # Load ảnh dưới dạng RGB
        try:
            img_pil = Image.open(imgpath).convert("RGB")
        except FileNotFoundError:
            # Fallback nếu đường dẫn ảnh bị lưu dạng tuyệt đối trong file pkl và không khớp cục bộ
            img_pil = Image.open(imgname).convert("RGB")
            
        if self.transform is not None:
            img_pil = self.transform(img_pil)

        gt_label = gt_label.astype(np.float32)
        if self.target_transform is not None:
            gt_label = self.target_transform(gt_label)
            
        return img_pil, gt_label, imgname

    def __len__(self):
        return len(self.img_ids)

    def get_prompts(self):
        """
        Trả về cặp danh sách prompt (phủ định, khẳng định) của toàn bộ thuộc tính trong dataset.
        Giúp tối ưu hóa huấn luyện: mã hóa văn bản một lần ngoài vòng lặp (Text Features caching).
        """
        return self.neg_prompts, self.pos_prompts
