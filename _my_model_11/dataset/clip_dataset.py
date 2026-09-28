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
        T.Resize((height, width)),
        T.RandomHorizontalFlip(p=0.5),
        
        # 2. Xoay ảnh nhẹ
        T.RandomApply([T.RandomRotation(10)], p=0.5),
        
        # 3. Giả lập điều kiện ánh sáng và camera hồng ngoại (ban đêm)
        T.RandomApply([T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)], p=0.6),
        T.RandomGrayscale(p=0.15),
        
        # 4. Giả lập camera mờ (out-net / low resolution)
        T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.2),
        
        T.ToTensor(),
        normalize,
        T.RandomErasing(p=0.3, scale=(0.02, 0.2), value='random')
    ])

    valid_transform = T.Compose([
        T.Resize((height, width)),
        T.ToTensor(),
        normalize
    ])

    return train_transform, valid_transform


class PARDataset(data.Dataset):
    """
    Lớp Dataset đọc dữ liệu thuộc tính người đi bộ tùy chỉnh cho CLIP.
    """
    def __init__(self, pkl_path, img_dir=None, bg_img_dir=None, split='train', transform=None, target_transform=None, cache_in_memory=False):
        """
        Tham số:
            pkl_path (str): Đường dẫn đến file .pkl chứa thông tin nhãn và cấu trúc dữ liệu.
            img_dir (str, optional): Đường dẫn thư mục chứa ảnh gốc.
            bg_img_dir (str, optional): Đường dẫn thư mục chứa ảnh background (inpainted).
            split (str): Tập dữ liệu phân chia ('train', 'val', 'test').
            transform (callable, optional): Các phép biến đổi ảnh (ví dụ: sinh từ get_clip_transforms).
            target_transform (callable, optional): Biến đổi nhãn.
            cache_in_memory (bool): Kích hoạt lưu toàn bộ ảnh trên RAM.
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
        self.bg_root_path = bg_img_dir if bg_img_dir is not None else ''
        
        self.transform = transform
        self.target_transform = target_transform

        # 5. Cấu hình RAM Caching
        self.cache_in_memory = cache_in_memory
        self.image_cache = {}
        self.bg_cache = {}
        if self.cache_in_memory:
            print(f"[{split.upper()} Dataset] Đã kích hoạt RAM Cache (sẽ nạp {self.img_num} ảnh vào RAM trong epoch đầu)")
            
        # 5.5 Tự động quét và lập chỉ mục ảnh background (nếu bị chia trong thư mục con)
        self.bg_file_index = {}
        if self.bg_root_path and os.path.exists(self.bg_root_path):
            print(f"[{split.upper()} Dataset] Đang quét toàn bộ ảnh background (bao gồm thư mục con) tại {self.bg_root_path}...")
            for root, dirs, files in os.walk(self.bg_root_path):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.bg_file_index[file] = os.path.join(root, file)
            print(f"[{split.upper()} Dataset] Tìm thấy {len(self.bg_file_index)} ảnh background sẵn sàng ghép nối.")

        # 6. Tự động sinh danh sách các câu prompt CLIP khẳng định và phủ định cho toàn bộ tập dữ liệu
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
        
        if self.cache_in_memory and imgname in self.image_cache:
            img_pil = self.image_cache[imgname]
            bg_img_pil = self.bg_cache[imgname]
        else:
            imgpath = os.path.join(self.root_path, imgname)
            
            if self.bg_root_path:
                basename = os.path.basename(imgname)
                # Tìm tên file trong từ điển đã quét, nếu không có thì thử ghép đường dẫn cơ bản
                bg_imgpath = self.bg_file_index.get(basename, os.path.join(self.bg_root_path, imgname))
            else:
                bg_imgpath = imgpath
            
            try:
                img_pil = Image.open(imgpath).convert("RGB")
            except FileNotFoundError:
                img_pil = Image.open(imgname).convert("RGB")
                
            try:
                bg_img_pil = Image.open(bg_imgpath).convert("RGB")
            except FileNotFoundError:
                bg_img_pil = img_pil.copy()
                
            if self.cache_in_memory:
                # Lưu vào System RAM (giữ bản copy để không bị override transform)
                self.image_cache[imgname] = img_pil.copy()
                self.bg_cache[imgname] = bg_img_pil.copy()

        if self.transform is not None:
            # Transform trên bản copy RAM hoặc load từ ổ cứng
            img_pil = self.transform(img_pil)
            bg_img_pil = self.transform(bg_img_pil)

        gt_label = gt_label.astype(np.float32)
        if self.target_transform is not None:
            gt_label = self.target_transform(gt_label)
            
        return img_pil, bg_img_pil, gt_label, imgname

    def __len__(self):
        return len(self.img_ids)

    def get_prompts(self):
        """
        Trả về cặp danh sách prompt (phủ định, khẳng định) của toàn bộ thuộc tính trong dataset.
        Giúp tối ưu hóa huấn luyện: mã hóa văn bản một lần ngoài vòng lặp (Text Features caching).
        """
        return self.neg_prompts, self.pos_prompts
