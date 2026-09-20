#!/usr/bin/env python3
"""
generate_domain_tokens.py
=========================
Pipeline sinh Domain Tokens từ ảnh domain sử dụng VMamba-Small vision encoder.

Mỗi domain folder chứa các ảnh inpainted (background-only, không có người).
Script sẽ:
  1. Load VMamba-Small pretrained (ImageNet-1K)
  2. Đưa mỗi ảnh qua vision encoder → (32, 768) visual feature embeddings
  3. Tính mean & std (Welford's online algorithm) qua tất cả ảnh trong mỗi domain
  4. Lưu domain tokens vào file .pt

Sử dụng:
    python generate_domain_tokens.py [--batch_size 32] [--device cuda] [--resume]
    
Output:
    domain_tokens.pt - Dictionary chứa domain tokens cho 6 domains
"""

import sys
import os
import argparse
import time
import glob
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms import functional as TF

# ============================================================================
# Thêm project paths
# ============================================================================
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.join(_SCRIPT_DIR, '..')
sys.path.insert(0, _PROJECT_DIR)

from model.vision_encoder import VMambaVisionEncoder

# ============================================================================
# Đường dẫn mặc định
# ============================================================================
DEFAULT_DOMAIN_DIR = os.path.join(
    _PROJECT_DIR, '..', 'domain_noper_images_inpainted_refined'
)
DEFAULT_OUTPUT_PATH = os.path.join(_SCRIPT_DIR, 'domain_tokens.pt')
DEFAULT_CHECKPOINT_PATH = os.path.join(_SCRIPT_DIR, '_progress.pt')

# Danh sách domains (tên folder)
DOMAIN_NAMES = [
    'Construction Site',
    'Kitchens',
    'Market',
    'Outdoors',
    'School',
    'Ski Resort',
]

# Kích thước ảnh
IMG_HEIGHT = 448
IMG_WIDTH = 448


# ============================================================================
# PadToFixedSize - Giống hệt trong clip_dataset.py
# ============================================================================
class PadToFixedSize:
    """
    Thêm padding (viền đen) để đưa ảnh về kích thước cố định
    mà không làm thay đổi tỷ lệ ảnh bên trong.
    """
    def __init__(self, height, width, fill=0):
        self.height = height
        self.width = width
        self.fill = fill

    def __call__(self, img):
        w, h = img.size

        # Nếu ảnh to hơn khung, dùng thumbnail thu nhỏ (giữ nguyên tỉ lệ)
        if w > self.width or h > self.height:
            img.thumbnail((self.width, self.height), Image.Resampling.LANCZOS)
            w, h = img.size

        pad_left = max(0, (self.width - w) // 2)
        pad_right = max(0, self.width - w - pad_left)
        pad_top = max(0, (self.height - h) // 2)
        pad_bottom = max(0, self.height - h - pad_top)

        return TF.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=self.fill)


# ============================================================================
# Dataset cho Domain Images
# ============================================================================
class DomainImageDataset(Dataset):
    """Dataset đơn giản đọc tất cả ảnh trong 1 folder."""

    SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform

        # Liệt kê tất cả ảnh trong folder
        self.image_paths = []
        for fname in sorted(os.listdir(folder_path)):
            ext = os.path.splitext(fname)[1].lower()
            if ext in self.SUPPORTED_EXTENSIONS:
                self.image_paths.append(os.path.join(folder_path, fname))

        print(f"  Tìm thấy {len(self.image_paths)} ảnh trong {os.path.basename(folder_path)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            # Trả về ảnh đen nếu không đọc được
            print(f"  [WARN] Không đọc được ảnh: {img_path}: {e}")
            img = Image.new('RGB', (IMG_WIDTH, IMG_HEIGHT), (0, 0, 0))

        if self.transform is not None:
            img = self.transform(img)

        return img


# ============================================================================
# Welford's Online Algorithm cho Mean & Variance
# ============================================================================
class WelfordAccumulator:
    """
    Tính mean và variance tăng dần (online) sử dụng Welford's algorithm.
    Không cần lưu trữ tất cả features vào RAM.
    
    Cho mỗi domain, tích lũy features shape (32, 768) từ từng ảnh.
    """

    def __init__(self):
        self.count = 0
        self.mean = None      # (32, 768)
        self.M2 = None        # (32, 768) - sum of squared differences

    def update_batch(self, batch_features):
        """
        Cập nhật với một batch features.
        
        Args:
            batch_features: Tensor shape (B, 32, 768)
        """
        for i in range(batch_features.shape[0]):
            x = batch_features[i]  # (32, 768)
            self._update_single(x)

    def _update_single(self, x):
        """Cập nhật với 1 sample, x: (32, 768)."""
        self.count += 1

        if self.mean is None:
            self.mean = x.clone()
            self.M2 = torch.zeros_like(x)
        else:
            delta = x - self.mean
            self.mean += delta / self.count
            delta2 = x - self.mean
            self.M2 += delta * delta2

    def get_mean(self):
        """Trả về mean: (32, 768)."""
        return self.mean

    def get_std(self):
        """Trả về std (population): (32, 768)."""
        if self.count < 2:
            return torch.zeros_like(self.mean)
        variance = self.M2 / self.count
        return torch.sqrt(variance + 1e-8)

    def get_count(self):
        return self.count

    def state_dict(self):
        """Serialization cho resume."""
        return {
            'count': self.count,
            'mean': self.mean,
            'M2': self.M2,
        }

    def load_state_dict(self, state):
        """Load từ checkpoint."""
        self.count = state['count']
        self.mean = state['mean']
        self.M2 = state['M2']


# ============================================================================
# Main Pipeline
# ============================================================================
def get_transform():
    """Transform cho domain images: PadToFixedSize(448, 448) + CLIP Normalize."""
    clip_mean = (0.48145466, 0.4578275, 0.40821073)
    clip_std = (0.26862954, 0.26130258, 0.27577711)

    return T.Compose([
        PadToFixedSize(IMG_HEIGHT, IMG_WIDTH, fill=0),
        T.ToTensor(),
        T.Normalize(mean=clip_mean, std=clip_std),
    ])


def create_encoder(device, pretrained_path=None):
    """Tạo VMamba-Small vision encoder."""
    print("=" * 60)
    print("Khởi tạo VMamba-Small Vision Encoder...")
    print("=" * 60)

    encoder = VMambaVisionEncoder(
        variant='small',
        embed_dim=768,
        pretrained=True,
        pretrained_path=pretrained_path,
    )
    encoder = encoder.to(device)
    encoder.eval()

    # Đếm tham số
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"Tổng tham số: {total_params / 1e6:.1f}M")
    print(f"Device: {device}")
    print()

    return encoder


@torch.no_grad()
def extract_domain_features(encoder, domain_folder, transform, device,
                             batch_size=32, num_workers=4,
                             accumulator=None, start_idx=0):
    """
    Trích xuất features cho 1 domain và tính mean/std online.
    
    Args:
        encoder: VMamba-Small encoder
        domain_folder: Đường dẫn folder ảnh
        transform: Image transform pipeline
        device: torch device
        batch_size: Batch size
        num_workers: Số worker cho DataLoader
        accumulator: WelfordAccumulator đã có (cho resume)
        start_idx: Index bắt đầu (cho resume)
    
    Returns:
        accumulator: WelfordAccumulator chứa mean/std
    """
    dataset = DomainImageDataset(domain_folder, transform=transform)

    if len(dataset) == 0:
        print(f"  [WARN] Không tìm thấy ảnh trong {domain_folder}")
        return accumulator

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    if accumulator is None:
        accumulator = WelfordAccumulator()

    total_images = len(dataset)
    processed = start_idx
    start_time = time.time()

    try:
        # Dùng tqdm nếu có, fallback nếu không
        from tqdm import tqdm
        pbar = tqdm(dataloader, desc=f"  Processing", unit="batch")
    except ImportError:
        pbar = dataloader
        print(f"  (tqdm không có sẵn, hiển thị progress thủ công)")

    batch_count = 0
    for batch_imgs in pbar:
        # Skip các batch đã xử lý rồi (resume)
        batch_start = batch_count * batch_size
        batch_count += 1
        if batch_start < start_idx:
            continue

        batch_imgs = batch_imgs.to(device, non_blocking=True)

        # Mixed precision nếu có CUDA
        if device.type == 'cuda':
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                features = encoder(batch_imgs)  # (B, 32, 768)
            features = features.float()  # Chuyển về float32 cho tính toán mean/std
        else:
            features = encoder(batch_imgs)  # (B, 32, 768)

        # Chuyển về CPU để tích lũy (tiết kiệm VRAM)
        features = features.cpu()
        accumulator.update_batch(features)

        processed += batch_imgs.shape[0]

        # Cập nhật progress
        if hasattr(pbar, 'set_postfix'):
            elapsed = time.time() - start_time
            speed = processed / max(elapsed, 1)
            pbar.set_postfix({
                'done': f"{processed}/{total_images}",
                'speed': f"{speed:.1f} img/s"
            })
        elif processed % (batch_size * 10) == 0:
            elapsed = time.time() - start_time
            speed = processed / max(elapsed, 1)
            print(f"  Progress: {processed}/{total_images} ({speed:.1f} img/s)")

    elapsed = time.time() - start_time
    print(f"  Hoàn thành: {processed} ảnh trong {elapsed:.1f}s "
          f"({processed / max(elapsed, 1):.1f} img/s)")

    return accumulator


def main():
    parser = argparse.ArgumentParser(
        description='Generate domain tokens using VMamba-Small vision encoder'
    )
    parser.add_argument('--domain_dir', type=str, default=DEFAULT_DOMAIN_DIR,
                        help='Đường dẫn folder chứa các domain folders')
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT_PATH,
                        help='Đường dẫn file output .pt')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size cho DataLoader')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Số workers cho DataLoader')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (cuda/cpu/auto)')
    parser.add_argument('--pretrained_path', type=str, default=None,
                        help='Đường dẫn checkpoint VMamba-Small local')
    parser.add_argument('--resume', action='store_true',
                        help='Tiếp tục từ checkpoint nếu bị ngắt')

    args = parser.parse_args()

    # Xác định device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    # Kiểm tra domain_dir
    domain_dir = os.path.abspath(args.domain_dir)
    assert os.path.isdir(domain_dir), f"Domain directory không tồn tại: {domain_dir}"

    print("=" * 60)
    print("DOMAIN TOKEN GENERATION PIPELINE")
    print("=" * 60)
    print(f"Domain dir  : {domain_dir}")
    print(f"Output      : {args.output}")
    print(f"Batch size  : {args.batch_size}")
    print(f"Device      : {device}")
    print(f"Image size  : {IMG_HEIGHT}x{IMG_WIDTH}")
    print(f"Resume      : {args.resume}")
    print()

    # Load checkpoint nếu resume
    progress = {}
    if args.resume and os.path.exists(DEFAULT_CHECKPOINT_PATH):
        print("Đang load checkpoint tiến trình...")
        progress = torch.load(DEFAULT_CHECKPOINT_PATH, map_location='cpu', weights_only=False)
        print(f"  Đã xử lý: {list(progress.get('completed_domains', []))}")
        print()

    # Tạo encoder
    encoder = create_encoder(device, pretrained_path=args.pretrained_path)

    # Transform
    transform = get_transform()

    # Kết quả
    domain_tokens = progress.get('domain_tokens', OrderedDict())

    # Duyệt từng domain
    total_start = time.time()
    completed_domains = progress.get('completed_domains', [])

    for domain_name in DOMAIN_NAMES:
        # Skip nếu đã xử lý (resume)
        if domain_name in completed_domains:
            print(f"[SKIP] {domain_name} (đã xử lý)")
            continue

        domain_folder = os.path.join(domain_dir, domain_name)
        if not os.path.isdir(domain_folder):
            print(f"[WARN] Folder không tồn tại: {domain_folder}")
            continue

        print(f"\n{'=' * 60}")
        print(f"Domain: {domain_name}")
        print(f"{'=' * 60}")

        # Trích xuất features và tính mean/std
        accumulator = extract_domain_features(
            encoder=encoder,
            domain_folder=domain_folder,
            transform=transform,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        if accumulator is None or accumulator.get_count() == 0:
            print(f"  [WARN] Không có features cho {domain_name}")
            continue

        # Lưu domain token
        mean = accumulator.get_mean()  # (32, 768)
        std = accumulator.get_std()    # (32, 768)
        count = accumulator.get_count()

        domain_tokens[domain_name] = {
            'mean': mean,   # (32, 768) - domain token chính
            'std': std,     # (32, 768) - độ lệch chuẩn
            'count': count, # số ảnh đã xử lý
        }

        print(f"\n  Domain Token cho '{domain_name}':")
        print(f"    Mean shape : {mean.shape}")
        print(f"    Std shape  : {std.shape}")
        print(f"    Count      : {count}")
        print(f"    Mean norm  : {mean.norm().item():.4f}")
        print(f"    Std  norm  : {std.norm().item():.4f}")

        # Lưu checkpoint tiến trình
        completed_domains.append(domain_name)
        checkpoint = {
            'domain_tokens': domain_tokens,
            'completed_domains': completed_domains,
        }
        torch.save(checkpoint, DEFAULT_CHECKPOINT_PATH)
        print(f"  Đã lưu checkpoint tiến trình.")

    # ========================================================================
    # Lưu kết quả cuối cùng
    # ========================================================================
    print(f"\n{'=' * 60}")
    print("KẾT QUẢ CUỐI CÙNG")
    print(f"{'=' * 60}")

    for name, tokens in domain_tokens.items():
        print(f"  {name:25s} | count={tokens['count']:>6d} | "
              f"mean_norm={tokens['mean'].norm().item():.4f} | "
              f"std_norm={tokens['std'].norm().item():.4f}")

    # Lưu file output
    output_data = OrderedDict()
    for name, tokens in domain_tokens.items():
        output_data[name] = {
            'mean': tokens['mean'],   # (32, 768)
            'std': tokens['std'],     # (32, 768)
            'count': tokens['count'],
        }

    # Thêm metadata
    output_data['_metadata'] = {
        'encoder': 'VMamba-Small',
        'embed_dim': 768,
        'num_tokens': 32,
        'image_size': (IMG_HEIGHT, IMG_WIDTH),
        'num_domains': len(domain_tokens),
        'domain_names': list(domain_tokens.keys()),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    torch.save(output_data, args.output)
    print(f"\nĐã lưu domain tokens tại: {args.output}")

    # Dọn dẹp checkpoint
    if os.path.exists(DEFAULT_CHECKPOINT_PATH):
        os.remove(DEFAULT_CHECKPOINT_PATH)
        print("Đã xóa file checkpoint tiến trình.")

    total_elapsed = time.time() - total_start
    print(f"\nTổng thời gian: {total_elapsed:.1f}s ({total_elapsed / 60:.1f} phút)")
    print("DONE!")


if __name__ == '__main__':
    main()
