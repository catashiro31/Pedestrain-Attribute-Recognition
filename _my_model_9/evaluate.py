# evaluate.py
# Script đánh giá mô hình CLIMP-PAR trên tập Test
# Hỗ trợ tính toán và hiển thị đầy đủ các metrics: mA, Accuracy, Precision, Recall, F1

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Giả định chạy từ thư mục _my_model_1
from config import get_kaggle_config
from model import CLIMPPAR
from losses import CLIMPPARLoss
from dataset.clip_dataset import PARDataset, get_clip_transforms
from train import compute_metrics

def evaluate_model(model, dataloader, criterion, config, device):
    """Đánh giá mô hình trên tập test"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    print(f"Bắt đầu đánh giá trên {len(dataloader.dataset)} mẫu...")
    
    with torch.inference_mode():
        for images, bg_images, labels, _ in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device, non_blocking=True)
            bg_images = bg_images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Forward pass (sử dụng AMP nếu config cho phép)
            with torch.amp.autocast('cuda', enabled=config.use_amp, dtype=torch.bfloat16):
                logits, _ = model(images, bg_images)
                loss = criterion(logits, labels)
            
            total_loss += loss.item()
            # Áp dụng sigmoid để lấy xác suất
            all_preds.append(torch.sigmoid(logits).float().cpu().numpy())
            all_labels.append(labels.float().cpu().numpy())
            
    # Gộp kết quả
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    avg_loss = total_loss / max(len(dataloader), 1)
    metrics = compute_metrics(all_preds, all_labels)
    metrics['loss'] = avg_loss
    
    return metrics

def main(args):
    # Lấy config chuẩn
    config = get_kaggle_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Sử dụng thiết bị: {device}")
    
    # 1. Khởi tạo Dataset & DataLoader cho tập TEST
    _, test_transform = get_clip_transforms(config.img_height, config.img_width)
    
    # Nếu truyền path dataset thì ghi đè config
    pkl_path = args.pkl_path if args.pkl_path else config.pkl_path
    img_dir = args.img_dir if args.img_dir else config.img_dir
    
    bg_img_dir = args.bg_img_dir if args.bg_img_dir else config.bg_img_dir
    
    test_dataset = PARDataset(
        pkl_path=pkl_path,
        img_dir=img_dir,
        bg_img_dir=bg_img_dir,
        split='test',  # Đánh giá trên tập TEST
        transform=test_transform
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    print(f"Dataset Test: {len(test_dataset)} mẫu, {test_dataset.attr_num} thuộc tính")
    
    # 2. Khởi tạo Model
    print("Khởi tạo mô hình...")
    # Tắt pretrained vì ta sẽ load checkpoint
    config.vmamba_pretrained = False 
    model = CLIMPPAR(config, test_dataset.attributes)
    
    # 3. Load Checkpoint
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Không tìm thấy checkpoint tại {args.checkpoint}")
    
    print(f"Đang tải trọng số từ: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    
    # Checkpoint có thể lưu dạng DDP (thêm prefix 'module.') hoặc Single GPU
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    
    # Lọc bỏ prefix 'module.' nếu mô hình train bằng DDP nhưng test bằng Single GPU
    clean_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            clean_state_dict[k[7:]] = v
        else:
            clean_state_dict[k] = v
            
    model.load_state_dict(clean_state_dict)
    model = model.to(device)
    print("Tải trọng số thành công!")
    
    # Không cache Text Features (Prompt được sinh động từ BackgroundEncoder)
    
    # 5. Đánh giá
    criterion = CLIMPPARLoss(
        loss_type=config.loss_type,
        asl_gamma_neg=getattr(config, 'asl_gamma_neg', 4.0),
        asl_gamma_pos=getattr(config, 'asl_gamma_pos', 1.0),
        asl_clip=getattr(config, 'asl_clip', 0.05)
    )
    metrics = evaluate_model(model, test_loader, criterion, config, device)
    
    # 6. In kết quả tổng quan
    print(f"\n{'='*60}")
    print("KẾT QUẢ ĐÁNH GIÁ TỔNG QUAN TRÊN TẬP TEST")
    print(f"{'='*60}")
    print(f"  Test Loss:   {metrics['loss']:.4f}")
    print(f"  mA:          {metrics['mA']:.4f}")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  Precision:   {metrics['precision']:.4f}")
    print(f"  Recall:      {metrics['recall']:.4f}")
    print(f"  F1 Score:    {metrics['f1']:.4f}")
    
    # 7. In kết quả chi tiết từng thuộc tính
    print(f"\n{'='*80}")
    print(f"{'CHI TIẾT TỪNG THUỘC TÍNH (PER-ATTRIBUTE METRICS)':^80}")
    print(f"{'='*80}")
    print(f"{'Thuộc tính':<25} | {'mA':<8} | {'Acc':<8} | {'Prec':<8} | {'Rec':<8} | {'F1':<8}")
    print(f"{'-'*80}")
    
    per_attr = metrics.get('per_attribute', [])
    for idx, attr_name in enumerate(test_dataset.attributes):
        if idx < len(per_attr):
            m = per_attr[idx]
            print(f"{attr_name:<25} | {m['mA']:.4f}   | {m['accuracy']:.4f}   | {m['precision']:.4f}   | {m['recall']:.4f}   | {m['f1']:.4f}")
    print(f"{'='*80}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate CLIMP-PAR on Test Set')
    parser.add_argument('--checkpoint', type=str, required=True, help='Đường dẫn tới file checkpoint (.pth)')
    parser.add_argument('--pkl_path', type=str, default='', help='Đường dẫn dataset pkl (nếu muốn ghi đè)')
    parser.add_argument('--img_dir', type=str, default='', help='Đường dẫn ảnh (nếu muốn ghi đè)')
    parser.add_argument('--bg_img_dir', type=str, default='', help='Đường dẫn ảnh background (nếu muốn ghi đè)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size khi test (mặc định 64)')
    
    args = parser.parse_args()
    main(args)
