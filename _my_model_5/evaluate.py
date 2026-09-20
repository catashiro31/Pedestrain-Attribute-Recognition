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

def evaluate_model(model, dataloader, criterion, config, text_features_cache, device):
    """Đánh giá mô hình trên tập test"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    print(f"Bắt đầu đánh giá trên {len(dataloader.dataset)} mẫu...")
    
    with torch.no_grad():
        for images, labels, _ in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass (sử dụng AMP nếu config cho phép)
            with torch.amp.autocast('cuda', enabled=config.use_amp):
                # Gọi forward của CLIMPPAR với text features đã cache
                logits, _ = model(images, cached_text_features=text_features_cache)
                loss = criterion(logits, labels)
            
            total_loss += loss.item()
            # Áp dụng sigmoid để lấy xác suất
            all_preds.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
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
    
    test_dataset = PARDataset(
        pkl_path=pkl_path,
        img_dir=img_dir,
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
    config.resnet_pretrained = False 
    model = CLIMPPAR(config)
    
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
    
    # 4. Cache Text Features (Tương tự train)
    print("Đang cache text features...")
    model.eval()
    neg_prompts, pos_prompts = test_dataset.get_prompts()
    
    with torch.no_grad():
        text_features_cache = model.text_encoder.encode_prompts(
            pos_prompts, batch_size=32
        )
    text_features_cache = text_features_cache.to(device)
    
    # 5. Đánh giá
    criterion = CLIMPPARLoss(loss_type=config.loss_type)
    metrics = evaluate_model(model, test_loader, criterion, config, text_features_cache, device)
    
    # 6. In kết quả
    print(f"\n{'='*40}")
    print("KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST")
    print(f"{'='*40}")
    print(f"  Test Loss:   {metrics['loss']:.4f}")
    print(f"  mA:          {metrics['mA']:.4f}")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  Precision:   {metrics['precision']:.4f}")
    print(f"  Recall:      {metrics['recall']:.4f}")
    print(f"  F1 Score:    {metrics['f1']:.4f}")
    print(f"{'='*40}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate CLIMP-PAR on Test Set')
    parser.add_argument('--checkpoint', type=str, required=True, help='Đường dẫn tới file checkpoint (.pth)')
    parser.add_argument('--pkl_path', type=str, default='', help='Đường dẫn dataset pkl (nếu muốn ghi đè)')
    parser.add_argument('--img_dir', type=str, default='', help='Đường dẫn ảnh (nếu muốn ghi đè)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size khi test (mặc định 64)')
    
    args = parser.parse_args()
    main(args)
