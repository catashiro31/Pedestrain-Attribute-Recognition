# predict.py
# Script dự đoán (inference) CLIMP-PAR v8
# Hỗ trợ: dự đoán đơn ảnh, batch ảnh, và đánh giá trên tập test

import os
import sys
import argparse
import json
import numpy as np
from PIL import Image

import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

# Thêm đường dẫn project
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(PROJECT_DIR)
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, PROJECT_DIR)

from config import CLIMPConfig, get_kaggle_config
from model import CLIMPPAR
from dataset.clip_dataset import PARDataset, get_clip_transforms


# ============================================================================
# Metrics (giống train.py)
# ============================================================================
def compute_metrics(predictions, labels, threshold=0.5):
    """Tính các metric cho PAR: mA, Accuracy, Precision, Recall, F1."""
    preds_binary = (predictions >= threshold).astype(np.float32)
    
    num_attrs = labels.shape[1]
    attr_accuracies = []
    for i in range(num_attrs):
        tp = ((preds_binary[:, i] == 1) & (labels[:, i] == 1)).sum()
        tn = ((preds_binary[:, i] == 0) & (labels[:, i] == 0)).sum()
        p = (labels[:, i] == 1).sum()
        n = (labels[:, i] == 0).sum()
        acc_pos = tp / max(p, 1)
        acc_neg = tn / max(n, 1)
        attr_accuracies.append((acc_pos + acc_neg) / 2.0)
    mA = np.mean(attr_accuracies)
    
    tp = (preds_binary * labels).sum(axis=1)
    fp = (preds_binary * (1 - labels)).sum(axis=1)
    fn = ((1 - preds_binary) * labels).sum(axis=1)
    
    precision = (tp / (tp + fp + 1e-8)).mean()
    recall = (tp / (tp + fn + 1e-8)).mean()
    f1 = (2 * precision * recall / (precision + recall + 1e-8))
    accuracy = (preds_binary == labels).mean()
    
    return {
        'mA': float(mA),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
    }


# ============================================================================
# Load Model
# ============================================================================
def load_model(checkpoint_path, config=None, device='cuda'):
    """
    Load mô hình từ checkpoint.
    
    Args:
        checkpoint_path: Đường dẫn đến file .pth
        config: CLIMPConfig. Nếu None, sẽ dùng config lưu trong checkpoint.
        device: 'cuda' hoặc 'cpu'
    
    Returns:
        model: CLIMPPAR đã load weights
        config: CLIMPConfig
        attribute_names: List[str] (nếu có trong checkpoint)
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    
    # Khôi phục config
    if config is None:
        if 'config' in ckpt and isinstance(ckpt['config'], dict):
            config = CLIMPConfig(**ckpt['config'])
        else:
            config = get_kaggle_config()
    
    # Khởi tạo model
    model = CLIMPPAR(config)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Thông tin checkpoint
    epoch = ckpt.get('epoch', '?')
    best_mA = ckpt.get('best_mA', '?')
    print(f"  Epoch: {epoch}, Best mA: {best_mA}")
    
    return model, config


# ============================================================================
# Predict single image
# ============================================================================
@torch.no_grad()
def predict_single_image(model, image_path, attribute_names, domain_tokens, 
                          transform, device='cuda', threshold=0.5):
    """
    Dự đoán thuộc tính cho 1 ảnh.
    
    Args:
        model: CLIMPPAR model (eval mode)
        image_path: Đường dẫn ảnh
        attribute_names: List[str] tên thuộc tính
        domain_tokens: (32, 768) domain token tensor
        transform: transform cho ảnh
        device: 'cuda' hoặc 'cpu'
        threshold: ngưỡng phân loại
    
    Returns:
        results: dict {attribute_name: (probability, predicted_label)}
    """
    # Load và transform ảnh
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)  # (1, 3, H, W)
    
    # Chuẩn bị domain tokens
    dt = domain_tokens.unsqueeze(0).to(device)  # (1, 32, 768)
    
    # Forward pass
    with autocast(enabled=True):
        logits, _ = model(img_tensor, dt, attribute_names)
    
    probs = torch.sigmoid(logits).cpu().numpy()[0]  # (num_attrs,)
    
    # Tạo kết quả
    results = {}
    for i, attr_name in enumerate(attribute_names):
        prob = float(probs[i])
        label = 1 if prob >= threshold else 0
        results[attr_name] = {
            'probability': round(prob, 4),
            'predicted': label
        }
    
    return results


# ============================================================================
# Evaluate on test set
# ============================================================================
@torch.no_grad()
def evaluate_test_set(model, config, device='cuda'):
    """
    Đánh giá mô hình trên tập test.
    
    Returns:
        metrics: dict chứa mA, accuracy, precision, recall, f1
    """
    _, val_transform = get_clip_transforms(config.img_height, config.img_width)
    
    test_dataset = PARDataset(
        pkl_path=config.pkl_path,
        img_dir=config.img_dir,
        split='test',
        transform=val_transform,
        domain_tokens_path=config.domain_tokens_path
    )
    
    attribute_names = test_dataset.attributes
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size * 2,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    print(f"\nEvaluating on test set ({len(test_dataset)} samples)...")
    
    all_preds = []
    all_labels = []
    
    for batch_idx, (images, labels, _, domain_tokens) in enumerate(test_loader):
        images = images.to(device)
        domain_tokens = domain_tokens.to(device)
        
        with autocast(enabled=config.use_amp):
            logits, _ = model(images, domain_tokens, attribute_names)
        
        all_preds.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(labels.numpy())
        
        if (batch_idx + 1) % 50 == 0:
            print(f"  Batch {batch_idx+1}/{len(test_loader)}")
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    metrics = compute_metrics(all_preds, all_labels)
    
    print(f"\n{'='*50}")
    print(f"Test Results:")
    print(f"  mA:        {metrics['mA']:.4f}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    print(f"{'='*50}")
    
    # Per-attribute accuracy
    preds_binary = (all_preds >= 0.5).astype(np.float32)
    print(f"\nPer-attribute accuracy:")
    for i, attr_name in enumerate(attribute_names):
        tp = ((preds_binary[:, i] == 1) & (all_labels[:, i] == 1)).sum()
        tn = ((preds_binary[:, i] == 0) & (all_labels[:, i] == 0)).sum()
        p = (all_labels[:, i] == 1).sum()
        n = (all_labels[:, i] == 0).sum()
        acc = (tp + tn) / (p + n)
        print(f"  {attr_name:30s}: {acc:.4f} (pos={int(p)}, neg={int(n)})")
    
    return metrics


# ============================================================================
# Batch predict from folder
# ============================================================================
@torch.no_grad()
def predict_folder(model, folder_path, attribute_names, domain_tokens,
                   transform, device='cuda', threshold=0.5, output_json=None):
    """
    Dự đoán thuộc tính cho tất cả ảnh trong folder.
    
    Args:
        folder_path: Đường dẫn thư mục chứa ảnh
        output_json: Nếu có, lưu kết quả ra file JSON
    
    Returns:
        all_results: dict {filename: {attr: {probability, predicted}}}
    """
    # Tìm tất cả ảnh
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    image_files = sorted([
        f for f in os.listdir(folder_path)
        if os.path.splitext(f)[1].lower() in valid_exts
    ])
    
    print(f"\nDự đoán cho {len(image_files)} ảnh trong {folder_path}")
    
    all_results = {}
    for idx, filename in enumerate(image_files):
        img_path = os.path.join(folder_path, filename)
        results = predict_single_image(
            model, img_path, attribute_names, domain_tokens,
            transform, device, threshold
        )
        all_results[filename] = results
        
        # In kết quả tóm tắt
        positive_attrs = [k for k, v in results.items() if v['predicted'] == 1]
        print(f"  [{idx+1}/{len(image_files)}] {filename}: {positive_attrs}")
    
    # Lưu ra JSON
    if output_json:
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\nKết quả đã lưu tại: {output_json}")
    
    return all_results


# ============================================================================
# Entry point
# ============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CLIMP-PAR v8 Prediction')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--mode', type=str, default='test', choices=['test', 'single', 'folder'],
                        help='Chế độ: test (đánh giá tập test), single (1 ảnh), folder (thư mục)')
    parser.add_argument('--image', type=str, default='', help='Đường dẫn ảnh (mode=single)')
    parser.add_argument('--folder', type=str, default='', help='Đường dẫn thư mục ảnh (mode=folder)')
    parser.add_argument('--output', type=str, default='', help='Đường dẫn file JSON kết quả')
    parser.add_argument('--pkl_path', type=str, default='', help='Path to dataset pkl')
    parser.add_argument('--img_dir', type=str, default='', help='Path to image directory')
    parser.add_argument('--domain_tokens_path', type=str, default='', help='Path to domain_tokens.pt')
    parser.add_argument('--threshold', type=float, default=0.5, help='Ngưỡng phân loại')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    args = parser.parse_args()
    
    # Load config và model
    config = get_kaggle_config()
    if args.pkl_path:
        config.pkl_path = args.pkl_path
    if args.img_dir:
        config.img_dir = args.img_dir
    if args.domain_tokens_path:
        config.domain_tokens_path = args.domain_tokens_path
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    model, config = load_model(args.checkpoint, config, device)
    
    if args.mode == 'test':
        # Đánh giá trên tập test
        evaluate_test_set(model, config, device)
    
    elif args.mode == 'single':
        # Dự đoán 1 ảnh
        assert args.image, "Phải cung cấp --image khi dùng mode=single"
        
        _, val_transform = get_clip_transforms(config.img_height, config.img_width)
        
        # Load attribute names từ dataset
        test_dataset = PARDataset(
            pkl_path=config.pkl_path,
            img_dir=config.img_dir,
            split='test',
            transform=val_transform,
            domain_tokens_path=config.domain_tokens_path
        )
        attribute_names = test_dataset.attributes
        
        # Load domain tokens
        default_domain_token = torch.zeros(32, 768)
        if config.domain_tokens_path and os.path.exists(config.domain_tokens_path):
            dt = torch.load(config.domain_tokens_path, map_location='cpu')
            # Dùng token đầu tiên tìm được
            for key in dt:
                if key != '_metadata' and 'mean' in dt[key]:
                    default_domain_token = dt[key]['mean']
                    break
        
        results = predict_single_image(
            model, args.image, attribute_names, default_domain_token,
            val_transform, device, args.threshold
        )
        
        print(f"\nKết quả cho {args.image}:")
        print(f"{'='*50}")
        for attr_name, info in results.items():
            marker = "✓" if info['predicted'] == 1 else "✗"
            print(f"  {marker} {attr_name:30s}: {info['probability']:.4f}")
    
    elif args.mode == 'folder':
        # Dự đoán cho thư mục ảnh
        assert args.folder, "Phải cung cấp --folder khi dùng mode=folder"
        
        _, val_transform = get_clip_transforms(config.img_height, config.img_width)
        
        test_dataset = PARDataset(
            pkl_path=config.pkl_path,
            img_dir=config.img_dir,
            split='test',
            transform=val_transform,
            domain_tokens_path=config.domain_tokens_path
        )
        attribute_names = test_dataset.attributes
        
        # Load domain tokens
        default_domain_token = torch.zeros(32, 768)
        if config.domain_tokens_path and os.path.exists(config.domain_tokens_path):
            dt = torch.load(config.domain_tokens_path, map_location='cpu')
            for key in dt:
                if key != '_metadata' and 'mean' in dt[key]:
                    default_domain_token = dt[key]['mean']
                    break
        
        output_json = args.output if args.output else os.path.join(
            config.output_dir, 'predictions.json'
        )
        
        predict_folder(
            model, args.folder, attribute_names, default_domain_token,
            val_transform, device, args.threshold, output_json
        )
