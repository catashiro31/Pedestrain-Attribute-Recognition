import os
import sys
# Thêm đường dẫn project vào sys.path để có thể import từ config, model, dataset
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import umap
import matplotlib.pyplot as plt

from config import get_kaggle_config
from model.climp_par import CLIMPPAR
from dataset.clip_dataset import PARDataset, get_clip_transforms

def extract_features(dataloader, model, device, max_batches=10):
    """
    Trích xuất đặc trưng từ model.
    """
    model.eval()
    all_attr_embs = []
    all_style_embs = []
    all_disentangled = []
    all_labels = []
    
    # Chúng ta có thể trích xuất embedding của một thuộc tính cụ thể, ví dụ backpack
    attr_idx = 9 

    with torch.no_grad(), torch.amp.autocast('cuda'):
        for i, (images, bg_images, labels, _) in enumerate(tqdm(dataloader, desc="Extracting")):
            if i >= max_batches:
                break
            images = images.to(device, non_blocking=True)
            bg_images = bg_images.to(device, non_blocking=True)
            
            logits, features = model(images, bg_images, return_features=True)
            disentangled, style_emb, bg_context = features
            
            # Tính lại Attr_Embs (trước khi trừ nhiễu)
            attr_embs = disentangled + style_emb.unsqueeze(1)
            
            all_attr_embs.append(attr_embs[:, attr_idx, :].cpu().numpy())
            all_style_embs.append(style_emb.cpu().numpy())
            all_disentangled.append(disentangled[:, attr_idx, :].cpu().numpy())
            all_labels.append(labels[:, attr_idx].cpu().numpy())
            
    return (
        np.concatenate(all_attr_embs, axis=0), 
        np.concatenate(all_style_embs, axis=0), 
        np.concatenate(all_disentangled, axis=0),
        np.concatenate(all_labels, axis=0)
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Đường dẫn tới file checkpoint')
    parser.add_argument('--attr_idx', type=int, default=9, help='Index của thuộc tính (Mặc định 9 là Backpack)')
    parser.add_argument('--output_dir', type=str, default='output', help='Thư mục lưu kết quả')
    args = parser.parse_args()
    
    # Tạo thư mục output nếu chưa có
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    
    config = get_kaggle_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Sử dụng thiết bị: {device}")
    
    _, test_transform = get_clip_transforms(config.img_height, config.img_width)
    
    # Load dataset train (Source Domain) và test (Target Domain)
    train_dataset = PARDataset(config.pkl_path, config.img_dir, config.bg_img_dir, split='train', transform=test_transform)
    test_dataset = PARDataset(config.pkl_path, config.img_dir, config.bg_img_dir, split='test', transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    
    # Tắt pretrain VMamba khi load checkpoint
    config.vmamba_pretrained = False
    model = CLIMPPAR(config, train_dataset.attributes)
    
    print(f"Đang tải trọng số từ: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    clean_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict)
    model = model.to(device)
    
    print("Trích xuất từ Source (Train)...")
    src_attr, src_style, src_dis, src_labels = extract_features(train_loader, model, device, max_batches=20)
    
    print("Trích xuất từ Target (Test)...")
    tgt_attr, tgt_style, tgt_dis, tgt_labels = extract_features(test_loader, model, device, max_batches=20)
    
    # Gộp lại và tạo label miền (Domain Label)
    all_attr = np.vstack([src_attr, tgt_attr])
    all_dis = np.vstack([src_dis, tgt_dis])
    domain_labels = np.array(['Source'] * len(src_attr) + ['Target'] * len(tgt_attr))
    
    print("Chạy UMAP cho Attr_Embs (Trước Disentanglement)...")
    reducer = umap.UMAP(n_components=2, random_state=42)
    emb_attr_2d = reducer.fit_transform(all_attr)
    
    print("Chạy UMAP cho Disentangled_Embs (Sau Disentanglement)...")
    emb_dis_2d = reducer.fit_transform(all_dis)
    
    # Vẽ biểu đồ
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    for domain, color in zip(['Source', 'Target'], ['blue', 'red']):
        idx = domain_labels == domain
        axes[0].scatter(emb_attr_2d[idx, 0], emb_attr_2d[idx, 1], c=color, label=domain, alpha=0.6, s=10)
        axes[1].scatter(emb_dis_2d[idx, 0], emb_dis_2d[idx, 1], c=color, label=domain, alpha=0.6, s=10)
        
    attr_name = train_dataset.attributes[args.attr_idx]
    
    axes[0].set_title(f'Attribute Embeddings (Before) - Thuộc tính: {attr_name}')
    axes[0].legend()
    axes[1].set_title(f'Disentangled Embeddings (After) - Thuộc tính: {attr_name}')
    axes[1].legend()
    
    out_path = os.path.join(out_dir, f'umap_attr_{args.attr_idx}.png')
    plt.savefig(out_path)
    print(f"Đã lưu biểu đồ vào {out_path}")

if __name__ == '__main__':
    main()
