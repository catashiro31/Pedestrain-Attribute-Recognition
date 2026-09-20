# visualize.py
# Kịch bản (Script) trực quan hóa kết quả dự đoán của mô hình CLIMP-PAR
# Vẽ ảnh kèm theo danh sách nhãn Ground Truth và Dự đoán (xác suất)

import os
import argparse
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from config import get_kaggle_config
from model import CLIMPPAR
from dataset.clip_dataset import PARDataset, get_clip_transforms

def load_model(checkpoint_path, config, device):
    print("Khởi tạo mô hình...")
    config.vmamba_pretrained = False
    model = CLIMPPAR(config)
    
    print(f"Đang tải trọng số từ: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    
    clean_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            clean_state_dict[k[7:]] = v
        else:
            clean_state_dict[k] = v
            
    model.load_state_dict(clean_state_dict)
    model = model.to(device)
    model.eval()
    print("Tải trọng số thành công!")
    return model

def visualize_samples(model, dataset, raw_img_dir, config, device, num_samples=4, threshold=0.5, save_path=None):
    """
    Lấy ngẫu nhiên num_samples từ dataset, chạy inference và vẽ biểu đồ.
    """
    indices = random.sample(range(len(dataset)), num_samples)
    
    # 1. Chuẩn bị (Cache) Text features để inference nhanh
    print("Đang chuẩn bị text features...")
    neg_prompts, pos_prompts = dataset.get_prompts()
    with torch.no_grad():
        text_features_cache = model.text_encoder.encode_prompts(pos_prompts, batch_size=32)
    text_features_cache = text_features_cache.to(device)
    
    attribute_names = dataset.attributes
    
    fig, axes = plt.subplots(num_samples, 1, figsize=(10, 6 * num_samples))
    if num_samples == 1:
        axes = [axes]
        
    print(f"Bắt đầu dự đoán {num_samples} ảnh...")
    for idx, ax in zip(indices, axes):
        # Lấy dữ liệu
        img_tensor, gt_label, imgname = dataset[idx]
        
        # Đường dẫn ảnh gốc để vẽ
        img_path = os.path.join(raw_img_dir, imgname)
        try:
            raw_img = Image.open(img_path).convert("RGB")
        except:
            raw_img = Image.open(imgname).convert("RGB") # fallback
            
        # Suy luận (Inference)
        img_batch = img_tensor.unsqueeze(0).to(device) # (1, 3, H, W)
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=config.use_amp):
                logits, _ = model(img_batch, cached_text_features=text_features_cache)
                probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
                
        # Phân tích kết quả
        preds_binary = (probs >= threshold).astype(np.int32)
        gt_label = gt_label.astype(np.int32)
        
        # Lọc danh sách thuộc tính
        gt_attrs = []
        pred_attrs = []
        
        for i, attr in enumerate(attribute_names):
            if gt_label[i] == 1:
                gt_attrs.append(attr)
            
            if preds_binary[i] == 1:
                # Đánh giá đúng sai cho prediction
                status = "✔" if gt_label[i] == 1 else "✖ (FP)"
                pred_attrs.append(f"{attr}: {probs[i]:.1%} {status}")
                
        # Tìm những thuộc tính bị bỏ sót (False Negatives)
        missed_attrs = []
        for i, attr in enumerate(attribute_names):
            if gt_label[i] == 1 and preds_binary[i] == 0:
                missed_attrs.append(f"{attr}: {probs[i]:.1%} ✖ (FN)")
                
        # ---------------------
        # Vẽ biểu đồ
        # ---------------------
        ax.imshow(raw_img)
        ax.axis('off')
        
        # Định dạng text hiển thị
        gt_text = "Ground Truth:\n" + "\n".join(["- " + a for a in gt_attrs])
        
        pred_text = "Dự đoán (Probs >= 50%):\n" 
        pred_text += "\n".join(["- " + a for a in pred_attrs]) if pred_attrs else "Không có"
        
        if missed_attrs:
            pred_text += "\n\nBỏ sót (Missed):\n" + "\n".join(["- " + a for a in missed_attrs])
            
        # Đặt text bên phải bức ảnh
        ax.text(1.05, 0.95, gt_text, transform=ax.transAxes, fontsize=11, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
        
        ax.text(1.5, 0.95, pred_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        ax.set_title(f"Image: {imgname}", fontsize=12)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"\nĐã lưu ảnh trực quan hóa tại: {save_path}")
    else:
        plt.show()

def main(args):
    config = get_kaggle_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Dataset (Tập test)
    _, test_transform = get_clip_transforms(config.img_height, config.img_width)
    pkl_path = args.pkl_path if args.pkl_path else config.pkl_path
    img_dir = args.img_dir if args.img_dir else config.img_dir
    
    test_dataset = PARDataset(
        pkl_path=pkl_path,
        img_dir=img_dir,
        split='test',
        transform=test_transform
    )
    
    model = load_model(args.checkpoint, config, device)
    
    save_path = args.output if args.output else 'visualization.png'
    
    visualize_samples(
        model=model, 
        dataset=test_dataset, 
        raw_img_dir=img_dir, 
        config=config, 
        device=device,
        num_samples=args.num_samples,
        threshold=args.threshold,
        save_path=save_path
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize CLIMP-PAR Predictions')
    parser.add_argument('--checkpoint', type=str, required=True, help='Đường dẫn file .pth')
    parser.add_argument('--pkl_path', type=str, default='', help='Đường dẫn dataset .pkl')
    parser.add_argument('--img_dir', type=str, default='', help='Đường dẫn thư mục ảnh gốc')
    parser.add_argument('--num_samples', type=int, default=4, help='Số lượng ảnh muốn vẽ (mặc định 4)')
    parser.add_argument('--threshold', type=float, default=0.5, help='Ngưỡng xác suất (mặc định 0.5)')
    parser.add_argument('--output', type=str, default='visualization.png', help='Đường dẫn lưu ảnh đầu ra')
    
    args = parser.parse_args()
    main(args)
