import os
import sys
# Thêm đường dẫn project vào sys.path để có thể import từ config, model, dataset
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from config import get_kaggle_config
from model.climp_par import CLIMPPAR
from dataset.clip_dataset import PARDataset, get_clip_transforms

class CLIMPPARWrapper(torch.nn.Module):
    """
    Wrapper bọc lại model để Grad-CAM có thể truyền input tensor trực tiếp 
    và nhận logits tương ứng thay vì tuple outputs.
    """
    def __init__(self, model, bg_images):
        super().__init__()
        self.model = model
        self.bg_images = bg_images
        
    def forward(self, images):
        # Trả ra logits (đầu ra đầu tiên của CLIMPPAR)
        logits, _ = self.model(images, self.bg_images, return_features=False)
        return logits

def reshape_transform(tensor):
    # VMamba backbone trả về tensor có kích thước (B, H, W, C)
    # Grad-CAM yêu cầu định dạng (B, C, H, W) nên ta cần permute lại
    result = tensor.permute(0, 3, 1, 2)
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Đường dẫn tới file checkpoint')
    parser.add_argument('--attr_idx', type=int, default=9, help='Class ID để theo dõi Grad-CAM (Ví dụ 9: Backpack)')
    parser.add_argument('--output_dir', type=str, default='output', help='Thư mục lưu kết quả')
    args = parser.parse_args()
    
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    
    config = get_kaggle_config()
    # Tắt gradient checkpointing để tính toán backward gradient cho GradCAM mượt mà
    config.gradient_checkpointing = False
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Sử dụng thiết bị: {device}")
    
    _, test_transform = get_clip_transforms(config.img_height, config.img_width)
    test_dataset = PARDataset(config.pkl_path, config.img_dir, config.bg_img_dir, split='test', transform=test_transform)
    
    # Dùng batch_size 1 để lấy từng ảnh một hiển thị
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    
    config.vmamba_pretrained = False
    model = CLIMPPAR(config, test_dataset.attributes)
    
    print(f"Đang tải trọng số từ: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    clean_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict)
    model = model.to(device)
    model.eval()
    
    # Layer mục tiêu cho Grad-CAM: block cuối cùng của VMamba backbone
    target_layers = [model.vision_encoder.backbone.layers[-1]]
    
    num_images_to_test = args.num_images if hasattr(args, 'num_images') else 5
    
    print(f"Đang xử lý {num_images_to_test} ảnh ngẫu nhiên...")
    
    for i in range(num_images_to_test):
        # Lấy 1 ảnh ngẫu nhiên từ test_loader
        images, bg_images, labels, img_names = next(iter(test_loader))
        images = images.to(device)
        bg_images = bg_images.to(device)
        
        # Bọc model với wrapper truyền sẵn background context
        wrapper_model = CLIMPPARWrapper(model, bg_images)
        
        # Khởi tạo GradCAM
        cam = GradCAM(model=wrapper_model, target_layers=target_layers, reshape_transform=reshape_transform)
        
        # Tìm các thuộc tính dương tính (Ground truth = 1) của ảnh này
        positive_attrs = torch.where(labels[0] == 1)[0].tolist()
        
        # Nếu không có thuộc tính nào, chọn đại 3 thuộc tính ngẫu nhiên
        if len(positive_attrs) == 0:
            positive_attrs = [9, 15, 23]
            
        # Chọn tối đa 3 thuộc tính để vẽ
        selected_attrs = positive_attrs[:3]
        num_cols = len(selected_attrs) + 1
        
        fig, axes = plt.subplots(1, num_cols, figsize=(5 * num_cols, 5))
        
        # Trích xuất ảnh thật từ input tensor
        clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1).to(device)
        clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1).to(device)
        
        unnorm_img = images[0] * clip_std + clip_mean
        unnorm_img = unnorm_img.clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
        
        axes[0].imshow(unnorm_img)
        axes[0].set_title(f"Ảnh Gốc\n{img_names[0]}")
        axes[0].axis('off')
        
        for idx, attr_idx in enumerate(selected_attrs):
            targets = [ClassifierOutputTarget(attr_idx)]
            
            # Tính toán CAM
            grayscale_cam = cam(input_tensor=images, targets=targets)[0, :]
            cam_image = show_cam_on_image(unnorm_img, grayscale_cam, use_rgb=True)
            
            attr_name = test_dataset.attributes[attr_idx]
            gt_label = labels[0, attr_idx].item()
            
            ax = axes[idx + 1]
            ax.imshow(cam_image)
            ax.set_title(f"Grad-CAM: '{attr_name}'\nGround Truth: {gt_label}")
            ax.axis('off')
            
        plt.tight_layout()
        # Đặt tên file dựa trên tên ảnh
        safe_img_name = img_names[0].replace('/', '_').replace('.jpg', '')
        out_path = os.path.join(out_dir, f'gradcam_multi_{safe_img_name}.png')
        plt.savefig(out_path)
        plt.close(fig)
        print(f"Đã lưu hình ảnh Grad-CAM vào: {out_path}")

if __name__ == '__main__':
    main()
