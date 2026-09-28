import os
import sys
import argparse
import torch
import numpy as np
from PIL import Image
import cv2
import pickle

from config import get_kaggle_config
from model import CLIMPPAR
from dataset.clip_dataset import get_clip_transforms

# ==============================================================================
# BỔ SUNG ĐƯỜNG DẪN IMPORT SAM3 VÀ LAMA
# ==============================================================================
SAM_DIR = "/workspace/Person Segmentation"
LAMA_DIR = "/workspace/Remove Person"

if SAM_DIR not in sys.path:
    sys.path.insert(0, SAM_DIR)
if LAMA_DIR not in sys.path:
    sys.path.insert(0, LAMA_DIR)
if os.path.join(LAMA_DIR, 'lama') not in sys.path:
    sys.path.insert(0, os.path.join(LAMA_DIR, 'lama'))

try:
    # SAM3
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    
    # LaMa
    import yaml
    from omegaconf import OmegaConf
    from torch.utils.data._utils.collate import default_collate
    from saicinpainting.training.trainers import load_checkpoint
    from saicinpainting.evaluation.utils import move_to_device
    from saicinpainting.evaluation.refinement import refine_predict
    from saicinpainting.evaluation.data import pad_img_to_modulo
except ImportError as e:
    print(f"[LỖI] Không thể import các module SAM3/LaMa: {e}")
    print("Vui lòng đảm bảo các thư mục 'Person Segmentation' và 'Remove Person' tồn tại trong /workspace.")

# ==============================================================================
# PIPELINE BACKGROUND EXTRACTION (SAM3 + LaMa Refine)
# ==============================================================================

class BackgroundExtractor:
    """
    Class xử lý Pipeline tách nền tự động cho inference thực tế.
    Tích hợp trực tiếp SAM3 và LaMa từ các thư mục của dự án.
    """
    def __init__(self, device='cuda'):
        self.device = device
        print("[BackgroundExtractor] Đang khởi tạo SAM3 và LaMa...")
        
        # 1. Khởi tạo SAM3
        print("  - Loading SAM3...")
        self.sam_model = build_sam3_image_model().to(device)
        self.sam_processor = Sam3Processor(self.sam_model)
        
        # 2. Khởi tạo LaMa
        print("  - Loading LaMa (big-lama)...")
        model_path = os.path.join(LAMA_DIR, 'lama', 'big-lama')
        checkpoint_name = 'best.ckpt'
        train_config_path = os.path.join(model_path, 'config.yaml')
        
        with open(train_config_path, 'r') as f:
            train_config = OmegaConf.create(yaml.safe_load(f))
            
        train_config.training_model.predict_only = True
        train_config.visualizer.kind = 'noop'
        
        checkpoint_path = os.path.join(model_path, 'models', checkpoint_name)
        # Giữ LaMa ở CPU để chạy module Refine (tự chuyển từng phần lên GPU)
        self.lama_model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cpu')
        self.lama_model.freeze()
        
        self.refiner_config = {
            'gpu_ids': '0',           
            'modulo': 8,
            'n_iters': 15,            
            'lr': 0.002,              
            'min_side': 512,          
            'max_scales': 3,          
            'px_budget': 1800000,     
        }
        print("[BackgroundExtractor] Đã sẵn sàng!")

    def get_person_mask(self, img_pil):
        """
        Sử dụng SAM3 để lấy mask của người đi bộ.
        """
        with torch.autocast("cuda", dtype=torch.bfloat16):
            inference_state = self.sam_processor.set_image(img_pil)
            output = self.sam_processor.set_text_prompt(state=inference_state, prompt="person")
            masks = output["masks"]  # shape: [N, 1, H, W], boolean tensor

            if masks.numel() == 0 or masks.shape[0] == 0:
                # Không tìm thấy người -> trả về mask rỗng
                w, h = img_pil.size
                merged = np.zeros((h, w), dtype=np.uint8)
            else:
                # Gộp tất cả các mask lại
                merged = masks.squeeze(1).any(dim=0).cpu().numpy().astype(np.uint8) * 255
                
        return merged

    def inpaint_background(self, img_pil, mask):
        """
        Sử dụng LaMa + Refine để xóa vùng mask và tái tạo lại background.
        """
        # Chuẩn bị Tensor cho LaMa (chuẩn [0, 1])
        image = np.array(img_pil.convert('RGB')) / 255.0
        image = image.astype(np.float32).transpose(2, 0, 1)  # (3, H, W)
        
        mask_f32 = mask.astype(np.float32) / 255.0
        mask_f32 = mask_f32[None, ...]  # (1, H, W)
        
        # Đưa vào dictionary batch format của LaMa
        result_dict = dict(image=image, mask=mask_f32)
        pad_out_to_modulo = 8
        
        result_dict['unpad_to_size'] = result_dict['image'].shape[1:]  # (H, W) gốc
        result_dict['image'] = pad_img_to_modulo(result_dict['image'], pad_out_to_modulo)
        result_dict['mask'] = pad_img_to_modulo(result_dict['mask'], pad_out_to_modulo)
        
        batch = default_collate([result_dict])
        
        # Chạy Refine (sẽ tự động dùng GPU được chỉ định trong gpu_ids)
        cur_res = refine_predict(batch, self.lama_model, **self.refiner_config)
        cur_res = cur_res[0].permute(1, 2, 0).detach().cpu().numpy()
        
        # Unpad
        orig_height, orig_width = result_dict['unpad_to_size']
        cur_res = cur_res[:orig_height, :orig_width]
        
        # Trả về ảnh RGB
        cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
        bg_pil = Image.fromarray(cur_res)
        return bg_pil

    def process(self, img_pil):
        """
        Thực thi toàn bộ pipeline: Image -> SAM3 Mask -> LaMa Inpaint -> Background Image
        """
        mask = self.get_person_mask(img_pil)
        bg_img_pil = self.inpaint_background(img_pil, mask)
        return mask, bg_img_pil

# ==============================================================================
# INFERENCE PIPELINE CHÍNH (CLIMP-PAR v4)
# ==============================================================================

def load_model(checkpoint_path, config, attributes, device):
    print("Khởi tạo mô hình CLIMP-PAR v4...")
    config.vmamba_pretrained = False
    model = CLIMPPAR(config, attributes)
    
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        clean_state_dict = {}
        for k, v in state_dict.get('model', state_dict).items():
            clean_state_dict[k.replace('module.', '')] = v
        model.load_state_dict(clean_state_dict)
        print(f"Tải trọng số từ {checkpoint_path} thành công!")
    else:
        print(f"CẢNH BÁO: Không tìm thấy checkpoint tại {checkpoint_path}")
        
    model.to(device)
    model.eval()
    return model

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = get_kaggle_config()
    
    # Load attributes từ file pickle (MSP60k)
    pkl_path = args.pkl_path if args.pkl_path else config.pkl_path
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
            attributes = getattr(data, 'attributes', getattr(data, 'attr_name', []))
    except Exception as e:
        print(f"Không thể đọc file PKL: {e}. Sử dụng danh sách giả...")
        attributes = [f"attr_{i}" for i in range(57)]

    # 1. Load pipeline Background Extractor (Kế thừa SAM3 + LaMa)
    bg_extractor = BackgroundExtractor(device=device)

    # 2. Load model
    model = load_model(args.checkpoint, config, attributes, device)
    
    # 3. Transform
    _, test_transform = get_clip_transforms(config.img_height, config.img_width)
    
    if not os.path.exists(args.img_path):
        print(f"Lỗi: Không tìm thấy ảnh {args.img_path}")
        return

    print(f"\n--- ĐANG XỬ LÝ ẢNH: {args.img_path} ---")
    img_pil = Image.open(args.img_path).convert("RGB")
    
    # BƯỚC 1: Tính toán offline background (Runtime Pipeline)
    print("1. Chạy SAM3 + LaMa (Refine) để tách nền...")
    mask, bg_img_pil = bg_extractor.process(img_pil)
    
    if args.save_bg:
        Image.fromarray(mask).save("debug_mask.png")
        bg_img_pil.save("debug_extracted_bg.jpg")
        print(f"   Đã lưu mask và background tạm ra: debug_mask.png, debug_extracted_bg.jpg")

    # BƯỚC 2: Chuyển đổi tensor
    print("2. Chuyển đổi Tensor...")
    img_tensor = test_transform(img_pil).unsqueeze(0).to(device)
    bg_img_tensor = test_transform(bg_img_pil).unsqueeze(0).to(device)
    
    # BƯỚC 3: Dự đoán bằng CLIMP-PAR
    print("3. CLIMP-PAR Inferencing...")
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=config.use_amp):
        logits, _ = model(img_tensor, bg_img_tensor)
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
    
    # BƯỚC 4: In kết quả
    print("\n" + "="*50)
    print("KẾT QUẢ DỰ ĐOÁN (Threshold = 0.5)")
    print("="*50)
    
    predicted_attrs = []
    for i, attr in enumerate(attributes):
        prob = probs[i]
        if prob >= 0.5:
            predicted_attrs.append((attr, prob))
            print(f" [V] {attr:<20} : {prob*100:.2f}%")
            
    print("="*50)
    print(f"Tổng số thuộc tính phát hiện: {len(predicted_attrs)}/{len(attributes)}")

if __name__ == '__main__':
    # Tối ưu cho Ampere / BFloat16 theo config cũ của SAM3
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    parser = argparse.ArgumentParser(description='Inference CLIMP-PAR v4 End-to-End (Tích hợp SAM3 + LaMa)')
    parser.add_argument('--img_path', type=str, required=True, help='Đường dẫn tới ảnh')
    parser.add_argument('--checkpoint', type=str, required=True, help='Đường dẫn checkpoint model')
    parser.add_argument('--pkl_path', type=str, default='/workspace/MSP60k/SUBMIT/dataset_random.pkl', help='Đường dẫn dataset pkl')
    parser.add_argument('--save_bg', action='store_true', help='Lưu mask và ảnh background')
    
    args = parser.parse_args()
    main(args)
