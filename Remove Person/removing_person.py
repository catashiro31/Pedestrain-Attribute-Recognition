#!/usr/bin/env python3
"""
Removing person from images using LaMa inpainting model.
Processes all domains in domain_noper_images folder.

Input structure:
    domain_noper_images/
        Construction Site/
            part1__001_00010_002.jpg
            part1__001_00010_002_mask.png
        Market/
            part0_03_00007_005.jpg
            part0_03_00007_005_mask.png
        ...

Output structure (mirrors input):
    domain_noper_images_inpainted/
        Construction Site/
            part1__001_00010_002.jpg
        Market/
            part0_03_00007_005.jpg
        ...
"""

import os
import sys
import glob
import logging
import traceback

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import cv2
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate

# Thêm lama vào sys.path để import được các module
LAMA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lama')
sys.path.insert(0, LAMA_DIR)

from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.evaluation.refinement import refine_predict
from saicinpainting.evaluation.data import (
    load_image, pad_img_to_modulo, scale_image
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
LOGGER = logging.getLogger(__name__)

# ===================== CẤU HÌNH =====================
INPUT_DIR = '/workspace/domain_noper_images'
OUTPUT_DIR = '/workspace/domain_noper_images_inpainted'
MODEL_PATH = '/workspace/lama/big-lama'
CHECKPOINT = 'best.ckpt'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PAD_OUT_TO_MODULO = 8
IMG_SUFFIX = '.jpg'
OUT_EXT = '.jpg'

# Refine: bật/tắt bước tinh chỉnh sau inpainting
ENABLE_REFINE = True
REFINER_CONFIG = {
    'gpu_ids': '0',           # GPU ids, ví dụ '0,1' cho multi-GPU
    'modulo': PAD_OUT_TO_MODULO,
    'n_iters': 15,            # Số iteration refinement mỗi scale
    'lr': 0.002,              # Learning rate
    'min_side': 512,          # Kích thước tối thiểu mỗi cạnh
    'max_scales': 3,          # Số scale tối đa cho image-mask pyramid
    'px_budget': 1800000,     # Budget pixel (H*W <= px_budget)
}
# =====================================================


def load_lama_model(model_path, checkpoint_name, device, use_refine=False):
    """Load pretrained LaMa model."""
    train_config_path = os.path.join(model_path, 'config.yaml')
    with open(train_config_path, 'r') as f:
        train_config = OmegaConf.create(yaml.safe_load(f))

    train_config.training_model.predict_only = True
    train_config.visualizer.kind = 'noop'

    checkpoint_path = os.path.join(model_path, 'models', checkpoint_name)
    model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cpu')
    model.freeze()

    # Khi dùng refine, model ở CPU (refine_predict tự chuyển lên GPU theo từng phần)
    # Khi không refine, chuyển model lên device cho inference nhanh
    if not use_refine:
        model.to(device)

    LOGGER.info(f"Model loaded (refine={'ON' if use_refine else 'OFF'}, device={device})")
    return model


def prepare_batch(image_path, mask_path, pad_out_to_modulo=8):
    """Prepare a single image-mask pair as a batch for inference."""
    image = load_image(image_path, mode='RGB')       # (3, H, W) float32 [0,1]
    mask = load_image(mask_path, mode='L')            # (H, W) float32 [0,1]
    mask = mask[None, ...]                            # (1, H, W)

    result = dict(image=image, mask=mask)

    if pad_out_to_modulo is not None and pad_out_to_modulo > 1:
        result['unpad_to_size'] = result['image'].shape[1:]  # (H, W) trước khi pad
        result['image'] = pad_img_to_modulo(result['image'], pad_out_to_modulo)
        result['mask'] = pad_img_to_modulo(result['mask'], pad_out_to_modulo)

    return result


def inpaint_single(model, image_path, mask_path, device, pad_out_to_modulo=8,
                   use_refine=False, refiner_config=None):
    """Run LaMa inpainting on a single image-mask pair. Returns BGR uint8 numpy array.
    
    Nếu use_refine=True, chạy thêm bước refine_predict (multi-scale iterative refinement)
    để cải thiện chất lượng inpainting, đặc biệt hiệu quả với ảnh lớn.
    """
    sample = prepare_batch(image_path, mask_path, pad_out_to_modulo)
    batch = default_collate([sample])

    if use_refine and refiner_config is not None:
        assert 'unpad_to_size' in batch, "Unpadded size is required for refinement"
        cur_res = refine_predict(batch, model, **refiner_config)
        cur_res = cur_res[0].permute(1, 2, 0).detach().cpu().numpy()
    else:
        with torch.no_grad():
            batch = move_to_device(batch, device)
            batch['mask'] = (batch['mask'] > 0) * 1
            batch = model(batch)

            cur_res = batch['inpainted'][0].permute(1, 2, 0).detach().cpu().numpy()

            unpad_to_size = batch.get('unpad_to_size', None)
            if unpad_to_size is not None:
                orig_height, orig_width = unpad_to_size
                cur_res = cur_res[:orig_height, :orig_width]

    cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
    cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
    return cur_res


def find_all_mask_image_pairs(input_dir, img_suffix='.jpg'):
    """
    Tìm tất cả các cặp (image, mask) trong input_dir.
    Quy tắc đặt tên: 
        image: <name>.jpg
        mask:  <name>_mask.png
    """
    mask_files = sorted(glob.glob(os.path.join(input_dir, '**', '*_mask.png'), recursive=True))
    pairs = []
    for mask_path in mask_files:
        # Từ mask path, suy ra image path
        # mask: /path/to/part0_03_00007_005_mask.png
        # image: /path/to/part0_03_00007_005.jpg
        img_path = mask_path.rsplit('_mask', 1)[0] + img_suffix
        if os.path.exists(img_path):
            pairs.append((img_path, mask_path))
        else:
            LOGGER.warning(f"Image not found for mask: {mask_path} (expected: {img_path})")
    return pairs


def main():
    LOGGER.info(f"Input directory:  {INPUT_DIR}")
    LOGGER.info(f"Output directory: {OUTPUT_DIR}")
    LOGGER.info(f"Model:            {MODEL_PATH}")
    LOGGER.info(f"Device:           {DEVICE}")

    # Load model
    device = torch.device(DEVICE)
    model = load_lama_model(MODEL_PATH, CHECKPOINT, device, use_refine=ENABLE_REFINE)
    LOGGER.info(f"Refine: {'ENABLED' if ENABLE_REFINE else 'DISABLED'}")

    # Tìm tất cả cặp image-mask
    pairs = find_all_mask_image_pairs(INPUT_DIR, img_suffix=IMG_SUFFIX)
    total = len(pairs)
    LOGGER.info(f"Found {total} image-mask pairs across all domains")

    if total == 0:
        LOGGER.warning("No pairs found. Check directory structure and file naming.")
        return

    processed = 0
    failed = 0

    for img_path, mask_path in pairs:
        try:
            # Tính output path: giữ nguyên cấu trúc thư mục con
            rel_path = os.path.relpath(img_path, INPUT_DIR)
            # Đổi extension nếu cần
            out_name = os.path.splitext(rel_path)[0] + OUT_EXT
            out_path = os.path.join(OUTPUT_DIR, out_name)

            # Tạo thư mục output nếu chưa có
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            # Skip nếu đã xử lý rồi (cho phép resume)
            if os.path.exists(out_path):
                processed += 1
                if processed % 500 == 0:
                    LOGGER.info(f"[PROGRESS] {processed}/{total} (skipped existing)")
                continue

            # Inpaint (+ refine nếu bật)
            result = inpaint_single(
                model, img_path, mask_path, device, PAD_OUT_TO_MODULO,
                use_refine=ENABLE_REFINE, refiner_config=REFINER_CONFIG
            )
            cv2.imwrite(out_path, result)
            processed += 1

            if processed % 100 == 0:
                LOGGER.info(f"[PROGRESS] {processed}/{total} processed")

        except Exception as ex:
            failed += 1
            LOGGER.error(f"Failed: {img_path} - {ex}\n{traceback.format_exc()}")

    LOGGER.info(f"\n[DONE] Processed: {processed}, Failed: {failed}, Total: {total}")


if __name__ == '__main__':
    main()
