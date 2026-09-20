import cv2
import numpy as np
import os
import glob

# Thư mục gốc chứa các ảnh mask
INPUT_DIR = 'domain_noper_images'

# Tạo ma trận kernel, tăng giá trị (15, 15) để vùng xóa lan rộng hơn
kernel = np.ones((15, 15), np.uint8)

# Đếm số file đã xử lý
processed = 0
skipped = 0

# Duyệt qua tất cả các thư mục con và tìm tất cả file mask
for root, dirs, files in os.walk(INPUT_DIR):
    # Lọc chỉ lấy file mask (kết thúc bằng _mask.png)
    mask_files = sorted([f for f in files if f.endswith('_mask.png')])

    for mask_filename in mask_files:
        mask_path = os.path.join(root, mask_filename)

        # Đọc mask dưới dạng ảnh xám
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if mask is None:
            print(f"[SKIP] Không đọc được: {mask_path}")
            skipped += 1
            continue

        # Dilate mask
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)

        # Ghi đè file mask gốc bằng mask đã dilate
        cv2.imwrite(mask_path, dilated_mask)
        processed += 1

        if processed % 100 == 0:
            print(f"[PROGRESS] Đã xử lý {processed} mask...")

print(f"\n[DONE] Hoàn tất! Đã xử lý {processed} mask, bỏ qua {skipped} file.")