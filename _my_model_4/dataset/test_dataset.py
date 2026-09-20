# test_dataset.py
# Script kiểm thử tự động bộ đọc dữ liệu và tiền xử lý ảnh cho CLIP-PAR.
# Script này tự động tạo dữ liệu mô phỏng (mock dataset) để kiểm chứng tính đúng đắn của code.

import os
import tempfile
import pickle
import numpy as np
from PIL import Image
import torch

from clip_dataset import CLIPPARDataset, get_clip_transforms

# Lớp mô phỏng thông tin dataset được đóng gói giống cấu trúc trong OpenPAR
class DummyDatasetInfo:
    def __init__(self, image_names, labels, partition, attributes, root):
        self.image_name = image_names
        self.label = labels
        self.partition = partition
        self.attributes = attributes
        self.root = root

def create_mock_dataset(temp_dir):
    """
    Tạo tập dữ liệu mô phỏng gồm ảnh giả lập và file pickle chứa nhãn thuộc tính.
    """
    img_dir = os.path.join(temp_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    
    # 1. Tạo 2 ảnh giả lập kích thước khác nhau (để test tính năng PadToSquare)
    img1_path = os.path.join(img_dir, "pedestrian_1.jpg")
    Image.new("RGB", (120, 300), color=(255, 0, 0)).save(img1_path)  # Ảnh cao gầy
    
    img2_path = os.path.join(img_dir, "pedestrian_2.jpg")
    Image.new("RGB", (200, 150), color=(0, 255, 0)).save(img2_path)  # Ảnh dẹt ngang
    
    # 2. Định nghĩa danh sách ảnh, nhãn thuộc tính và thuộc tính mẫu
    image_names = ["pedestrian_1.jpg", "pedestrian_2.jpg"]
    
    # 2 mẫu, mỗi mẫu có 3 thuộc tính: ['female', 'backpack', 'glasses']
    labels = np.array([
        [1, 0, 1],  # pedestrian_1: Nữ, không đeo balo, có đeo kính
        [0, 1, 0]   # pedestrian_2: Nam, có đeo balo, không đeo kính
    ], dtype=np.int64)
    
    partition = {
        'train': np.array([0, 1], dtype=np.int64)
    }
    
    attributes = ['female', 'backpack', 'glasses']
    
    # Đóng gói đối tượng thông tin
    dataset_info = DummyDatasetInfo(image_names, labels, partition, attributes, img_dir)
    
    # 3. Ghi ra file pickle tạm thời
    pkl_path = os.path.join(temp_dir, "mock_dataset.pkl")
    with open(pkl_path, 'wb') as f:
        pickle.dump(dataset_info, f)
        
    return pkl_path, img_dir

def run_tests():
    print("=" * 60)
    print("STARTING DATASET LOADER TESTS FOR CLIP-PAR")
    print("=" * 60)
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"[*] Creating mock dataset in temporary directory: {temp_dir}")
        pkl_path, img_dir = create_mock_dataset(temp_dir)
        
        # 1. Test CLIP Transforms initialization
        print("\n[1] Testing CLIP transforms initialization...")
        train_trans, valid_trans = get_clip_transforms(height=224, width=224)
        print(" -> Success! Transforms configured with CLIP norm and PadToSquare.")
        
        # 2. Test CLIPPARDataset initialization
        print("\n[2] Testing CLIPPARDataset loading...")
        dataset = CLIPPARDataset(
            pkl_path=pkl_path,
            img_dir=img_dir,
            split='train',
            transform=train_trans
        )
        print(f" -> Success! Loaded samples: {len(dataset)}")
        print(f" -> Attributes detected: {dataset.attributes}")
        
        # 3. Test Prompt Auto-Generation
        print("\n[3] Testing prompt auto-generation...")
        neg_prompts, pos_prompts = dataset.get_prompts()
        print(" -> Positive Prompts:")
        for i, prompt in enumerate(pos_prompts):
            print(f"    - Attribute '{dataset.attributes[i]}': {prompt}")
        print(" -> Negative Prompts:")
        for i, prompt in enumerate(neg_prompts):
            print(f"    - Attribute '{dataset.attributes[i]}': {prompt}")
            
        assert len(pos_prompts) == len(dataset.attributes)
        assert len(neg_prompts) == len(dataset.attributes)
        
        # 4. Test image fetching and pre-processing
        print("\n[4] Testing data retrieval (__getitem__)...")
        img_tensor, label_tensor, img_name = dataset[0]
        
        print(f" -> Sample 1 retrieved: {img_name}")
        print(f" -> Image Tensor Shape: {img_tensor.shape} (Expected: torch.Size([3, 224, 224]))")
        print(f" -> Attributes Label (float32): {label_tensor} (Expected: [1.0, 0.0, 1.0])")
        
        assert isinstance(img_tensor, torch.Tensor)
        assert img_tensor.shape == (3, 224, 224)
        assert np.array_equal(label_tensor, np.array([1., 0., 1.], dtype=np.float32))
        
        # Test sample 2 (landscape image to verify PadToSquare)
        img_tensor_2, label_tensor_2, img_name_2 = dataset[1]
        assert img_tensor_2.shape == (3, 224, 224)
        assert np.array_equal(label_tensor_2, np.array([0., 1., 0.], dtype=np.float32))
        
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
