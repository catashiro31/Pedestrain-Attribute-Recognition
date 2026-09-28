import torch
from config import get_kaggle_config
from model import CLIMPPAR

def test():
    config = get_kaggle_config()
    config.vmamba_pretrained = False
    
    # Attributes mock
    attributes = [f"attr_{i}" for i in range(57)]
    
    # Giả lập input
    B = 2
    images = torch.randn(B, 3, 256, 128)
    bg_images = torch.randn(B, 3, 256, 128)
    
    model = CLIMPPAR(config, attributes)
    
    print("Testing CLIMP-PAR v4...")
    # Tắt việc load weights để test cho nhanh
    # (hoặc mock)
    logits, _ = model(images, bg_images)
    
    print("Output shape:", logits.shape)
    assert logits.shape == (B, 57), f"Expected ({B}, 57), got {logits.shape}"
    
    # Tính toán FLOPs và Params
    try:
        from thop import profile, clever_format
        
        # Tạo dummy input với batch_size = 1
        dummy_img = torch.randn(1, 3, 256, 128)
        dummy_bg = torch.randn(1, 3, 256, 128)
        
        # Profile cần tuple args cho hàm forward
        macs, params = profile(model, inputs=(dummy_img, dummy_bg), verbose=False)
        macs, params = clever_format([macs, params], "%.2f")
        
        print(f"\n{'='*40}")
        print(f"MODEL COMPLEXITY (Input: 256x128)")
        print(f"{'='*40}")
        print(f"Parameters: {params}")
        print(f"MACs/FLOPs: {macs}")
        print(f"{'='*40}\n")
    except ImportError:
        print("\n[CẢNH BÁO] Không tìm thấy thư viện 'thop'. Chạy 'pip install thop' để xem FLOPs & Params.")
        
    print("Test passed!")

if __name__ == '__main__':
    test()
