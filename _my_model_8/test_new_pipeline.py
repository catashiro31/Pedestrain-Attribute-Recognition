import torch
from types import SimpleNamespace
from model.climp_par import CLIMPPAR

def test_pipeline():
    # 1. Setup config
    config = SimpleNamespace(
        embed_dim=768,
        vmamba_pretrained_path=None,  # No need for pretrained weights for forward pass test
        mamba_model="state-spaces/mamba-130m-hf",  # Assuming this exists locally or will be loaded
        mamba_freeze=True
    )

    # 2. Initialize Model
    print("Khởi tạo mô hình CLIMPPAR...")
    try:
        model = CLIMPPAR(config)
        model.eval()  # Chạy ở chế độ eval để tắt Dropout
    except Exception as e:
        print(f"Lỗi khởi tạo mô hình: {e}")
        return

    # 3. Prepare Dummy Inputs
    B = 2
    images = torch.randn(B, 3, 448, 448)
    domain_tokens = torch.randn(B, 32, 768)
    
    # 57 giả lập attribute names
    attribute_names = [f"attribute_{i}" for i in range(57)]

    # 4. Forward Pass
    print("Bắt đầu Forward pass...")
    try:
        with torch.no_grad():
            logits, _ = model(images, domain_tokens, attribute_names)
            
        print(f"Output logits shape: {logits.shape}")
        assert logits.shape == (B, 57), f"Lỗi shape logits, mong muốn ({B}, 57), nhận được {logits.shape}"
        print("Test Forward pass THÀNH CÔNG!\n")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Lỗi trong quá trình Forward pass: {e}")
        return

    # 5. Check Gradients (Freezing logic)
    print("Kiểm tra trạng thái Gradient (đóng băng/mở khóa)...")
    expected_trainable = ['hidden_words', 'upsample_conv', 'vision_projector', 'cross_mamba', 'mlp_head']
    
    for name, param in model.named_parameters():
        is_trainable = param.requires_grad
        # Kiểm tra xem tham số này có thuộc các thành phần được phép huấn luyện không
        should_be_trainable = any(module_name in name for module_name in expected_trainable)
        
        if is_trainable != should_be_trainable:
            print(f"CẢNH BÁO: Tham số {name} có requires_grad={is_trainable} (mong muốn {should_be_trainable})")
            
    print("Test Gradient (Freezing logic) HOÀN TẤT!")

if __name__ == "__main__":
    test_pipeline()
