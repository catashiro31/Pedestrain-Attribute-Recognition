import torch
from config import get_kaggle_config
from model import CLIMPPAR

def test():
    config = get_kaggle_config()
    config.vmamba_pretrained = False
    
    # Giả lập input
    B = 2
    images = torch.randn(B, 3, 448, 448)
    
    # Giả lập cached word embeddings (57 attributes, mỗi attribute có L_w=12 tokens)
    N_attr = 57
    L_w = 12
    d_model = 768  # d_model của Mamba-130m
    cached_word_embeddings = torch.randn(N_attr, L_w, d_model)
    cached_word_mask = torch.ones(N_attr, L_w, dtype=torch.long)
    # Giả lập padding: 3 token cuối mỗi attribute là padding
    cached_word_mask[:, -3:] = 0
    
    model = CLIMPPAR(config)
    
    print("=" * 60)
    print("Testing CLIMP-PAR v6 (CoCoOp Conditional Prompt Learning)...")
    print("=" * 60)
    
    # Test 1: CoCoOp mode (chế độ chính)
    print("\n[Test 1] CoCoOp mode (cached_word_embeddings)...")
    logits, _ = model(
        images, 
        cached_word_embeddings=cached_word_embeddings,
        cached_word_mask=cached_word_mask
    )
    print(f"  Output shape: {logits.shape}")
    assert logits.shape == (B, 57), f"Expected ({B}, 57), got {logits.shape}"
    print("  ✔ Test 1 passed!")
    
    # Test 2: Legacy mode (tương thích v4)
    print("\n[Test 2] Legacy mode (cached_text_features)...")
    cached_text_features = torch.randn(57, 768)
    logits2, _ = model(images, cached_text_features=cached_text_features)
    print(f"  Output shape: {logits2.shape}")
    assert logits2.shape == (B, 57), f"Expected ({B}, 57), got {logits2.shape}"
    print("  ✔ Test 2 passed!")
    
    # Test 3: Kiểm tra Meta-Net
    print("\n[Test 3] Meta-Net output shape...")
    image_global = torch.randn(B, 768)
    bias = model.prompt_learner.meta_net(image_global)
    print(f"  Meta-Net output shape: {bias.shape}")
    assert bias.shape == (B, 768), f"Expected ({B}, 768), got {bias.shape}"
    print("  ✔ Test 3 passed!")
    
    # Test 4: ConditionalPromptLearner
    print("\n[Test 4] ConditionalPromptLearner output shapes...")
    prompt_embs, prompt_mask = model.prompt_learner(
        image_global, cached_word_embeddings, cached_word_mask
    )
    n_ctx = config.n_ctx
    expected_B_N = B * N_attr
    expected_L = n_ctx + L_w
    print(f"  Prompt embeddings: {prompt_embs.shape}")
    print(f"  Prompt mask: {prompt_mask.shape}")
    assert prompt_embs.shape == (expected_B_N, expected_L, d_model), \
        f"Expected ({expected_B_N}, {expected_L}, {d_model}), got {prompt_embs.shape}"
    assert prompt_mask.shape == (expected_B_N, expected_L), \
        f"Expected ({expected_B_N}, {expected_L}), got {prompt_mask.shape}"
    print("  ✔ Test 4 passed!")
    
    # Test 5: Kiểm tra số parameters
    print("\n[Test 5] Parameter count...")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    prompt_params = sum(p.numel() for p in model.prompt_learner.parameters())
    print(f"  Total: {total_params / 1e6:.1f}M")
    print(f"  Trainable: {trainable_params / 1e6:.1f}M")
    print(f"  CoCoOp Prompt Learner: {prompt_params / 1e3:.1f}K")
    print("  ✔ Test 5 passed!")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✔")
    print("=" * 60)

if __name__ == '__main__':
    test()
