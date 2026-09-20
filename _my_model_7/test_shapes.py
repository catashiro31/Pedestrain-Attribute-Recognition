import torch
from config import get_kaggle_config
from model import CLIMPPAR
from model.attribute_gcn import AttributeGCN

def test():
    config = get_kaggle_config()
    config.vmamba_pretrained = False
    
    # === Test 1: AttributeGCN standalone ===
    print("Test 1: AttributeGCN standalone...")
    gcn = AttributeGCN(num_attrs=57, d_model=768, num_layers=2)
    
    B = 2
    x = torch.randn(B, 57, 768)
    out = gcn(x)
    
    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {out.shape}")
    assert out.shape == (B, 57, 768), f"Expected (2, 57, 768), got {out.shape}"
    
    # Kiểm tra adjacency matrix
    A = gcn.get_adjacency_matrix()
    print(f"  Adjacency matrix shape: {A.shape}")
    assert A.shape == (57, 57), f"Expected (57, 57), got {A.shape}"
    
    # Kiểm tra giá trị adjacency nằm trong [0, 1]
    assert A.min() >= 0 and A.max() <= 1, "Adjacency values should be in [0, 1]"
    print("  ✓ Test 1 passed!")
    
    # === Test 2: Full model forward ===
    print("\nTest 2: Full CLIMP-PAR v7 forward...")
    images = torch.randn(B, 3, 256, 128)
    cached_text_features = torch.randn(57, 768)
    
    model = CLIMPPAR(config)
    
    logits, _ = model(images, cached_text_features=cached_text_features)
    
    print(f"  Output shape: {logits.shape}")
    assert logits.shape == (B, 57), f"Expected (2, 57), got {logits.shape}"
    print("  ✓ Test 2 passed!")
    
    # === Test 3: Gradient flow qua GCN ===
    print("\nTest 3: Gradient flow through GCN...")
    x = torch.randn(B, 57, 768, requires_grad=True)
    gcn = AttributeGCN(num_attrs=57, d_model=768, num_layers=2)
    out = gcn(x)
    loss = out.sum()
    loss.backward()
    
    assert x.grad is not None, "Gradient should flow through GCN"
    assert gcn.A_raw.grad is not None, "Adjacency matrix should receive gradient"
    print("  ✓ Test 3 passed!")
    
    # === Test 4: So sánh parameters v4 vs v7 ===
    print("\nTest 4: Parameter count comparison...")
    total_params = sum(p.numel() for p in model.parameters())
    gcn_params = sum(p.numel() for p in model.attr_gcn.parameters())
    print(f"  Total params:     {total_params / 1e6:.2f}M")
    print(f"  GCN params:       {gcn_params / 1e6:.2f}M")
    print(f"  GCN overhead:     {gcn_params / total_params * 100:.2f}%")
    print("  ✓ Test 4 passed!")
    
    print("\n" + "=" * 40)
    print("All tests passed! ✓")

if __name__ == '__main__':
    test()
