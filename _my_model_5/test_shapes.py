import torch
from config import get_kaggle_config
from model import CLIMPPAR

def test():
    config = get_kaggle_config()
    config.resnet_pretrained = False
    
    # Giả lập input
    B = 2
    images = torch.randn(B, 3, 448, 448)
    cached_text_features = torch.randn(57, 768)
    
    model = CLIMPPAR(config)
    
    print("Testing CLIMP-PAR v3...")
    logits, _ = model(images, cached_text_features=cached_text_features)
    
    print("Output shape:", logits.shape)
    assert logits.shape == (B, 57), f"Expected (2, 57), got {logits.shape}"
    print("Test passed!")

if __name__ == '__main__':
    test()
