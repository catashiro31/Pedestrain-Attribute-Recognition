import torch
import triton
if not hasattr(triton, 'set_allocator'):
    triton.set_allocator = lambda *args, **kwargs: None
import os
from config import get_kaggle_config
from model.climp_par import CLIMPPAR

config = get_kaggle_config()
config.batch_size = 8
model = CLIMPPAR(config).cuda()
model.eval()
print("Model initialized on GPU.")

x = torch.randn(8, 3, 256, 128).cuda()
bg = torch.randn(8, 3, 256, 128).cuda()

print("Inputs created on GPU. Running forward pass...")
with torch.no_grad():
    out, _ = model(x, bg)
print(f"Forward pass success! Shape: {out.shape}")
