import torch
from model.vision_encoder import VMambaVisionEncoder
model = VMambaVisionEncoder().cuda()
x = torch.randn(2, 3, 256, 128).cuda()
out = model(x)
print(out.shape)
