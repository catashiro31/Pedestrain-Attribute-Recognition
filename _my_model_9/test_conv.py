import torch
import torch.nn.functional as F
try:
    print(torch.backends.cudnn.version())
    x = torch.randn(1, 3, 256, 128).cuda()
    w = torch.randn(16, 3, 3, 3).cuda()
    b = torch.randn(16).cuda()
    y = F.conv2d(x, w, b, padding=1)
    print("Conv2d success:", y.shape)
except Exception as e:
    print("Conv2d failed:", e)
