import torch
import torch.nn as nn
import sys
import os
from VMamba.classification.models.vmamba import VSSM

class VMambaClassifier(nn.Module):
    def __init__(self, num_classes=6, model_type='tiny'):
        """
        Mô hình phân loại dựa trên backbone VMamba Small
        """
        super(VMambaClassifier, self).__init__()
        
        # Khởi tạo backbone VMamba
        # Thông số dưới đây mô phỏng cấu hình cho các biến thể phổ biến của VMamba (như VSSM tiny/small/base)
        self.backbone = VSSM(
            depths=[2, 2, 12, 2], dims=96, drop_path_rate=0.3, 
            patch_size=4, in_chans=3, num_classes=1000, 
            ssm_d_state=64, ssm_ratio=1.0, ssm_dt_rank="auto", ssm_act_layer="gelu",
            ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0, 
            ssm_init="v2", forward_type="m0_noz", 
            mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
            patch_norm=True, norm_layer="ln",
            downsample_version="v3", patchembed_version="v2", 
            use_checkpoint=False, posembed=False, imgsize=224
        )

        # Thay thế lớp classifier (fully connected) cuối cùng để phù hợp với số lượng class của chúng ta
        in_features = self.backbone.head.in_features
        self.backbone.head = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.backbone(x)

if __name__ == "__main__":
    # Khởi tạo model với 6 class
    model = VMambaClassifier(num_classes=6, model_type='tiny')
    
    # Tạo tensor giả lập 1 batch gồm 2 ảnh (batch_size=2, channels=3, H=224, W=224)
    dummy_input = torch.randn(2, 3, 224, 224)
    
    # Chạy thử model
    outputs = model(dummy_input)
    
    print(f"Kích thước tensor đầu ra: {outputs.shape}") # Mong đợi: [2, 6]
    print(f"Mô hình hoạt động bình thường!")
