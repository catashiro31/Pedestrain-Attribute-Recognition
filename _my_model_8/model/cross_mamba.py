import torch
import torch.nn as nn
import torch.nn.functional as F

class MinimalSSM(nn.Module):
    """
    Triển khai cốt lõi của Selective State Space (SSM) bằng PyTorch thuần.
    Phù hợp cho sequence length nhỏ (như 32 hoặc 57) mà không cần build thư viện C/CUDA.
    """
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        self.dt_proj = nn.Linear(d_model, d_model)
        self.x_proj = nn.Linear(d_model, d_state * 2)
        
        # Khởi tạo A và D
        self.A_log = nn.Parameter(torch.log(torch.ones(d_model, d_state)))
        self.D = nn.Parameter(torch.ones(d_model))
        
    def forward(self, x):
        B, L, D = x.shape
        
        # Selective parameters
        dt = F.softplus(self.dt_proj(x)) # (B, L, D)
        x_proj = self.x_proj(x)          # (B, L, 2 * d_state)
        B_mat, C_mat = torch.split(x_proj, self.d_state, dim=-1) # Cả hai có shape (B, L, N)
        
        A = -torch.exp(self.A_log) # (D, N)
        
        y = torch.zeros(B, L, D, device=x.device, dtype=x.dtype)
        h = torch.zeros(B, D, self.d_state, device=x.device, dtype=x.dtype)
        
        # Parallel scan giả lập (Sequential cho L nhỏ thì cực nhanh)
        for i in range(L):
            dt_i = dt[:, i].unsqueeze(-1)          # (B, D, 1)
            dA = torch.exp(dt_i * A)               # (B, D, N)
            dB = dt_i * B_mat[:, i].unsqueeze(1)   # (B, D, N)
            
            h = dA * h + dB * x[:, i].unsqueeze(-1)
            y[:, i] = (h * C_mat[:, i].unsqueeze(1)).sum(dim=-1)
            
        y = y + x * self.D
        return y


class CrossModalMambaBlock(nn.Module):
    """
    Thiết kế lai: 2 nhánh Mamba độc lập nhưng chia sẻ chung một Gate (Gating chéo)
    được tính toán từ tích Element-wise của Vision và Text.
    """
    def __init__(self, d_model):
        super().__init__()
        self.norm_z = nn.LayerNorm(d_model)
        self.norm_t = nn.LayerNorm(d_model)
        
        # Central Shared Gate
        self.gate_proj = nn.Linear(d_model, d_model)
        self.gate_act = nn.SiLU()
        
        # Vision Branch
        self.z_proj1 = nn.Linear(d_model, d_model)
        self.z_conv = nn.Conv1d(d_model, d_model, kernel_size=4, padding=3, groups=d_model)
        self.z_act = nn.SiLU()
        self.z_ssm = MinimalSSM(d_model)
        self.z_proj2 = nn.Linear(d_model, d_model)
        
        # Text Branch
        self.t_proj1 = nn.Linear(d_model, d_model)
        self.t_conv = nn.Conv1d(d_model, d_model, kernel_size=4, padding=3, groups=d_model)
        self.t_act = nn.SiLU()
        self.t_ssm = MinimalSSM(d_model)
        self.t_proj2 = nn.Linear(d_model, d_model)

    def forward(self, z, t):
        """
        z: Vision features (B, L, D)
        t: Text features (B, L, D)
        """
        # 1. Norm
        z_norm = self.norm_z(z)
        t_norm = self.norm_t(t)
        
        # 2. Central Gate
        m = z_norm * t_norm
        gate = self.gate_act(self.gate_proj(m))
        
        # 3. Vision Branch
        z_ = self.z_proj1(z_norm)
        # Conv1d cần shape (B, D, L)
        z_ = z_.transpose(1, 2)
        z_ = self.z_conv(z_)[..., :z.shape[1]] # Lấy đủ L token (bỏ phần padding dư)
        z_ = z_.transpose(1, 2)
        z_ = self.z_act(z_)
        z_ = self.z_ssm(z_)
        # Element-wise multiplication với Shared Gate
        z_out = self.z_proj2(z_ * gate) + z
        
        # 4. Text Branch
        t_ = self.t_proj1(t_norm)
        t_ = t_.transpose(1, 2)
        t_ = self.t_conv(t_)[..., :t.shape[1]]
        t_ = t_.transpose(1, 2)
        t_ = self.t_act(t_)
        t_ = self.t_ssm(t_)
        # Element-wise multiplication với Shared Gate
        t_out = self.t_proj2(t_ * gate) + t
        
        return z_out, t_out
