# attribute_gcn.py
# Graph Convolutional Network cho khai thác mối quan hệ giữa các thuộc tính
# Được đặt sau Text Encoder, trước bước nén (text_compress)
#
# Ý tưởng: Trong PAR, các thuộc tính có mối tương quan mạnh
# (ví dụ: "mặc váy" ↔ "nữ giới", "đội mũ bảo hiểm" ↔ "đi xe máy")
# GCN cho phép mỗi attribute embedding "nhìn thấy" và trao đổi thông tin
# với các thuộc tính liên quan thông qua đồ thị với learnable adjacency matrix.

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvLayer(nn.Module):
    """
    Một lớp Graph Convolution cơ bản.
    
    Công thức: h' = Â @ h @ W
    Trong đó:
    - Â: Normalized adjacency matrix (num_nodes, num_nodes)
    - h: Node features (B, num_nodes, d_in)
    - W: Weight matrix (d_in, d_out)
    
    Args:
        d_in (int): Chiều đặc trưng đầu vào
        d_out (int): Chiều đặc trưng đầu ra
    """
    def __init__(self, d_in, d_out):
        super().__init__()
        self.weight = nn.Linear(d_in, d_out, bias=False)
        self.bias = nn.Parameter(torch.zeros(d_out))
        
    def forward(self, adj, x):
        """
        Args:
            adj (Tensor): Normalized adjacency matrix (num_nodes, num_nodes)
            x (Tensor): Node features (B, num_nodes, d_in)
            
        Returns:
            out (Tensor): Updated node features (B, num_nodes, d_out)
        """
        # 1. Linear transform: (B, N, d_in) → (B, N, d_out)
        x = self.weight(x)
        
        # 2. Message passing: Â @ h → (B, N, d_out)
        # adj: (N, N), x: (B, N, d_out)
        out = torch.matmul(adj, x)
        
        # 3. Add bias
        out = out + self.bias
        
        return out


class AttributeGCN(nn.Module):
    """
    GCN trên đồ thị 57 attribute nodes.
    
    Khai thác mối tương quan giữa các thuộc tính người đi bộ thông qua
    Graph Convolution với learnable adjacency matrix.
    
    Thiết kế:
    - Adjacency matrix A_raw (num_attrs, num_attrs): learnable, khởi tạo 
      đường chéo lớn (gần identity) → ban đầu mỗi node chủ yếu giữ thông tin
      của chính mình, dần dần học thêm mối quan hệ với các node khác
    - Qua sigmoid → A ∈ [0, 1] → normalize → Â (symmetric normalization)
    - Mỗi GCN layer có residual connection + pre-norm để gradient ổn định
    
    Input:  (B, num_attrs, d_model) = (B, 57, 768)
    Output: (B, num_attrs, d_model) = (B, 57, 768)
    
    Args:
        num_attrs (int): Số lượng thuộc tính (nodes). Mặc định: 57
        d_model (int): Chiều embedding. Mặc định: 768
        num_layers (int): Số lớp GCN. Mặc định: 2
        dropout (float): Dropout rate. Mặc định: 0.1
    """
    def __init__(self, num_attrs=57, d_model=768, num_layers=2, dropout=0.1):
        super().__init__()
        self.num_attrs = num_attrs
        self.d_model = d_model
        self.num_layers = num_layers
        
        # 1. Learnable adjacency matrix
        # Khởi tạo: đường chéo = 2.0 (sau sigmoid ≈ 0.88), ngoài đường chéo = 0.0 (sau sigmoid = 0.5)
        # → Ban đầu self-loop mạnh, cross-connection vừa phải
        A_init = torch.zeros(num_attrs, num_attrs)
        A_init.fill_diagonal_(2.0)  # Self-connection mạnh
        self.A_raw = nn.Parameter(A_init)
        
        # 2. GCN layers (giữ nguyên chiều d_model)
        self.gcn_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.activations = nn.ModuleList()
        
        for _ in range(num_layers):
            self.gcn_layers.append(GraphConvLayer(d_model, d_model))
            self.norms.append(nn.LayerNorm(d_model))
            self.activations.append(nn.GELU())
        
        # 3. Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 4. Output norm
        self.output_norm = nn.LayerNorm(d_model)
        
    def _normalize_adj(self, A):
        """
        Symmetric normalization: Â = D^(-1/2) @ A @ D^(-1/2)
        
        Đây là cách chuẩn hóa kinh điển trong GCN (Kipf & Welling, 2017).
        Đảm bảo scale ổn định khi số lượng edges tăng.
        
        Args:
            A (Tensor): Adjacency matrix (N, N), giá trị ∈ [0, 1]
            
        Returns:
            A_hat (Tensor): Normalized adjacency matrix (N, N)
        """
        # Degree matrix
        D = A.sum(dim=-1)  # (N,)
        D_inv_sqrt = torch.pow(D.clamp(min=1e-8), -0.5)  # (N,)
        
        # D^(-1/2) @ A @ D^(-1/2)
        # Tương đương: diag(D_inv_sqrt) @ A @ diag(D_inv_sqrt)
        A_hat = D_inv_sqrt.unsqueeze(-1) * A * D_inv_sqrt.unsqueeze(0)
        
        return A_hat
    
    def get_adjacency_matrix(self):
        """
        Trả về adjacency matrix sau sigmoid (dùng để phân tích/visualize).
        
        Returns:
            A (Tensor): Soft adjacency matrix (num_attrs, num_attrs), giá trị ∈ [0, 1]
        """
        return torch.sigmoid(self.A_raw)
    
    def forward(self, x):
        """
        Forward pass: Làm giàu attribute embeddings qua GCN.
        
        Args:
            x (Tensor): Attribute embeddings từ Text Encoder, shape (B, num_attrs, d_model)
            
        Returns:
            x (Tensor): Enriched attribute embeddings, shape (B, num_attrs, d_model)
        """
        # 1. Tính adjacency matrix
        A = torch.sigmoid(self.A_raw)       # (N, N), giá trị ∈ [0, 1]
        A_hat = self._normalize_adj(A)      # (N, N), normalized
        
        # 2. Chạy qua các GCN layers với residual + pre-norm
        for gcn_layer, norm, act in zip(self.gcn_layers, self.norms, self.activations):
            residual = x
            x = norm(x)                         # Pre-Norm
            x = gcn_layer(A_hat, x)             # GCN: Â @ h @ W + b
            x = act(x)                          # GELU activation
            x = self.dropout(x)                 # Dropout
            x = residual + x                    # Residual connection
        
        # 3. Output norm
        x = self.output_norm(x)
        
        return x
