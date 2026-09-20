import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvolution(nn.Module):
    """
    Lớp GCN cơ bản (Graph Convolutional Layer).
    Z = A * X * W + b
    """
    def __init__(self, in_features, out_features, dropout=0.1):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        # x: (N_attrs, in_features)
        # adj: (N_attrs, N_attrs)
        
        # 1. Chiếu sang không gian mới (Linear projection)
        support = torch.matmul(x, self.weight)
        support = self.dropout(support)
        
        # 2. Nhân với ma trận kề để lấy thông tin từ các node lân cận
        output = torch.matmul(adj, support)
        return output + self.bias


class SemanticGCN(nn.Module):
    """
    Mô hình GCN xử lý Semantic Embeddings.
    Tự động xây dựng ma trận kề (Adjacency Matrix) dựa trên độ tương đồng Cosine
    giữa các Text Embeddings, sau đó áp dụng 1 lớp GCN.
    """
    def __init__(self, embed_dim=768, dropout=0.1):
        super(SemanticGCN, self).__init__()
        # Chỉ dùng 1 lớp GCN như yêu cầu để tránh over-smoothing
        self.gcn1 = GraphConvolution(embed_dim, embed_dim, dropout)

    def forward(self, text_features):
        # text_features shape: (N_attrs, embed_dim)
        
        # 1. Xây dựng Adjacency Matrix (Ma trận tương quan) động từ chính Text Encoder
        # Chuẩn hóa L2 để tính Cosine Similarity
        norm_features = F.normalize(text_features, dim=-1)
        # Ma trận tương quan A: (N_attrs, N_attrs)
        adj = torch.matmul(norm_features, norm_features.T)
        
        # 2. Chuẩn hóa ma trận kề (Row-wise Softmax) để đảm bảo tổng mỗi hàng = 1
        # Điều này giúp GCN không làm bùng nổ giá trị của vector
        adj = F.softmax(adj, dim=-1)
        
        # 3. Đưa qua lớp GCN
        out = self.gcn1(text_features, adj)
        out = F.relu(out)
        
        # 4. Residual Connection (Cộng skip-connection)
        # Rất quan trọng để giữ lại đặc trưng gốc của từng thuộc tính,
        # GCN chỉ đóng vai trò bổ sung thêm ngữ cảnh từ các thuộc tính khác.
        out = out + text_features
        
        return out
