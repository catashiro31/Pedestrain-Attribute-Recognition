# losses.py
# Hàm loss cho CLIMP-PAR
# Kết hợp Contrastive Loss (CLIP-style) + BCE Loss cho multi-label classification

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss cho multi-label classification.
    Dựa trên bài báo: "Asymmetric Loss For Multi-Label Classification"
    """
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-5):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits, labels):
        # Tính xác suất
        probs = torch.sigmoid(logits)
        xs_pos = probs
        xs_neg = 1 - probs

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg_shifted = (xs_neg + self.clip).clamp(max=1)
        else:
            xs_neg_shifted = xs_neg

        # Tách Gradient cho Focal Weight để tránh gradient bùng nổ (theo Official ASL)
        with torch.no_grad():
            pt0 = xs_pos * labels
            pt1 = xs_neg_shifted * (1 - labels)
            pt = pt0 + pt1
            
            one_sided_gamma = self.gamma_pos * labels + self.gamma_neg * (1 - labels)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)

        # Tính Base Loss an toàn cho FP16:
        # 1. Positive Loss: Dùng BCEWithLogitsLoss (log-sum-exp trick) thay vì log(sigmoid) để tránh NaN gradient
        bce_pos = F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits), reduction='none')
        los_pos = labels * bce_pos

        # 2. Negative Loss: Do xs_neg_shifted >= clip (0.05) nên log rất an toàn (gradient cực đại là 1/0.05 = 20)
        # Dùng clamp 1e-4 để bảo vệ thêm nếu clip = 0
        los_neg = (1 - labels) * -torch.log(xs_neg_shifted.clamp(min=1e-4))
        
        loss = (los_pos + los_neg) * one_sided_w

        # TỐI ƯU: Tính tổng theo class (dim=1), sau đó tính trung bình theo batch (dim=0)
        return loss.sum(dim=1).mean()

class CLIMPPARLoss(nn.Module):
    """
    Loss tổng hợp cho CLIMP-PAR:
    1. BCE Loss: Binary Cross-Entropy cho multi-label attribute prediction
       - Mỗi thuộc tính là bài toán phân loại nhị phân độc lập
       - logits_per_image[i, j] → xác suất ảnh i có thuộc tính j
    
    Giải thích:
    - Với PAR (multi-label), mỗi ảnh có thể có NHIỀU thuộc tính đồng thời
    - BCE phù hợp hơn Contrastive Loss (InfoNCE) vì InfoNCE giả định 
      mỗi ảnh chỉ match 1 text → không đúng cho multi-label
    """
    
    def __init__(self, loss_type='bce', asl_gamma_neg=4, asl_gamma_pos=1, asl_clip=0.05):
        super().__init__()
        self.loss_type = loss_type
        self.asymmetric_loss = AsymmetricLoss(gamma_neg=asl_gamma_neg, gamma_pos=asl_gamma_pos, clip=asl_clip)
        
    def forward(self, logits_per_image, labels):
        """
        Args:
            logits_per_image (Tensor): (B, N_attrs) — logits từ cosine similarity có scale
            labels (Tensor): (B, N_attrs) — ground truth multi-hot labels, 0 hoặc 1
            
        Returns:
            loss (Tensor): scalar loss
        """
        if self.loss_type == 'bce':
            return self.bce_loss(logits_per_image, labels)
        elif self.loss_type == 'weighted_bce':
            return self.weighted_bce_loss(logits_per_image, labels)
        elif self.loss_type == 'asl':
            return self.asl_loss(logits_per_image, labels)
        elif self.loss_type == 'hybrid_asl_bce':
            return self.hybrid_asl_bce(logits_per_image, labels)
        elif self.loss_type == 'hybrid_loss':
            return self.hybrid_loss(logits_per_image, labels)
        else:
            return self.bce_loss(logits_per_image, labels)
            
    def bce_loss(self, logits, labels):
        """Binary Cross-Entropy with Logits."""
        return F.binary_cross_entropy_with_logits(logits, labels)
    
    def weighted_bce_loss(self, logits, labels):
        """
        Weighted BCE: tự động cân bằng positive/negative ratio.
        Quan trọng vì PAR thường có class imbalance nặng
        (ví dụ: 90% người không đội mũ → negative chiếm áp đảo).
        """
        # Tính pos_weight = num_neg / num_pos cho mỗi thuộc tính
        with torch.no_grad():
            pos_count = labels.sum(dim=0).clamp(min=1)       # (N_attrs,)
            neg_count = (labels.shape[0] - pos_count).clamp(min=1)
            pos_weight = neg_count / pos_count                # (N_attrs,)
        
        return F.binary_cross_entropy_with_logits(
            logits, labels, pos_weight=pos_weight
        )
    
    def asl_loss(self, logits, labels):
        """Asymmetric Loss."""
        return self.asymmetric_loss(logits, labels)
    
    def hybrid_loss(self, logits, labels, alpha=0.8, beta=0.2):
        """Hybrid loss = alpha * BCE + beta * Weighted BCE"""
        return alpha * self.bce_loss(logits, labels) + beta * self.weighted_bce_loss(logits, labels)

    def hybrid_asl_bce(self, logits, labels, alpha=0.6, beta=0.4):
        """
        Hybrid loss = alpha * BCE + beta * ASL
        Đồng bộ thang đo: Tính BCE theo cách sum(dim=1).mean() giống như ASL đang sử dụng.
        """
        bce_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
        bce_loss_scaled = bce_loss.sum(dim=1).mean()
        asl_loss = self.asymmetric_loss(logits, labels)
        
        return alpha * bce_loss_scaled + beta * asl_loss

class DisentanglementLoss(nn.Module):
    """
    Hàm Loss phụ trợ ép mạng lưới thực hiện đúng triết lý:
    1. Attribute Consistency Loss: disentangled phải độc lập với môi trường.
    2. Domain Grounding Loss: style_emb phải thật sự là thông tin môi trường.
    3. Orthogonal Loss: disentangled và style_emb phải trực giao.
    """
    def __init__(self, w_cons=1.0, w_ground=1.0, w_ortho=1.0):
        super().__init__()
        self.w_cons = w_cons
        self.w_ground = w_ground
        self.w_ortho = w_ortho
        
    def forward(self, disentangled, style_emb, bg_context):
        # 1. Attribute Consistency Loss
        # disentangled: (B, 57, 768). Tính phương sai theo Batch (ép các vector cùng thuộc tính giống nhau)
        loss_cons = disentangled.var(dim=0).mean()
        
        # 2. Domain Grounding Loss
        # Ép style_emb (B, 768) có độ tương đồng cao với bg_context (B, 768)
        cos_ground = F.cosine_similarity(style_emb, bg_context, dim=-1)
        loss_ground = (1.0 - cos_ground).mean()
        
        # 3. Orthogonal Loss
        # Ép disentangled và style_emb vuông góc với nhau
        cos_ortho = F.cosine_similarity(disentangled, style_emb.unsqueeze(1), dim=-1) # (B, 57)
        loss_ortho = (cos_ortho ** 2).mean()
        
        loss_total = self.w_cons * loss_cons + self.w_ground * loss_ground + self.w_ortho * loss_ortho
        return loss_total, loss_cons, loss_ground, loss_ortho