# losses.py
# Hàm loss cho CLIMP-PAR
# Kết hợp Contrastive Loss (CLIP-style) + BCE Loss cho multi-label classification

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
    
    def __init__(self, loss_type='bce'):
        super().__init__()
        self.loss_type = loss_type
        
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
        elif self.loss_type == 'hybrid_loss':
            return self.hybrid_loss(logits_per_image, label)
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
    
    def hybrid_loss(self, logits, labels, alpha=0.8, beta=0.2):
        """Hybrid loss = alpha * BCE + beta * Weighted BCE"""
        return alpha * self.bce_loss(logits, labels) + beta * self.weighted_bce_loss(logits, labels)