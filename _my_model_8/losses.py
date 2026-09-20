# losses.py
# Hàm loss cho CLIMP-PAR v8.1
# Bao gồm: BCE (cho multi-label classification), 
#          Global Image-Text Contrastive Loss,
#          Fine-grained Attribute Contrastive Loss

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for Multi-Label Classification
    Tập trung vào hard negatives và loại bỏ bớt gradient từ easy negatives.
    Giúp tối ưu hóa gián tiếp F1-score và đối phó với class imbalance.
    """
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, x, y):
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        pt0 = xs_pos * y
        pt1 = xs_neg * (1 - y)
        pt = pt0 + pt1
        one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
        one_sided_w = torch.pow(1 - pt, one_sided_gamma)

        loss *= one_sided_w
        return -loss.mean()


class CLIMPPARLoss(nn.Module):
    """
    Loss tổng hợp cho CLIMP-PAR v8.1:
    1. Base Loss: BCE hoặc Weighted BCE (cho multi-label prediction)
    2. Global Contrastive Loss: InfoNCE căn chỉnh toàn cục (Image - Text)
    3. Fine-grained Contrastive Loss: Max-similarity căn chỉnh chi tiết (Vision tokens - Attribute tokens)
    """
    
    def __init__(self, config=None):
        super().__init__()
        if config is not None:
            self.loss_type = getattr(config, 'loss_type', 'weighted_bce')
            self.alpha = getattr(config, 'global_contrastive_weight', 0.1)
            self.beta = getattr(config, 'finegrained_contrastive_weight', 0.1)
            init_temp = getattr(config, 'temperature_init', 0.07)
        else:
            self.loss_type = 'weighted_bce'
            self.alpha = 0.1
            self.beta = 0.1
            init_temp = 0.07
            
        if self.loss_type in ['asl', 'weighted_bce_asl']:
            self.asl_loss = AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05)
            
        # Learnable temperature parameter (giống CLIP)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / init_temp))
        
    def forward(self, logits_per_image, labels, features=None):
        """
        Args:
            logits_per_image (Tensor): (B, N_attrs) — logits chưa qua sigmoid
            labels (Tensor): (B, N_attrs) — ground truth multi-hot labels, 0 hoặc 1
            features (dict, optional): Chứa 'image_features', 'full_features', 'diff_features'
            
        Returns:
            loss (Tensor): scalar loss
        """
        # 1. Tính Base Classification Loss
        if self.loss_type == 'bce':
            base_loss = self.bce_loss(logits_per_image, labels)
        elif self.loss_type == 'asl':
            base_loss = self.asl_loss(logits_per_image, labels)
        elif self.loss_type == 'weighted_bce_asl':
            base_loss = self.weighted_bce_loss(logits_per_image, labels) + self.asl_loss(logits_per_image, labels)
        else:
            base_loss = self.weighted_bce_loss(logits_per_image, labels)
            
        total_loss = base_loss
        
        # Nếu không có features, chỉ dùng base_loss (ví dụ lúc warmup hoặc test)
        if features is None or (self.alpha == 0 and self.beta == 0):
            return total_loss
            
        # Trích xuất các feature
        image_features = features['image_features'] # (B, 32, 768)
        full_features = features['full_features']   # (B, 768)
        diff_features = features['diff_features']   # (B, 57, 768)
        
        # Giới hạn giá trị của logit_scale để tránh overflow (tối đa ~100)
        logit_scale = torch.clamp(self.logit_scale.exp(), max=100)
        
        # 2. Tính Global Image-Text Contrastive Loss (InfoNCE)
        if self.alpha > 0:
            global_loss = self.global_contrastive_loss(image_features, full_features, logit_scale)
            total_loss = total_loss + self.alpha * global_loss
            
        # 3. Tính Fine-grained Attribute Contrastive Loss
        if self.beta > 0:
            finegrained_loss = self.finegrained_contrastive_loss(image_features, diff_features, labels, logit_scale)
            total_loss = total_loss + self.beta * finegrained_loss
            
        return total_loss
    
    def bce_loss(self, logits, labels):
        """Binary Cross-Entropy with Logits."""
        return F.binary_cross_entropy_with_logits(logits, labels)
    
    def weighted_bce_loss(self, logits, labels):
        """
        Weighted BCE: tự động cân bằng positive/negative ratio.
        Quan trọng vì PAR thường có class imbalance nặng
        """
        with torch.no_grad():
            pos_count = labels.sum(dim=0).clamp(min=1)       # (N_attrs,)
            neg_count = (labels.shape[0] - pos_count).clamp(min=1)
            pos_weight = neg_count / pos_count                # (N_attrs,)
        
        return F.binary_cross_entropy_with_logits(
            logits, labels, pos_weight=pos_weight
        )
        
    def global_contrastive_loss(self, image_features, text_features, logit_scale):
        """
        InfoNCE Loss căn chỉnh toàn bộ ảnh với toàn bộ câu văn.
        image_features: (B, 32, 768) -> GAP -> (B, 768)
        text_features: (B, 768)
        """
        device = image_features.device
        B = image_features.shape[0]
        
        # Global Average Pooling cho hình ảnh
        image_global = image_features.mean(dim=1) # (B, 768)
        
        # L2 Normalize
        image_global = F.normalize(image_global, dim=-1)
        text_global = F.normalize(text_features, dim=-1)
        
        # Cosine similarity x Temperature
        logits_per_image = logit_scale * image_global @ text_global.T # (B, B)
        logits_per_text = logits_per_image.T                          # (B, B)
        
        # Ground truth là đường chéo chính (i == j)
        labels = torch.arange(B, dtype=torch.long, device=device)
        
        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)
        
        return (loss_i + loss_t) / 2
        
    def finegrained_contrastive_loss(self, image_features, attribute_features, labels, logit_scale):
        """
        Loss căn chỉnh chi tiết: ép các vùng của ảnh (vision tokens) khớp với các thuộc tính cụ thể.
        image_features: (B, 32, 768)
        attribute_features: (B, 57, 768)
        labels: (B, 57)
        """
        # L2 Normalize theo chiều feature (768)
        img_feat_norm = F.normalize(image_features, dim=-1)      # (B, 32, 768)
        attr_feat_norm = F.normalize(attribute_features, dim=-1) # (B, 57, 768)
        
        # Tính dot-product similarity matrix giữa các vùng của ảnh và các thuộc tính
        # B x 32 x 768 @ B x 768 x 57 -> B x 32 x 57
        sim_matrix = torch.bmm(img_feat_norm, attr_feat_norm.transpose(1, 2))
        
        # Áp dụng logit_scale
        sim_matrix = sim_matrix * logit_scale
        
        # Lấy vùng có similarity cao nhất đối với mỗi thuộc tính (Max Pooling dọc chiều 32)
        # Giả định: 1 thuộc tính sẽ xuất hiện ở 1 (hoặc 1 vài) token cục bộ, max() giúp tìm token đó.
        max_sim, _ = torch.max(sim_matrix, dim=1) # (B, 57)
        
        # Sử dụng BCE loss trên similarity này, ép nó hội tụ về labels thực tế
        # Nếu label=1, similarity phải cao; label=0, similarity phải thấp.
        # Ở đây ta dùng Weighted BCE cho cân bằng class.
        with torch.no_grad():
            pos_count = labels.sum(dim=0).clamp(min=1)       
            neg_count = (labels.shape[0] - pos_count).clamp(min=1)
            pos_weight = neg_count / pos_count
            
        loss = F.binary_cross_entropy_with_logits(max_sim, labels, pos_weight=pos_weight)
        
        return loss
