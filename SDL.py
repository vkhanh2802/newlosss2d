import torch
import torch.nn as nn
import torch.nn.functional as F
   
class SDL(nn.Module):
    def __init__(self, num_class=7, dim=768, k=2, size=32):
        super(SDL, self).__init__()
        self.dim = dim
        self.k = k
        self.num_class = num_class
        self.Queue = torch.nn.Parameter(torch.rand(num_class, size, dim),requires_grad=False)
        self.Probe = torch.nn.Parameter(torch.rand(num_class, size, num_class),requires_grad=False)

    def cacu_cosine_similarity(self, Q, x):
        # Q: [num_class, size, dim]
        # x: [dim]
        # 对Q和x进行归一化（转换为单位向量）
        
        x = x.expand(Q.shape[0], Q.shape[1], -1)
        Q_normalized = F.normalize(Q, p=2, dim=2)
        x_normalized = F.normalize(x, p=2, dim=2)
        similarities = F.cosine_similarity(Q_normalized, x_normalized, dim=2)
        # 选出相似度最大的k个
        topk_similarities, topk_indices = similarities.topk(self.k, dim=1)
        return topk_similarities, topk_indices

    def update(self, x, prob, label):
        argmax = label
        # x_prob = F.one_hot(label, self.num_class)
        x_prob = torch.nn.functional.softmax(prob, dim=1)
        # x: [batch_size, dim]
        # argmax: [batch_size]
        for i in range(x.shape[0]):
            queue = torch.cat(
                (self.Queue[argmax[i]], x[i].unsqueeze(0)), dim=0)[1:]
            self.Queue[argmax[i]] = queue
            probe = torch.cat(
                (self.Probe[argmax[i]], x_prob[i].unsqueeze(0)), dim=0)[1:]
            self.Probe[argmax[i]] = probe
            
    @torch.no_grad()
    def forward(self, x, probe, label):
        # x: [batch_size, dim]
        # probe: [batch_size, 7]
        # 更新Queue和Probe
        
        x_probe = probe.detach()
        probe = torch.zeros_like(x_probe, device=probe.device)  # to return
        for i in range(x.shape[0]):
            topk_similarities, topk_indices = self.cacu_cosine_similarity(
                self.Queue, x[i])
            
            # 跟据相似度，更新probe
            p = topk_similarities.unsqueeze(
                2) * self.Probe[torch.arange(self.num_class).unsqueeze(1), topk_indices]
            p = p.reshape(-1, self.num_class)
            probe[i] = torch.sum(p, dim=0) / torch.sum(topk_similarities)
      
        return probe

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class S2DFinalLoss(nn.Module):
    def __init__(self, num_classes=8, feat_dim=768, 
                 gamma=0.01, theta=0.1, margin_rr=0.15, relabel_margin=0.35):
        super(S2DFinalLoss, self).__init__()
        
        self.num_classes = num_classes
        self.gamma = gamma
        self.theta = theta
        self.margin_rr = margin_rr
        self.relabel_margin = relabel_margin
        self.warmup_epochs = 10 #Khoe relabeling
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim) * 0.01)

    def forward(self, logits, features, labels, alphas, soft_anchors, epoch=0):
        # =========================================================
        # 1. ÉP KIỂU VỀ FLOAT32 (BẮT BUỘC ĐỂ CHỐNG NAN TRONG AMP)
        # =========================================================
        logits = logits.float()
        features = features.float()
        alphas = alphas.float().view(-1) # Đảm bảo alphas luôn là vector 1D: [B]
        
        if soft_anchors is not None:
            soft_anchors = soft_anchors.float()
        centers = self.centers.float()
        
        B = logits.size(0)
        probs = F.softmax(logits, dim=1)
        # =========================================================
        # 2. RELABELING
        # =========================================================
        pseudo_labels = labels.clone()
        if epoch >= self.warmup_epochs:
            max_probs, max_idx = torch.max(probs, dim=1)
            gt_probs = probs.gather(1, labels.view(-1, 1)).view(-1)
            
            # Logic của SCN: Đổi nhãn khi model rất tự tin vào nhãn khác (max_probs - gt_probs > margin)
            # VÀ mẫu đó bị đánh giá là nhiễu/kém quan trọng (alphas < giá trị trung bình)
            relabel_mask = ((max_probs - gt_probs) > self.relabel_margin) & (alphas < alphas.mean())
            
            # Ghi đè nhãn mới cho các mẫu thỏa mãn
            pseudo_labels = torch.where(relabel_mask, max_idx, labels)
            
        pseudo_labels = pseudo_labels.detach()
        # =========================================================
        # 3. UNCERTAINTY-AWARE LABEL DISTRIBUTION
        # =========================================================
        one_hot_labels = F.one_hot(pseudo_labels, num_classes=self.num_classes).float()
        
        if soft_anchors is not None:
            # Đảm bảo soft_anchors không có NaN
            soft_anchors = torch.nan_to_num(soft_anchors, nan=0.0) 
            target_dist = alphas.unsqueeze(1) * one_hot_labels + (1.0 - alphas.unsqueeze(1)) * soft_anchors
        else:
            target_dist = one_hot_labels # Fallback an toàn nếu tắt SDL

        # Kẹp cả min và max, đồng thời tránh việc tổng bằng 0
        target_dist = torch.clamp(target_dist, min=1e-7, max=1.0)
        dist_sum = target_dist.sum(dim=1, keepdim=True)
        # Kẹp dist_sum tránh chia cho 0
        dist_sum = torch.clamp(dist_sum, min=1e-7) 
        target_dist = target_dist / dist_sum
        # Kẹp target_dist lại để hàm Logarit không bao giờ chạm tới log(0)
        target_dist = torch.clamp(target_dist, min=1e-7, max=1.0)
        target_dist = target_dist / target_dist.sum(dim=1, keepdim=True)
        # =========================================================
        # 4. CLASSIFICATION LOSS (KL-Divergence)
        # =========================================================
        log_probs = F.log_softmax(logits, dim=1)
        sample_kl = F.kl_div(log_probs, target_dist, reduction='none').sum(dim=1)
        L_cls = sample_kl.mean()
        
        # =========================================================
        # 5. RANK REGULARIZATION LOSS (RR-Loss)
        # =========================================================
        sorted_alphas, _ = torch.sort(alphas, descending=True)
        M = max(1, B // 2)
        alpha_H = sorted_alphas[:M].mean()
        alpha_L = sorted_alphas[M:].mean()
        L_RR = torch.clamp(self.margin_rr - (alpha_H - alpha_L), min=0.0)
        
        # =========================================================
        # 6. DISCRIMINATIVE LOSS (STABLE COSINE VERSION)
        # =========================================================
        # Dùng eps=1e-4 để quá trình Backward tuyệt đối không bị chia cho 0
        feat_norm = F.normalize(features, p=2, dim=1, eps=1e-4)
        centers_norm = F.normalize(centers, p=2, dim=1, eps=1e-4)
        
        batch_centers = centers_norm[pseudo_labels]
        cos_dist = 1.0 - (feat_norm * batch_centers).sum(dim=1)
        pull_loss = (alphas * cos_dist).mean()
        
        sim_matrix = torch.matmul(centers_norm, centers_norm.T) 
        mask_diag = torch.eye(self.num_classes, device=sim_matrix.device)
        push_loss = torch.sum(torch.exp(sim_matrix) * (1 - mask_diag)) / (self.num_classes * (self.num_classes - 1))
        
        L_D = pull_loss + push_loss
        
        # =========================================================
        # TỔNG HỢP LOSS
        # =========================================================
        L_total = L_cls + (self.gamma * L_D) + (self.theta * L_RR)
        
        return L_total, pseudo_labels