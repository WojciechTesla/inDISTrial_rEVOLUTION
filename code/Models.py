import torch.nn as nn
import torch.nn.functional as F
import torch
from typing import Optional

class SiameseNetwork(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int = 128, num_classes: Optional[int] = None):
        super(SiameseNetwork, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )
        self.classifier = nn.Linear(embedding_dim, num_classes) if num_classes else None

    def forward_once(self, x):
        x = self.encoder(x)
        return F.normalize(x, p=2, dim=1)

    def forward(self, x1, x2=None, mode='siamese'):
        if mode == 'siamese':
            # print("Siamese mode activated")
            return self.forward_once(x1), self.forward_once(x2)
        elif mode == 'classify':
            # print("Classification mode activated")
            emb = self.forward_once(x1)
            return self.classifier(emb)

class ContrastiveLoss(nn.Module):
    def __init__(self, margin: float = 1.7):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, emb1, emb2, label):
        d = F.pairwise_distance(emb1, emb2)
        sim_weight = 1.0  # you can tune this
        dissim_weight = 2.0  # increase this to penalize more
        loss = 0.5 * (label * sim_weight * d**2 + (1 - label) * dissim_weight * F.relu(self.margin - d)**2)
        return loss.mean()
    
def combined_loss(emb1, emb2, logits1, logits2, pair_label, y1, y2,
                  contrastive_margin=1.7, alpha=1.0, beta=0.5):
    contrastive = ContrastiveLoss(margin=contrastive_margin)(emb1, emb2, pair_label)
    ce_loss = nn.CrossEntropyLoss()
    classification = ce_loss(logits1, y1) + ce_loss(logits2, y2)
    return alpha * contrastive + beta * classification
    
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        features = F.normalize(features, dim=1)
        batch_size = features.size(0)

        mask = torch.eq(labels.view(-1, 1), labels.view(1, -1)).float().to(device)
        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        exp_logits = torch.exp(logits) * (1 - torch.eye(batch_size).to(device))
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        return -mean_log_prob_pos.mean()