import torch.nn as nn
import torch.nn.functional as F

class SiameseNetwork(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int = 128):
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

    def forward_once(self, x):
        x = self.encoder(x)
        return F.normalize(x, p=2, dim=1)

    def forward(self, x1, x2):
        emb1 = self.forward_once(x1)
        emb2 = self.forward_once(x2)
        return emb1, emb2

class ContrastiveLoss(nn.Module):
    def __init__(self, margin: float = 0.5):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, emb1, emb2, label):
        d = F.pairwise_distance(emb1, emb2)
        sim_weight = 1.0  # you can tune this
        dissim_weight = 2.0  # increase this to penalize more
        loss = 0.5 * (label * sim_weight * d**2 + (1 - label) * dissim_weight * F.relu(self.margin - d)**2)
        return loss.mean()