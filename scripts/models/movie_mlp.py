import torch.nn as nn
import torch

class MovieMLPModel(nn.Module):
    def __init__(self,
                features_dim,
                result_classes,
                mid_dim=16,
                mid_dim2=16
                ):
        super(MovieMLPModel, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(features_dim, mid_dim),
            nn.Tanh(),
            nn.Dropout(0.3),
            nn.Linear(mid_dim, mid_dim2),
            nn.Tanh(),
            nn.Dropout(0.3),
            nn.Linear(mid_dim2, result_classes),
        )

    def forward(self, x):
        out = self.classifier(x)
        return out
