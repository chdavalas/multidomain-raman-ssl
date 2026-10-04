import torch
from torch import nn
import logging

logger = logging.getLogger(__name__)

class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim)
        )
        
    def forward(self, x):
        return self.proj(x)  # Take CLS token
    
class LinearClassifier(nn.Module):
    def __init__(self, input_dim, out_dim=2):
        """
        Pure linear classification head for Siamese network.
        """
        super().__init__()
        # 2 * input_dim because we concatenate x1 and x2
        self.net = nn.Linear(2 * input_dim, out_dim)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=-1)  # Concatenate embeddings
        return self.net(x)

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, out_dim=2, dropout=0.2):
        """
        Args:
            input_dim (int): Dimension of each embedding (before concat).
            hidden_dim (int): Hidden layer size.
            out_dim (int): Output classes (2 for binary classification).
            dropout (float): Dropout probability.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=-1)  # Concatenate embeddings
        return self.net(x)

class SiameseNet(nn.Module):
    def __init__(self, encoder_a, encoder_b=None, projection_dim=None, concat_classify=True, cls_token=True, out_dim=2, head_type="mlp"):
        
        super().__init__()
        self.out_dim=out_dim
        self.encoder_a = encoder_a
        self.encoder_b = encoder_b if encoder_b is not None else encoder_a
        self.projection_dim = projection_dim
        self.concat_classify = concat_classify
        
        self.proj_a = ProjectionHead(encoder_a.dim, projection_dim)
        self.proj_b = ProjectionHead(encoder_b.dim if encoder_b else encoder_a.dim, projection_dim)

        self.cls_token = cls_token
        
        if cls_token:
            logger.info("Using cls_token.")
        else:
            logger.info("No cls_toekn used.")
            
        # --- NEW LOGIC HERE ---
        if head_type == "linear":
            self.classifier = LinearClassifier(input_dim=projection_dim, out_dim=self.out_dim)
            logger.info("Using Pure Linear Classification Head.")
        else:
            self.classifier = MLPClassifier(input_dim=projection_dim, out_dim=self.out_dim)
            logger.info("Using MLP Classification Head.")

    def forward(self, x1, x2):

        z1 = self.encoder_a(x1)
        z2 = self.encoder_b(x2)

        if self.cls_token:
            z1 = z1[:, 0, :]
            z2 = z2[:, 0, :]
        else:
            z1 = torch.mean(z1, dim=1)
            z2 = torch.mean(z2, dim=1)

        out1 = self.proj_a(z1)
        out2 = self.proj_b(z2)

        # --- UPDATED VARIABLE NAME ---
        classifier_out = self.classifier(out1, out2)
        
        return out1, out2, classifier_out