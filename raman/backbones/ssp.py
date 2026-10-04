import torch
import torch.nn as nn
import torch.nn.functional as F
from raman.backbones.spectra_encoder import spectra_encoder
import logging
from raman.mae.mae import MaskedAutoEncoderLit

logger = logging.getLogger(__name__)

# Code based on the repo: 
# https://github.com/XiaqiongFan/DeepRaman.git
# MIT license; see THIRD_PARTY_LICENSES/LICENSE-DeepRaman.txt and LICENSE.

class SpatialPyramidPooling(nn.Module):
    def __init__(self, pool_list):
        super(SpatialPyramidPooling, self).__init__()
        self.pool_list = pool_list
        self.num_outputs_per_channel = sum([i * i for i in pool_list])
        
    def forward(self, x):
        # Input shape: (batch_size, channels, height, width)
        batch_size, channels, height, width = x.size()
        outputs = []
        
        for pool_num, num_pool_regions in enumerate(self.pool_list):
            # Calculate window size
            win_height = height // num_pool_regions
            win_width = width // num_pool_regions
            
            for iy in range(num_pool_regions):
                for ix in range(num_pool_regions):
                    # Calculate window coordinates
                    h1 = iy * win_height
                    h2 = min(h1 + win_height, height)
                    w1 = ix * win_width
                    w2 = min(w1 + win_width, width)
                    
                    # Extract window and pool
                    window = x[:, :, h1:h2, w1:w2]
                    pooled = F.adaptive_max_pool2d(window, (1, 1))
                    outputs.append(pooled.view(batch_size, -1))
        
        # Concatenate all pooled features
        return torch.cat(outputs, dim=1)

class SSPModel(nn.Module):
    def __init__(self, input_channels=2, dim=32, depth=4, heads=4, mlp_dim=512, dim_head=64, pos_embedding_type='sin_cos', dropout=0.1, 
                 emb_dropout=0.1, fc1_size=1024, out_dim=2, pool_list=[1, 2, 3, 4], siamese_type='cnn', pretrained_ckpt=None, freeze_encoder=False):
        super(SSPModel, self).__init__()
        self.siamese_type = siamese_type
        self.out_dim = out_dim  
        
        if self.siamese_type == 'spectra':
            logger.info("The SSP model is using spectra encoder!")
            if pretrained_ckpt is not None:
                logger.info(f"Loading pretrained MAE encoder from {pretrained_ckpt}")
                mae_model = MaskedAutoEncoderLit.load_from_checkpoint(pretrained_ckpt)
                self.net = mae_model.model.encoder
                if freeze_encoder:
                    logger.info("Freezing pretrained encoder weights.")
                    for p in self.net.parameters():
                        p.requires_grad = False
            else:            
                logger.info("Using randomly initialized spectra encoder.")
                self.net = spectra_encoder(
                        input_dim=input_channels,
                        dim=dim,
                        depth=depth,
                        heads=heads,
                        mlp_dim=mlp_dim,
                        dim_head=dim_head,
                        pos_embedding_type=pos_embedding_type,
                        dropout=dropout,
                        emb_dropout=emb_dropout,
                        )
        elif self.siamese_type == 'cnn':
            logger.info("The SSP model is using Conv1D CNN!")
            self.conv1d = nn.Conv1d(input_channels, dim, kernel_size=7, padding=3)  
            self.bn1d = nn.BatchNorm1d(dim)
        
    
        self.conv2d = nn.Conv2d(
            in_channels=2,  
            out_channels=dim, 
            kernel_size=7, 
            padding=3  
        )
        self.bn2d = nn.BatchNorm2d(dim)
        
        self.spp = SpatialPyramidPooling([1, 2, 3, 4])
        
        spp_output_size = dim * sum(i*i for i in pool_list)
        
        self.fc1 = nn.Linear(spp_output_size, fc1_size)
        self.dropout = nn.Dropout(0.5)

        self.fc2 = nn.Linear(fc1_size, out_dim)
        
    def forward(self, x1, x2):
        if self.siamese_type == 'cnn':
            x1 = F.relu(self.bn1d(self.conv1d(x1)))
            x1 = F.max_pool1d(x1, kernel_size=3)
            
            x2 = F.relu(self.bn1d(self.conv1d(x2)))
            x2 = F.max_pool1d(x2, kernel_size=3)    
            
            x = torch.stack([x1, x2], dim=1) # (batch, 2, dim, seq_len//3)
        elif self.siamese_type == 'spectra':
            x1 = x1.permute(0,2,1)
            x2 = x2.permute(0,2,1)
            z1 = self.net(x1)
            z2 = self.net(x2)
            
            x = torch.stack([z1, z2], dim=1)
            x = x.permute(0,1,3,2)  # (batch, 2, dim, seq_len)
        
        x = F.relu(self.bn2d(self.conv2d(x))) 
        x = self.spp(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x