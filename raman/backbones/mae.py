import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from .spectra_encoder import spectra_encoder

# --- Pseudo Voigt with differentiable approximations ---
def pseudo_voigt(x, amplitudes, centers, sigma, gamma, eps=1e-12):
    B, N = x.shape
    _, M = centers.shape

    fG = 2.0 * torch.sqrt(torch.tensor(2.0 * math.log(2.0), device=x.device)) * sigma
    fL = 2.0 * gamma
    fV = 0.5346 * fL + torch.sqrt(0.2166 * fL**2 + fG**2 + eps)
    r = fL / fV
    eta = (1.36603 * r - 0.47719 * r**2 + 0.11116 * r**3).clamp(0.0, 1.0)

    x = x.unsqueeze(2)
    centers = centers.unsqueeze(1)
    fG = fG.unsqueeze(1)
    fL = fL.unsqueeze(1)
    eta = eta.unsqueeze(1)
    amp = amplitudes.unsqueeze(1)

    G = torch.exp(-4.0 * math.log(2.0) * ((x - centers) / (fG + eps))**2)
    G = G / (fG * math.sqrt(math.pi / (4.0 * math.log(2.0))))

    L = (fL / 2.0) / math.pi / ((x - centers)**2 + (fL / 2.0)**2 + eps)

    pv = amp * (eta * L + (1 - eta) * G)
    return pv.sum(dim=-1)

# --- Decoder with constrained parameters ---
class VoigtMLPDecoder(nn.Module):
    def __init__(self, latent_dim, n_peaks, hidden=512):
        super().__init__()
        self.n_peaks = n_peaks
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_peaks * 4)
        )

    def forward(self, z, w):
        B = z.size(0)
        z_m = z.mean(dim=1)
        params = self.net(z_m).view(B, self.n_peaks, 4)
        A_raw, mu_raw, sigma_raw, gamma_raw = params.unbind(-1)

        # --- Constrain parameters ---
        A = F.softplus(A_raw)
        
        # FIX 1: Force initialization to be narrow by subtracting 4.0 or 5.0
        # softplus(-5.0) is approx 0.006, which is a much better starting width for Raman peaks
        sigma = F.softplus(sigma_raw - 5.0) + 1e-5
        gamma = F.softplus(gamma_raw - 5.0) + 1e-5

        mu = torch.sigmoid(mu_raw)

        recon = pseudo_voigt(w, A, mu, sigma, gamma)
        recon = torch.cat([w.unsqueeze(-1), recon.unsqueeze(-1)], dim=-1)  # Add orignal wavenubers for second channel

        params = torch.stack([A, mu, sigma, gamma], dim=-1)  # Shape: (B, n_peaks, 4)

        return recon, params

class MaskedAutoencoder(nn.Module):
    def __init__(self, input_channels=2, d_model=4, encoder_depth=1, decoder_depth=1, nhead=1, mask_ratio=0.1, mlp_dim=256, dim_head=64, 
                 pos_embedding_type='sin_cos', dropout=0., emb_dropout=0., n_peaks=10, decoder_type='conv1d', use_fourier_pe=True, fourier_pe_dim=32):

        super(MaskedAutoencoder, self).__init__()
                 
        self.encoder = spectra_encoder(
            input_dim=input_channels,
            dim=d_model,
            depth=encoder_depth,
            heads=nhead,
            mlp_dim=mlp_dim,
            dim_head=dim_head,
            pos_embedding_type=pos_embedding_type,
            dropout=dropout,
            emb_dropout=emb_dropout,
            # use_fourier_pe=use_fourier_pe,
            # fourier_pe_dim=fourier_pe_dim
        )

        self.decoder_type = decoder_type
        self.voigt_decoder = VoigtMLPDecoder(
            latent_dim=d_model, 
            n_peaks=n_peaks
        )

        if decoder_type == 'conv1d':
            self.mae_decoder = nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.Conv1d(d_model, input_channels, kernel_size=5, padding=2)
            )
        else:
            self.mae_decoder = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, input_channels)
            )
        
        # Denoising Decoder (Identical architecture as normal one (MLP))
        self.denoising_decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, input_channels)
        )

        self.mask_ratio = mask_ratio
        self.d_model = d_model

        self.n_peaks = n_peaks

    def random_masking(self, T, num_masked, device):

        all_idx = torch.arange(T, device=device)
        window = torch.arange(num_masked, device=device)

        rand_add = torch.randperm(T-num_masked)[0].item()
        mask_idx = window+rand_add
        unmask_idx = all_idx[~torch.isin(all_idx, mask_idx)]

        return mask_idx, unmask_idx, "random"

    def peak_masking(self, x, i, T, num_masked, device):

        all_idx = torch.arange(T, device=device)

        max_peak = torch.argmax(x[i, :, -1]).item()
        
        left_side_size  = num_masked//2
        right_side_size = num_masked-num_masked//2

        left_side = max_peak-left_side_size
        right_side = max_peak+right_side_size
        
        if left_side<0:
            right_side+=abs(left_side)
            left_side=0

        if right_side>T-1:
            left_side-=right_side-(T-1)
            right_side=T-1


        mask_idx = torch.tensor([i for i in range(left_side, right_side)], device=device)
        unmask_idx = all_idx[~torch.isin(all_idx, mask_idx)]

        return mask_idx, unmask_idx, "peak"

    # Repeating method in order to have an option for reconstructing with voigt profiles
    def voigt_peak_masking(self, x, i, T, num_masked, device):
        mask_idx, unmask_idx, _ = self.peak_masking(x, i, T, num_masked, device)
        return mask_idx, unmask_idx, "voigt_peak"

    def front_masking(self, T, num_masked, device):

        all_idx = torch.arange(0, T, device=device)
        head_masked = T-num_masked

        unmask_idx = all_idx[:head_masked]

        mask_idx = torch.tensor([ i for i in range(head_masked, T)], device=device)
        
        return mask_idx, unmask_idx, "front"



    def mask_input(self, x, mode='all'):
        """
        x: Tensor of shape (B, T, C)
        mode: 'random' or 'peak'
        Returns:
            mask_indices: Tensor of shape (B, num_masked)
            unmask_indices: Tensor of shape (B, T - num_masked)
        """
        B, T, C = x.shape
        device = x.device
        num_masked = int(T * self.mask_ratio)

        available_mods = ['random', 'peak', 'voigt_peak', 'front', 'all']
        assert mode in available_mods, print(f"Unknown masking mode: {mode}")

        mask_indices_list, unmask_indices_list = [], []

        choice = torch.randperm(4)[0].item()

        for i in range(B):
            if mode == 'random':
                mask_idx, unmask_idx, task_name = self.random_masking(T, num_masked, device)
            elif mode == 'peak':
                mask_idx, unmask_idx, task_name = self.peak_masking(x, i, T, num_masked, device)
            elif mode == 'voigt_peak':
                mask_idx, unmask_idx, task_name = self.voigt_peak_masking(x, i, T, num_masked, device)
            elif mode == "front":
                mask_idx, unmask_idx, task_name = self.front_masking(T, num_masked, device)
            else:
                if choice==0:
                    mask_idx, unmask_idx, task_name = self.random_masking(T, num_masked, device)
                elif choice==1:
                    mask_idx, unmask_idx, task_name = self.peak_masking(x, i, T, num_masked, device)
                elif choice==2:
                    mask_idx, unmask_idx, task_name = self.voigt_peak_masking(x, i, T, num_masked, device)
                else: 
                    mask_idx, unmask_idx, task_name = self.front_masking(T, num_masked, device)
                
            mask_indices_list.append(mask_idx)
            unmask_indices_list.append(unmask_idx)

        return torch.stack(mask_indices_list), torch.stack(unmask_indices_list), task_name

    def forward(self, x, task=None):
        B, T, C = x.shape
        
        if task == 'denoising':
            # Skip masking: pass the full sequence
            masked_input = x.clone()
            mask_indices = None
            task_name = "denoising"
        else:
            # Normal MAE masking logic
            mask_indices, unmask_indices, task_name = self.mask_input(x)
            
            masked_input = torch.zeros_like(x)
            for i in range(B):
                masked_input[i, unmask_indices[i]] = x[i, unmask_indices[i]]
                
        # Encode the visible tokens
        encoded = self.encoder(masked_input)

        if self.decoder_type == 'conv1d':
            encoded = encoded.permute(0,2,1)
            reconstructed = self.mae_decoder(encoded)  # Skip the first token which is the [CLS] token
            encoded = encoded.permute(0,2,1)
            reconstructed = reconstructed.permute(0,2,1)[:, 1:]
        else:
            reconstructed = self.mae_decoder(encoded)[:, 1:]

        if mask_indices is not None:
            reconstructed_voigt, voigt_params = self.voigt_decoder(encoded, x[:, :, 0].gather(1, mask_indices))
        else:
            reconstructed_voigt, voigt_params = self.voigt_decoder(encoded, x[:, :, 0])

        reconstructed_denoised = self.denoising_decoder(encoded[:, 1:])
            
        return {
            "reconstruction": reconstructed, 
            "reconstruction_voigt": reconstructed_voigt, 
            "reconstruction_denoised": reconstructed_denoised,
            "mask_idx": mask_indices, 
            "task": task_name, 
            "voigt_peaks": voigt_params}