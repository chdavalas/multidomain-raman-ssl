import torch
import torch.nn.functional as F
import numpy as np

def identity(spectra: torch.Tensor) -> torch.Tensor:
    """
    No augmentation, return the original spectra.
    """
    return spectra

def intensity_scaling(spectra, scale_range=(0.9, 1.1)) -> np.ndarray:
    """
    Multiply intensities by a random scalar in scale_range.
    Returns a NumPy array for compatibility with downstream code.
    """
    if isinstance(spectra, np.ndarray):
        spectra = torch.tensor(spectra, dtype=torch.float32)

    if spectra.ndim == 1:
        spectra = spectra.unsqueeze(0)  # Shape (1, L)

    N, L = spectra.shape
    device = spectra.device

    scale_factors = torch.empty(N, 1, device=device).uniform_(*scale_range)
    scaled = spectra * scale_factors

    return scaled.squeeze(0).cpu().numpy() if N == 1 else scaled.cpu().numpy()

def wavelength_jitter(spectra: torch.Tensor, max_jitter=2) -> torch.Tensor:
    """
    Shift a 1D spectrum left or right by up to max_jitter indices.
    Args:
        spectrum: np.ndarray of shape (L,)
        max_jitter: maximum number of indices to shift
    Returns:
        jittered spectrum, shape (L,)
    """
    L = spectra.shape[0]
    shift = np.random.randint(-max_jitter, max_jitter + 1)
    return torch.tensor(np.roll(spectra, shift), dtype=torch.float32)

def spectral_noise_injection(spectra: torch.Tensor, noise_level=0.02) -> torch.Tensor:
    """
    Add Gaussian noise scaled by intensity.
    """
    noise = torch.randn_like(torch.from_numpy(spectra)) * noise_level * np.std(spectra, axis=0)
    noise = noise.numpy()
    return spectra + noise

def peak_shift_stretch(spectra, shift_range=(-5, 5), stretch_range=(0.95, 1.05)) -> np.ndarray:
    """
    Randomly shift and stretch a 1D spectrum using linear interpolation.
    Args:
        spectrum: np.ndarray of shape (L,)
        shift_range: tuple (min_shift, max_shift) in indices
        stretch_range: tuple (min_stretch, max_stretch) scaling factor
    Returns:
        shifted and stretched spectrum, shape (L,)
    """
    L = spectra.shape[0]
    x = np.linspace(0, 1, L)

    # Random shift and stretch
    shift = np.random.uniform(*shift_range) / L
    stretch = np.random.uniform(*stretch_range)

    x_aug = (x - 0.5) * stretch + 0.5 + shift
    x_aug = np.clip(x_aug, 0, 1)

    # Interpolate original spectrum to new x positions
    return torch.tensor(np.interp(x, x_aug, spectra), dtype=torch.float32)


def random_augment(
    spectra: torch.Tensor,
    augmentations=None,
    weights=[0.4, 0.15, 0.15, 0.15, 0.15],
) -> torch.Tensor:
    """
    Apply a random augmentation to each spectrum in the batch, using optional weights.
    Args:
        spectra (torch.Tensor): Batch of spectra, shape (N, L)
        augmentations (list): List of augmentation functions
        weights (list or torch.Tensor): Probabilities for choosing each augmentation
    Returns:
        torch.Tensor: Augmented spectra
    """
    if augmentations is None:
        augmentations = [
            identity,
            intensity_scaling,
            wavelength_jitter,
            spectral_noise_injection,
            peak_shift_stretch,
        ]

    num_augs = len(augmentations)

    if weights is None:
        weights = torch.ones(num_augs)
    else:
        weights = torch.tensor(weights, dtype=torch.float)

    # Sample one augmentation for each spectrum in the batch
    choice = torch.multinomial(weights, num_samples=1, replacement=True).item()
    return augmentations[choice](spectra)
