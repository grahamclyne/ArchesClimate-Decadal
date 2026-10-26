import torch
import torch.nn.functional as F
from torch import nn


class OptimizedSpectralLoss(nn.Module):
    """
    GPU-optimized spectral loss with minimal overhead.
    No numpy dependencies, no unnecessary padding.
    """

    def __init__(self, alpha=1.0, beta=0.5, gamma=0.3):
        super().__init__()
        self.alpha = alpha  # Power spectral density weight
        self.beta = beta  # Phase weight
        self.gamma = gamma  # Magnitude weight

    def _prepare_tensors(self, pred, target):
        """Handle tensor shapes and convert to float32 for FFT"""
        if pred.dim() == 5:  # Shape like [1, 17, 1, 143, 144]
            pred = pred.squeeze(2)
            target = target.squeeze(2)

        # Convert to float32 for FFT (cuFFT requirement)
        return pred.float(), target.float()

    def power_spectral_density_loss(self, pred, target):
        """GPU-optimized PSD loss without padding"""
        pred_f32, target_f32 = self._prepare_tensors(pred, target)

        # FFT along spatial dimensions
        pred_fft = torch.fft.fft2(pred_f32, dim=(-2, -1))
        target_fft = torch.fft.fft2(target_f32, dim=(-2, -1))

        # Power spectral density
        pred_psd = torch.abs(pred_fft) ** 2
        target_psd = torch.abs(target_fft) ** 2

        # Log scale with numerical stability
        pred_psd_log = torch.log(pred_psd + 1e-8)
        target_psd_log = torch.log(target_psd + 1e-8)

        return F.mse_loss(pred_psd_log, target_psd_log)

    def phase_loss(self, pred, target):
        """GPU-optimized phase loss"""
        pred_f32, target_f32 = self._prepare_tensors(pred, target)

        pred_fft = torch.fft.fft2(pred_f32, dim=(-2, -1))
        target_fft = torch.fft.fft2(target_f32, dim=(-2, -1))

        # Extract phases
        pred_phase = torch.angle(pred_fft)
        target_phase = torch.angle(target_fft)

        # Phase difference (wrapped to [-π, π]) - fully on GPU
        pi = torch.tensor(torch.pi, device=pred_phase.device, dtype=pred_phase.dtype)
        phase_diff = torch.remainder(pred_phase - target_phase + pi, 2 * pi) - pi

        return torch.mean(phase_diff**2)

    # def magnitude_loss(self, pred, target):
    #     """Direct magnitude comparison in frequency domain"""
    #     pred_f32, target_f32 = self._prepare_tensors(pred, target)

    #     pred_fft = torch.fft.fft2(pred_f32, dim=(-2, -1))
    #     target_fft = torch.fft.fft2(target_f32, dim=(-2, -1))

    #     pred_mag = torch.abs(pred_fft)
    #     target_mag = torch.abs(target_fft)

    # return F.mse_loss(pred_mag, target_mag)

    def forward(self, pred, target):
        """Combined spectral loss"""
        psd_loss = self.power_spectral_density_loss(pred, target)
        phase_loss = self.phase_loss(pred, target)
        # mag_loss = self.magnitude_loss(pred, target)
        # print(psd_loss)
        # print(phase_loss)
        # print(mag_loss)
        total_loss = self.alpha * psd_loss + self.beta * phase_loss

        return total_loss
