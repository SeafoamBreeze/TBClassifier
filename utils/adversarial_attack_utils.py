import io
import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F
import kornia
import pywt

def fgsm_attack(input_tensor: torch.Tensor, epsilon: float, data_grad: torch.Tensor) -> torch.Tensor:
    sign_data_grad = data_grad.sign()
    perturbed = input_tensor.detach() + epsilon * sign_data_grad
    perturbed = torch.clamp(perturbed, 0, 1)
    return perturbed

def apply_mitigation(
    image_tensor: torch.Tensor,
    method: str
) -> torch.Tensor:
    
    mitigator = MitigationTechniques()

    if method == "wavelet_squeeze":
        mitigated = mitigator.wavelet_squeeze(image_tensor)
    elif method == "compression_squeeze":
        mitigated = mitigator.compression_squeeze(image_tensor)
    elif method == "spatial_squeeze":
        mitigated = mitigator.spatial_squeeze(image_tensor)
    elif method == "wavelet_spatial_hybrid":
        mitigated = mitigator.wavelet_spatial_hybrid(image_tensor)
    elif method == "aggressive_wavelet_spatial_hybrid":
        mitigated = mitigator.aggressive_wavelet_spatial_hybrid(image_tensor)
    elif method == "asymmetric_smooth_first":
        mitigated = mitigator.asymmetric_smooth_first(image_tensor)

    return mitigated.clamp(0, 1)

def tensor_to_numpy(tensor: torch.Tensor, was_clamped: bool = False) -> np.ndarray:

    print(f"DEBUG tensor shape: {tensor.shape}, dim: {tensor.dim()}") 
    tensor = tensor.detach().cpu()
    
    if tensor.dim() == 4:
        if tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        else:
            tensor = tensor[0] 

    if tensor.dim() != 3:
        raise ValueError(f"Expected 3D tensor (C,H,W), got shape {tensor.shape}")
    
    img = tensor.permute(1, 2, 0).numpy()
    
    if was_clamped:
        img = np.clip(img, 0, 1)
    else:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img * std + mean
        img = np.clip(img, 0, 1)
    
    return img

class MitigationTechniques:
    def __init__(self):
        self.bits = 4  # Aggressive for X-ray: 16 levels
        self.levels = 2 ** self.bits
        
    # ============ BASE METHODS ============
    
    def bit_depth_reduction(self, x):
        """Reduce to 4-bit (16 levels) for X-ray"""
        return torch.round(x * (self.levels - 1)) / (self.levels - 1)
    
    def median_filter(self, x, kernel_size=2):
        """Median smoothing"""
        if kernel_size % 2 == 0:
            kernel_size += 1
        padding = kernel_size // 2
        unfolded = F.unfold(x, kernel_size, padding=padding)
        unfolded = unfolded.view(x.size(0), x.size(1), kernel_size * kernel_size, -1)
        median = unfolded.median(dim=2)[0]
        h, w = x.size(2), x.size(3)
        return median.view(x.size(0), x.size(1), h, w)
    
    def gaussian_blur(self, x, kernel_size=3, sigma=0.5):
        """Gaussian smoothing"""
        if kernel_size % 2 == 0:
            kernel_size += 1
        x_coord = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=torch.float32)
        gauss_1d = torch.exp(-x_coord.pow(2) / (2 * sigma ** 2))
        gauss_1d = gauss_1d / gauss_1d.sum()
        kernel_2d = gauss_1d.unsqueeze(1) * gauss_1d.unsqueeze(0)
        kernel_2d = kernel_2d.expand(x.size(1), 1, kernel_size, kernel_size)
        padding = kernel_size // 2
        return F.conv2d(x, kernel_2d.to(x.device), padding=padding, groups=x.size(1))
    
    def jpeg_compression(self, x, quality=75):
        """JPEG compression simulation"""
        block_size = 8
        b, c, h, w = x.shape
        pad_h = (block_size - h % block_size) % block_size
        pad_w = (block_size - w % block_size) % block_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        compressed = F.avg_pool2d(x, block_size, stride=block_size)
        compressed = F.interpolate(compressed, size=(h, w), mode='nearest')
        return compressed[:, :, :h, :w] if (pad_h > 0 or pad_w > 0) else compressed
    
    def random_resize_pad(self, x, scale=0.8):
        """Resize and pad back to original"""
        b, c, h, w = x.shape
        new_h, new_w = int(h * scale), int(w * scale)
        resized = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)
        pad_top = (h - new_h) // 2
        pad_bottom = h - new_h - pad_top
        pad_left = (w - new_w) // 2
        pad_right = w - new_w - pad_left
        return F.pad(resized, (pad_left, pad_right, pad_top, pad_bottom))
    
    def wavelet_denoise(self, x, wavelet='db4', levels=2, keep_approx_only=False):
        """
        Discrete Wavelet Transform denoising
        Keeps low-frequency (structure), removes high-frequency (adversarial noise)
        """
        
        x_np = x.cpu().numpy()
        denoised = []
        
        for img in x_np:
            # img shape: (C, H, W)
            reconstructed_channels = []
            for c in range(img.shape[0]):
                coeffs = pywt.wavedec2(img[c], wavelet, level=levels, mode='symmetric')
                if keep_approx_only:
                    # Aggressive: Keep only approximation, zero out all details
                    new_coeffs = [coeffs[0]]
                    for detail_tuple in coeffs[1:]:
                        zeroed_detail = tuple(np.zeros_like(d) for d in detail_tuple)
                        new_coeffs.append(zeroed_detail)
                else:
                    # Soft thresholding on detail coefficients (less aggressive)
                    new_coeffs = [coeffs[0]]  # Keep approximation as-is
                    for detail_tuple in coeffs[1:]:
                        thresholded_detail = tuple(
                            np.where(np.abs(d) > 0.1, d, 0) for d in detail_tuple
                        )
                        new_coeffs.append(thresholded_detail)
            
                # Reconstruct
                reconstructed = pywt.waverec2(new_coeffs, wavelet, mode='symmetric')
                reconstructed = reconstructed[:img.shape[1], :img.shape[2]]
                reconstructed_channels.append(reconstructed)
            denoised.append(np.stack(reconstructed_channels))
        
        result = torch.tensor(np.stack(denoised), dtype=x.dtype, device=x.device)
        return result
    
    # ============ PHASE 2: THREE-METHOD STRATEGIES ============
    
    def wavelet_squeeze(self, x):
        """
        Strategy 1: Wavelet-Squeeze
        Order: wavelet_denoise → bit_depth_reduction → median_filter
        Rationale: Frequency separation → Quantize → Clean artifacts
        Best for: Removing high-frequency adversarial perturbations
        """
        x = self.wavelet_denoise(x, wavelet='db4', levels=2)
        x = self.bit_depth_reduction(x)
        x = self.median_filter(x, kernel_size=2)
        return x
    
    def compression_squeeze(self, x):
        """
        Strategy 2: Compression-Squeeze
        Order: jpeg_compression → bit_depth_reduction → median_filter
        Rationale: Remove high-freq via JPEG → Quantize → Smooth artifacts
        Best for: JPEG-compatible adversarial noise
        """
        x = self.jpeg_compression(x, quality=75)
        x = self.bit_depth_reduction(x)
        x = self.median_filter(x, kernel_size=2)
        return x
    
    def spatial_squeeze(self, x):
        """
        Strategy 3: Spatial-Squeeze
        Order: random_resize_pad → gaussian_blur → bit_depth_reduction
        Rationale: Reduce dimensionality → Smooth → Quantize
        Best for: Spatially distributed adversarial patches
        """
        x = self.random_resize_pad(x, scale=0.8)
        x = self.gaussian_blur(x, kernel_size=3, sigma=0.5)
        x = self.bit_depth_reduction(x)
        return x
    
    def wavelet_spatial_hybrid(self, x):
        """
        Wavelet denoise first, then aggressive spatial squeeze
        """
        # Keep only approximation coefficients (lowest frequency)
        x = self.wavelet_denoise(x, levels=2, keep_approx_only=True)
        x = self.random_resize_pad(x, scale=0.75)  # More aggressive than 0.8
        x = self.bit_depth_reduction(x)
        x = self.gaussian_blur(x, kernel_size=5, sigma=1.0)  # Stronger blur
        return x
        
    def asymmetric_smooth_first(self, x):
        """
        Sequential 1x3 then 3x1 smoothing
        Better for directional adversarial noise
        """
        # Horizontal smoothing
        x = F.avg_pool2d(x, kernel_size=(1, 3), stride=1, padding=(0, 1))
        # Vertical smoothing  
        x = F.avg_pool2d(x, kernel_size=(3, 1), stride=1, padding=(1, 0))
        x = self.bit_depth_reduction(x)
        return x    