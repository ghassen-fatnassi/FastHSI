import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math

PROJ_PATH       = './data/projection.pt'
PSF_POINTS_PATH = './data/ctis_psf_points.pt'
HSI_PATH        = './data/butterfly_hsi.pt'
KERNEL_PATH     = './data/ctis_kernel.pt'
RECON_SAVE_PATH = './data/reconstruction_wiener.pt'
PLOT_PATH       = './plots/reconstruction_wiener.png'

GRID_COLS = 8
WIENER_K  = 0.1 # noise-to-signal ratio — lower = sharper, higher = more stable

# ══════════════════════════════════════════════════════════════════════════════
# LOAD
# ══════════════════════════════════════════════════════════════════════════════
projection = torch.load(PROJ_PATH,       weights_only=True)
psf_points = torch.load(PSF_POINTS_PATH)
hsi        = torch.load(HSI_PATH,        weights_only=True)
kernel     = torch.load(KERNEL_PATH,     weights_only=True)

C, h_src, w_src    = hsi.shape
H_canvas, W_canvas = projection.shape
hh, hw             = h_src // 2, w_src // 2

print(f"Projection : {list(projection.shape)}")
print(f"Bands      : {C}   Patch : {h_src}×{w_src}")
print(f"Kernel     : {list(kernel.shape)}")

# ══════════════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════════════
def psnr(a, b):
    return 10 * torch.log10(1.0 / F.mse_loss(a, b))

def sam(a, b):
    a_flat = a.reshape(C, -1).T
    b_flat = b.reshape(C, -1).T
    dot    = (a_flat * b_flat).sum(dim=1)
    norm_a = a_flat.norm(dim=1).clamp(min=1e-8)
    norm_b = b_flat.norm(dim=1).clamp(min=1e-8)
    angle  = torch.acos((dot / (norm_a * norm_b)).clamp(-1, 1))
    return torch.rad2deg(angle).mean()

# ══════════════════════════════════════════════════════════════════════════════
# BACKPROJECTION
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Backprojection ──")
recon_bp = torch.zeros(C, h_src, w_src)
for band_idx, points in enumerate(psf_points):
    patch_acc = torch.zeros(h_src, w_src)
    count     = 0
    for (cx, cy, w) in points[:-1]:
        cx_i, cy_i = int(round(cx)), int(round(cy))
        r0, r1 = cy_i - hh, cy_i - hh + h_src
        c0, c1 = cx_i - hw, cx_i - hw + w_src
        if r0 < 0 or r1 > H_canvas or c0 < 0 or c1 > W_canvas:
            continue
        patch_acc += projection[r0:r1, c0:c1]
        count     += 1
    if count > 0:
        recon_bp[band_idx] = patch_acc / count

print(f"PSNR : {psnr(recon_bp, hsi):.2f} dB  |  SAM : {sam(recon_bp, hsi):.4f}°")

# ══════════════════════════════════════════════════════════════════════════════
# WIENER DECONVOLUTION
#
# Each backprojected band was blurred by the PSF kernel during forward pass.
# Wiener filter inverts this in frequency domain:
#
#   H     = FFT(kernel, size=[H,W])          — PSF transfer function
#   G     = FFT(blurry_band)
#   X_hat = G * conj(H) / (|H|² + K)        — Wiener estimate
#
# K = noise/signal power ratio:
#   K → 0  : pure inverse filter (amplifies noise)
#   K → ∞  : no deconvolution (just smooths)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n── Wiener deconvolution  K={WIENER_K} ──")

# pad kernel to match patch size, then FFT  — do this once, reuse for all bands
kh, kw    = kernel.shape
pad_h     = (h_src - kh) // 2
pad_w     = (w_src - kw) // 2
kernel_p  = F.pad(kernel, (pad_w, w_src - kw - pad_w,
                            pad_h, h_src - kh - pad_h))       # [h_src, w_src]
H_fft     = torch.fft.fft2(kernel_p)                          # [h_src, w_src] complex
H_conj    = torch.conj(H_fft)
H_sq      = (H_fft * H_conj).real                             # |H|²  — real
Wiener_D  = H_conj / (H_sq + WIENER_K)                        # Wiener denominator

recon_wiener = torch.zeros_like(recon_bp)
for i in range(C):
    G              = torch.fft.fft2(recon_bp[i])               # FFT of blurry band
    X_hat          = torch.fft.ifft2(G * Wiener_D).real        # Wiener estimate
    X_hat          = torch.fft.fftshift(X_hat)                 # center
    recon_wiener[i] = X_hat

recon_wiener = recon_wiener.clamp(0, 1)

# ══════════════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nPSNR  backprojection : {psnr(recon_bp,     hsi):.2f} dB  |  SAM : {sam(recon_bp,     hsi):.4f}°")
print(f"PSNR  Wiener         : {psnr(recon_wiener, hsi):.2f} dB  |  SAM : {sam(recon_wiener, hsi):.4f}°")

# ══════════════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════════════
torch.save(recon_wiener, RECON_SAVE_PATH)
print(f"\nSaved → {RECON_SAVE_PATH}")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT
# ══════════════════════════════════════════════════════════════════════════════
rows = math.ceil(C / GRID_COLS)
plt.figure(figsize=(GRID_COLS * 2, rows * 2))
for i in range(C):
    plt.subplot(rows, GRID_COLS, i + 1)
    plt.imshow(recon_wiener[i].numpy(), cmap='gray', vmin=0, vmax=1)
    plt.title(f"band {i}", fontsize=7)
    plt.axis('off')
plt.suptitle(
    f"Wiener Deconvolution  K={WIENER_K}  |  "
    f"PSNR={psnr(recon_wiener,hsi):.2f} dB  SAM={sam(recon_wiener,hsi):.4f}°",
    fontsize=14)
plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=150)
plt.close()
print(f"Plot saved → {PLOT_PATH}")