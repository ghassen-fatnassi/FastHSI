import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time

# ══════════════════════════════════════════════════════════════════════════════
# Paths
# ══════════════════════════════════════════════════════════════════════════════
HSI_PATH            = './data/butterfly_hsi.pt'
PSF_SAVE_PATH       = './data/ctis_psf.pt'
PROJ_SAVE_PATH      = './data/projection.pt'
PROJ_PLOT_PATH      = './plots/ctis_projection.png'

# ══════════════════════════════════════════════════════════════════════════════
# FORWARD PROJECTION
# ═════════════════════════════════════════════════════════════════════════════════
psf=torch.load(PSF_SAVE_PATH, weights_only=True)
hsi = torch.load(HSI_PATH, weights_only=True)
print(f"  HSI shape : {list(hsi.shape)}")

C, H, W     = psf.shape
_, h_src, w_src = hsi.shape
assert hsi.shape[0] == C, f"Band mismatch: PSF={C}  HSI={hsi.shape[0]}"

print("Projecting...")
canvas = torch.zeros(H, W)

for lam in range(C):
    band   = hsi[lam]                                               # [h, w]
    pad_h  = (H - h_src) // 2;  pad_w = (W - w_src) // 2
    band_p = F.pad(band, (pad_w, W - w_src - pad_w,
                           pad_h, H - h_src - pad_h))              # [H, W]

    B_fft   = torch.fft.fft2(band_p)
    PSF_fft = torch.fft.fft2(psf[lam])
    result  = torch.fft.ifft2(B_fft * PSF_fft).real
    result  = torch.fft.fftshift(result)
    canvas += result.clamp(min=0)

canvas = canvas / (canvas.max() + 1e-8)
print(f"  Output shape : {list(canvas.shape)}")
print(f"  Output range : [{canvas.min():.4f}, {canvas.max():.4f}]")

torch.save(canvas, PROJ_SAVE_PATH)
print(f"  Saved → {PROJ_SAVE_PATH}")

# ── plot projection ───────────────────────────────────────────────────────────
plt.figure(figsize=(8, 8), facecolor='#080808')
plt.imshow(canvas.numpy(), cmap='gray', interpolation='bicubic')
plt.colorbar(label='Normalised intensity')
plt.title('CTIS projection', color='white', fontsize=12)
plt.axis('off')
plt.tight_layout()
plt.savefig(PROJ_PLOT_PATH, dpi=150, bbox_inches='tight', facecolor='#080808')
print(f"  Saved → {PROJ_PLOT_PATH}")
plt.close()