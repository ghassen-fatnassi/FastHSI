import numpy as np
import torch
import time
from matplotlib import pyplot as plt
import os

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
FIRST_LAMBDA_CENTER = 350
SHIFT               = 32
N_ORDERS            = 12
N_BANDS             = 32
PSF_SIGMA           = 4.0
GAUSS_SIGMA_0TH     = 8.0
CANVAS              = 2048

PSF_SAVE_PATH       = './data/ctis_psf.pt'

os.makedirs('./data', exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
CENTER = CANVAS // 2
ANGLES = [np.radians(i * 360.0 / N_ORDERS) for i in range(N_ORDERS)]

def gaussian_kernel(sigma: float, truncate: float = 4.0) -> torch.Tensor:
    half = int(np.ceil(truncate * sigma))
    size = 2 * half + 1
    y, x = torch.meshgrid(
        torch.arange(size, dtype=torch.float32) - half,
        torch.arange(size, dtype=torch.float32) - half,
        indexing='ij'
    )
    g = torch.exp(-(x**2 + y**2) / (2.0 * sigma**2))
    return g / g.sum()

def stamp_gaussian(canvas: torch.Tensor, cy: float, cx: float, kernel: torch.Tensor):
    H, W   = canvas.shape
    ph, pw = kernel.shape
    hh, hw = ph // 2, pw // 2
    r0 = int(round(cy)) - hh;  r1 = r0 + ph
    c0 = int(round(cx)) - hw;  c1 = c0 + pw
    dr0, dr1 = max(0, r0), min(H, r1)
    dc0, dc1 = max(0, c0), min(W, c1)
    if dr1 <= dr0 or dc1 <= dc0:
        return
    sr0 = dr0 - r0;  sr1 = sr0 + (dr1 - dr0)
    sc0 = dc0 - c0;  sc1 = sc0 + (dc1 - dc0)
    canvas[dr0:dr1, dc0:dc1] += kernel[sr0:sr1, sc0:sc1]

def spectral_gaussian_weights(n_bands: int, sigma: float) -> torch.Tensor:
    mid  = (n_bands - 1) / 2.0
    lams = torch.arange(n_bands, dtype=torch.float32)
    out = torch.exp(-((lams - mid) ** 2) / (2.0 * sigma ** 2))
    return out / out.sum()

# ──────────────────────────────────────────────────────────────────────────────
# BUILD PSF, SAVE PSF, POINTS, KERNEL
# ──────────────────────────────────────────────────────────────────────────────
print("Building PSF tensor...")
t0 = time.perf_counter()

kernel   = gaussian_kernel(PSF_SIGMA)       # Gaussian kernel
torch.save(kernel, './data/ctis_kernel.pt') # save kernel for later
weights0 = spectral_gaussian_weights(N_BANDS, GAUSS_SIGMA_0TH)
print("weights 0th order:", weights0)

psf      = torch.zeros(N_BANDS, CANVAS, CANVAS, dtype=torch.float32)

# Store all spot points per band
psf_points = []  # list of lists: [(band_id, cx, cy, weight), ...]

for j in range(N_BANDS):
    band_points = []
    r = FIRST_LAMBDA_CENTER + SHIFT * j
    for theta in ANGLES:
        cx = r * np.cos(theta) + CENTER
        cy = r * np.sin(theta) + CENTER
        stamp_gaussian(psf[j], cy, cx, kernel)
        band_points.append((cx, cy, 1.0))  # 1.0 = full weight for diffraction orders

    # add zero-order spot in center
    stamp_gaussian(psf[j], CENTER, CENTER, kernel * weights0[j].item())
    band_points.append((CENTER, CENTER, weights0[j].item()))

    psf_points.append(band_points)

dt = time.perf_counter() - t0
print(f"Peak value in final PSF: {psf.max():.5f}")

torch.save(psf, PSF_SAVE_PATH)
torch.save(psf_points, './data/ctis_psf_points.pt')  # save points for reuse
# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZE ONE PSF (ONE SPOT, ZOOMED)
# ══════════════════════════════════════════════════════════════════════════════

band_id  = N_BANDS // 2          # pick any from 0 to N_BANDS-1
theta_id = 0                     # pick one diffraction direction

r = FIRST_LAMBDA_CENTER + SHIFT * band_id
theta = ANGLES[theta_id]

cx = int(r * np.cos(theta) + CENTER)
cy = int(r * np.sin(theta) + CENTER)

# crop around the spot
crop_size = 12
patch = psf[band_id,
            cy - crop_size:cy + crop_size,
            cx - crop_size:cx + crop_size].numpy()

plt.figure(figsize=(5,5))
plt.imshow(patch, cmap='inferno')
plt.title(f"Zoomed PSF (band={band_id}, order={theta_id})")
plt.axis('off')
plt.savefig('./plots/psf_zoomed.png', bbox_inches='tight')
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZE ALL PSFs ON SAME CANVAS (SUM)
# ══════════════════════════════════════════════════════════════════════════════

psf_sum = psf.sum(dim=0).numpy()

plt.figure(figsize=(50,50), dpi=100)  # 50*100 = 5000 px
plt.imshow(psf_sum, cmap='inferno')
plt.title("All PSFs overlaid (sum over channels)")
plt.axis('off')
plt.savefig('./plots/psf_sum_fullres.png', dpi=100, bbox_inches='tight')
plt.close()