import torch
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import math

# ── Config / Hardcoded values ─────────────────────────────────────────────────
IMG_PATH       = './data/butterfly.png'
OUT_PATH       = './data/butterfly_hsi.pt'
IMG_SIZE       = 256
BANDS          = 32
START_LAMBDA   = 400  # nm
END_LAMBDA     = 710  # nm
NOISE_LEVEL    = 0.01
GRID_COLS      = 8

# Spectral basis parameters
BASIS_PARAMS = {
    "r": {"mu": 680, "sigma": 20},
    "g": {"mu": 580, "sigma": 40},
    "b": {"mu": 440, "sigma": 20}
}

# ── Load + preprocess ─────────────────────────────────────────────────────────
img = Image.open(IMG_PATH).convert("RGB")
img = img.resize((IMG_SIZE, IMG_SIZE), resample=Image.BICUBIC)
rgb = np.asarray(img).astype(np.float32) / 255.0  # [H,W,3]

H, W, _ = rgb.shape
B = BANDS

# ── Wavelength grid ───────────────────────────────────────────────────────────
wavelengths = np.linspace(START_LAMBDA, END_LAMBDA, B)
print("Wavelengths:", wavelengths.shape, wavelengths[0], "→", wavelengths[-1], "nm")

# ── Spectral basis ────────────────────────────────────────────────────────────
def gaussian(wl, mu, sigma):
    return np.exp(-0.5 * ((wl - mu) / sigma) ** 2)

basis_r = gaussian(wavelengths, BASIS_PARAMS["r"]["mu"], BASIS_PARAMS["r"]["sigma"])
basis_g = gaussian(wavelengths, BASIS_PARAMS["g"]["mu"], BASIS_PARAMS["g"]["sigma"])
basis_b = gaussian(wavelengths, BASIS_PARAMS["b"]["mu"], BASIS_PARAMS["b"]["sigma"])
basis = np.stack([basis_r, basis_g, basis_b], axis=0)
basis /= (basis.max(axis=1, keepdims=True) + 1e-8)
print("Basis shape:", basis.shape)
# ── Convert RGB → spectrum ───────────────────────────────────────────────────
spectral = np.tensordot(rgb, basis, axes=([2],[0]))
spectral /= spectral.max()
print("Spectral shape:", spectral.shape)

# ── Add spectral noise ───────────────────────────────────────────────────────
spectral += NOISE_LEVEL * np.random.randn(H, W, B)
spectral = np.clip(spectral, 0.0, 1.0)

# ── Convert to tensor (C,H,W) ───────────────────────────────────────────────
hsi = torch.from_numpy(spectral).permute(2,0,1).contiguous()

# ── Save HSI ────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)
torch.save(hsi, OUT_PATH)
print(f"HSI saved: {OUT_PATH}, shape={tuple(hsi.shape)}, min={hsi.min():.4f}, max={hsi.max():.4f}")

# ── Visualize HSI channels ───────────────────────────────────────────────────
x = torch.load(OUT_PATH, weights_only=True)
C, H, W = x.shape
rows = math.ceil(C / GRID_COLS)

plt.figure(figsize=(GRID_COLS*2, rows*2))
for i in range(C):
    plt.subplot(rows, GRID_COLS, i+1)
    plt.imshow(x[i].numpy(), cmap='gray', vmin=0, vmax=1)
    plt.title(f"{int(wavelengths[i])}nm", fontsize=8)
    plt.axis('off')
plt.suptitle("HSI Channels", fontsize=16)
plt.tight_layout()
#saving the plot
plt.savefig('./plots/hsi_channels.png', dpi=300)
print("HSI channels visualized and saved to './plots/hsi_channels.png'")



import numpy as np
import torch
import time
from matplotlib import pyplot as plt
import os

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
FIRST_LAMBDA_CENTER = 350
SHIFT               = 64
N_ORDERS            = 12
N_BANDS             = 32
PSF_SIGMA           = 4.0
GAUSS_SIGMA_0TH     = 8.0
CANVAS              = 5000

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