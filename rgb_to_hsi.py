import torch
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import math

# ── Config / Hardcoded values ─────────────────────────────────────────────────
IMG_PATH       = './data/butterfly.png'
OUT_PATH       = './data/butterfly_hsi.pt'
IMG_SIZE       = 512
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