# Kaggle-ready HSI family benchmark
# Paste this into a Kaggle notebook cell or save as a .py file and run.
# You only need to edit DATASET_DIRS.
#
# What it does:
# 1) scans 3 folders of .pt HSI cubes, each (29, 512, 512)
# 2) aligns dataset means with one fixed gain per dataset (minimal distribution alignment)
# 3) synthesizes coded measurements + backprojections + zero-order images
# 4) builds stratified train/val split across the 3 datasets
# 5) trains 4 experiments under the same budget:
#       - NAFNet-small, supervised only
#       - NAFNet-small, supervised + forward consistency
#       - Tiny Uformer, supervised only
#       - Tiny Uformer, supervised + forward consistency
# 6) uses gradient accumulation, tqdm postfix metrics, live metric plotting, checkpointing

import os
import gc
import math
import json
import time
import random
from glob import glob
from copy import deepcopy
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import clear_output, display

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm.auto import tqdm

# ============================================================
# USER CONFIG
# ============================================================
DATASET_DIRS = {
    "cave": "/kaggle/input/your-cave-folder",
    "harvard": "/kaggle/input/your-harvard-folder",
    "kaist": "/kaggle/input/your-kaist-folder",
}

SEED = 42
VAL_SAMPLES = 14
BATCH_SIZE = 2                   # 2 works well with DataParallel on 2xT4
GRAD_ACCUM_STEPS = 4             # effective batch ~= 8 on 1 GPU, ~= 16 on 2 GPUs with DP
NUM_WORKERS = 2
PIN_MEMORY = True
EPOCHS = 120
EARLY_STOP_PATIENCE = 20
LR = 2e-4
WEIGHT_DECAY = 1e-4
AMP = True
USE_DATAPARALLEL = True
LIVE_PLOT = True
SAVE_EVERY_EPOCH = True

# Minimal distribution alignment across datasets.
# One fixed gain per dataset, clipped after scaling.
ALIGN_DATASET_MEANS = True
GAIN_TARGET = "median"          # "median" or "mean"
GAIN_MIN = 0.25
GAIN_MAX = 10.0

# Synthetic coding operator config
NOISE_STD = 0.01
N_BANDS = 29
IMG_HW = (512, 512)
PSF_CENTER_WEIGHT = 0.50
PSF_ORDER_WEIGHT = 0.50
PSF_SIGMA = 2.0
N_ORDERS = 8
FIRST_RADIUS = 12.0
RADIUS_STEP = 3.5
ZERO_ORDER_BLUR_SIGMA = 1.25

# Loss weights
LAMBDA_L1 = 1.0
LAMBDA_SAM = 0.1
LAMBDA_FWD = 0.2

# Experiments
EXPERIMENTS = [
    {"name": "nafnet_sup",   "model": "nafnet",  "use_forward_loss": False},
    {"name": "nafnet_phys",  "model": "nafnet",  "use_forward_loss": True},
    {"name": "uformer_sup",  "model": "uformer", "use_forward_loss": False},
    {"name": "uformer_phys", "model": "uformer", "use_forward_loss": True},
]

# Output dirs
ROOT_OUT = Path("/kaggle/working/hsi_family_benchmark")
CACHE_DIR = ROOT_OUT / "cache"
RUNS_DIR = ROOT_OUT / "runs"
PLOTS_DIR = ROOT_OUT / "plots"
ROOT_OUT.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# UTILITIES
# ============================================================
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = True

seed_everything(SEED)
torch.set_float32_matmul_precision("high")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0
print(f"device={DEVICE}, num_gpus={N_GPUS}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def psnr_metric(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # pred,target: [B,C,H,W] in [0,1]
    mse = F.mse_loss(pred, target, reduction="none").mean(dim=(1, 2, 3))
    psnr = 10.0 * torch.log10(1.0 / (mse + eps))
    return psnr.mean()


def sam_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # mean spectral angle in radians
    dot = (pred * target).sum(dim=1)
    pred_n = torch.linalg.norm(pred, dim=1)
    targ_n = torch.linalg.norm(target, dim=1)
    cos = dot / (pred_n * targ_n + eps)
    cos = torch.clamp(cos, -1.0 + 1e-6, 1.0 - 1e-6)
    return torch.acos(cos).mean()


def sam_deg_metric(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return sam_loss(pred, target, eps) * (180.0 / math.pi)


def ensure_float_cube(x: torch.Tensor) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    x = x.float()
    if x.ndim != 3:
        raise ValueError(f"Expected [C,H,W], got {tuple(x.shape)}")
    return x


def hflip(x: torch.Tensor) -> torch.Tensor:
    return torch.flip(x, dims=[-1])


def vflip(x: torch.Tensor) -> torch.Tensor:
    return torch.flip(x, dims=[-2])


def rot90(x: torch.Tensor, k: int) -> torch.Tensor:
    return torch.rot90(x, k=k, dims=[-2, -1])


def apply_joint_aug(inp: torch.Tensor, target: torch.Tensor, measurement: torch.Tensor):
    if random.random() < 0.5:
        inp = hflip(inp)
        target = hflip(target)
        measurement = hflip(measurement)
    if random.random() < 0.5:
        inp = vflip(inp)
        target = vflip(target)
        measurement = vflip(measurement)
    k = random.randint(0, 3)
    if k:
        inp = rot90(inp, k)
        target = rot90(target, k)
        measurement = rot90(measurement, k)
    return inp.contiguous(), target.contiguous(), measurement.contiguous()


def plot_history(history: dict, save_path: str, title: str = "history"):
    plt.figure(figsize=(12, 8))
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(history["train_loss"], label="train_loss")
    ax1.plot(history["val_loss"], label="val_loss")
    ax1.legend(); ax1.grid(True, alpha=0.3); ax1.set_title("Loss")

    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(history["train_psnr"], label="train_psnr")
    ax2.plot(history["val_psnr"], label="val_psnr")
    ax2.legend(); ax2.grid(True, alpha=0.3); ax2.set_title("PSNR")

    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(history["train_sam_deg"], label="train_sam_deg")
    ax3.plot(history["val_sam_deg"], label="val_sam_deg")
    ax3.legend(); ax3.grid(True, alpha=0.3); ax3.set_title("SAM (deg)")

    ax4 = plt.subplot(2, 2, 4)
    ax4.plot(history["lr"], label="lr")
    ax4.legend(); ax4.grid(True, alpha=0.3); ax4.set_title("LR")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if LIVE_PLOT:
        clear_output(wait=True)
        display(Image.open(save_path))
    plt.close()


# ============================================================
# SYNTHETIC CODING OPERATOR
# ============================================================
class SimpleSpectralCoder(nn.Module):
    def __init__(
        self,
        n_bands=29,
        hw=(512, 512),
        psf_sigma=2.0,
        n_orders=8,
        first_radius=12.0,
        radius_step=3.5,
        center_weight=0.5,
        order_weight=0.5,
        zero_blur_sigma=1.25,
        noise_std=0.01,
    ):
        super().__init__()
        self.n_bands = n_bands
        self.hw = hw
        self.psf_sigma = psf_sigma
        self.n_orders = n_orders
        self.first_radius = first_radius
        self.radius_step = radius_step
        self.center_weight = center_weight
        self.order_weight = order_weight
        self.zero_blur_sigma = zero_blur_sigma
        self.noise_std = noise_std

        psf = self._build_psf_bank()
        psf_fft = torch.fft.rfft2(psf, dim=(-2, -1))
        self.register_buffer("psf", psf, persistent=False)
        self.register_buffer("psf_fft", psf_fft, persistent=False)

        blur = self._gaussian_kernel2d(self.zero_blur_sigma)
        blur = blur[None, None]  # [1,1,k,k]
        self.register_buffer("zero_blur", blur, persistent=False)

    def _gaussian_kernel2d(self, sigma: float, truncate: float = 4.0) -> torch.Tensor:
        half = int(math.ceil(truncate * sigma))
        size = 2 * half + 1
        yy, xx = torch.meshgrid(
            torch.arange(size, dtype=torch.float32) - half,
            torch.arange(size, dtype=torch.float32) - half,
            indexing="ij",
        )
        g = torch.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
        g = g / g.sum()
        return g

    def _stamp(self, canvas: torch.Tensor, cy: float, cx: float, kernel: torch.Tensor, weight: float = 1.0):
        H, W = canvas.shape
        kh, kw = kernel.shape
        hh, hw = kh // 2, kw // 2
        r0 = int(round(cy)) - hh
        c0 = int(round(cx)) - hw
        r1 = r0 + kh
        c1 = c0 + kw

        rr0, rr1 = max(0, r0), min(H, r1)
        cc0, cc1 = max(0, c0), min(W, c1)
        if rr1 <= rr0 or cc1 <= cc0:
            return

        kr0 = rr0 - r0
        kc0 = cc0 - c0
        kr1 = kr0 + (rr1 - rr0)
        kc1 = kc0 + (cc1 - cc0)
        canvas[rr0:rr1, cc0:cc1] += weight * kernel[kr0:kr1, kc0:kc1]

    def _build_psf_bank(self) -> torch.Tensor:
        H, W = self.hw
        cy0, cx0 = H // 2, W // 2
        angles = [2.0 * math.pi * i / self.n_orders for i in range(self.n_orders)]
        kernel = self._gaussian_kernel2d(self.psf_sigma)
        psf = torch.zeros(self.n_bands, H, W, dtype=torch.float32)

        for b in range(self.n_bands):
            radius = self.first_radius + self.radius_step * b
            order_w = self.order_weight / max(self.n_orders, 1)
            for theta in angles:
                cx = cx0 + radius * math.cos(theta)
                cy = cy0 + radius * math.sin(theta)
                self._stamp(psf[b], cy, cx, kernel, weight=order_w)
            self._stamp(psf[b], cy0, cx0, kernel, weight=self.center_weight)
            s = psf[b].sum()
            if s > 0:
                psf[b] /= s
        return psf

    @torch.no_grad()
    def synthesize(self, cube: torch.Tensor, seed: int = 0):
        # cube: [29,H,W] in [0,1]
        cube = cube.unsqueeze(0).to(self.psf_fft.device)  # [1,C,H,W]
        y = self.forward(cube)                             # [1,1,H,W]

        if self.noise_std > 0:
            gen = torch.Generator(device=y.device)
            gen.manual_seed(seed)
            noise = torch.randn(y.shape, device=y.device, generator=gen) * self.noise_std
            y = torch.clamp(y + noise, min=0.0)

        y = y / (y.amax(dim=(-2, -1), keepdim=True) + 1e-8)
        bp = self.backproject(y)
        bp = bp / (bp.amax(dim=(-2, -1), keepdim=True) + 1e-8)

        z0 = cube.mean(dim=1, keepdim=True)  # [1,1,H,W]
        pad = self.zero_blur.shape[-1] // 2
        z0 = F.conv2d(z0, self.zero_blur, padding=pad)
        z0 = z0 / (z0.amax(dim=(-2, -1), keepdim=True) + 1e-8)

        return y.squeeze(0).cpu(), bp.squeeze(0).cpu(), z0.squeeze(0).cpu()

    def forward(self, cube: torch.Tensor) -> torch.Tensor:
        # cube: [B,C,H,W] -> y: [B,1,H,W]
        X = torch.fft.rfft2(cube, dim=(-2, -1))
        Y = torch.fft.irfft2(X * self.psf_fft.unsqueeze(0), s=self.hw, dim=(-2, -1))
        y = Y.sum(dim=1, keepdim=True)
        y = torch.clamp(y, min=0.0)
        return y

    def backproject(self, measurement: torch.Tensor) -> torch.Tensor:
        # measurement: [B,1,H,W] -> [B,C,H,W]
        Y = torch.fft.rfft2(measurement, dim=(-2, -1))
        bp = torch.fft.irfft2(Y * torch.conj(self.psf_fft).unsqueeze(0), s=self.hw, dim=(-2, -1))
        bp = torch.clamp(bp, min=0.0)
        return bp


# ============================================================
# FILE DISCOVERY + SPLIT + CACHING
# ============================================================
def scan_hsi_files(dataset_dirs: dict):
    records = []
    for source, root in dataset_dirs.items():
        files = sorted(glob(os.path.join(root, "*.pt")))
        if not files:
            raise FileNotFoundError(f"No .pt files found in {root}")
        for fp in files:
            records.append({
                "source": source,
                "path": fp,
                "name": os.path.basename(fp),
            })
    return records


def compute_dataset_means(records):
    by_source = defaultdict(list)
    for rec in tqdm(records, desc="Scanning dataset means"):
        x = ensure_float_cube(torch.load(rec["path"], map_location="cpu"))
        if tuple(x.shape) != (N_BANDS, IMG_HW[0], IMG_HW[1]):
            raise ValueError(f"Bad shape for {rec['path']}: {tuple(x.shape)}; expected {(N_BANDS, *IMG_HW)}")
        by_source[rec["source"]].append(x.mean().item())
        del x
    means = {k: float(np.mean(v)) for k, v in by_source.items()}
    if GAIN_TARGET == "median":
        target = float(np.median(list(means.values())))
    else:
        target = float(np.mean(list(means.values())))
    gains = {}
    for src, m in means.items():
        g = target / max(m, 1e-8)
        g = float(np.clip(g, GAIN_MIN, GAIN_MAX))
        gains[src] = g
    return means, gains, target


def stratified_split(records, val_total=14, seed=42):
    rng = random.Random(seed)
    groups = defaultdict(list)
    for i, rec in enumerate(records):
        groups[rec["source"]].append(i)

    total = len(records)
    quotas = {}
    remainders = []
    assigned = 0
    for src, idxs in groups.items():
        exact = len(idxs) * val_total / total
        q = int(math.floor(exact))
        quotas[src] = q
        assigned += q
        remainders.append((exact - q, src))

    remaining = val_total - assigned
    remainders.sort(reverse=True)
    for _, src in remainders[:remaining]:
        quotas[src] += 1

    train_idx, val_idx = [], []
    for src, idxs in groups.items():
        idxs = idxs.copy()
        rng.shuffle(idxs)
        q = quotas[src]
        val_idx.extend(idxs[:q])
        train_idx.extend(idxs[q:])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def build_cache(records, gains, coder: SimpleSpectralCoder, rebuild=False):
    cache_records = []
    cache_meta_path = CACHE_DIR / "cache_meta.json"
    if cache_meta_path.exists() and not rebuild:
        with open(cache_meta_path, "r") as f:
            cache_records = json.load(f)
        print(f"Loaded cache metadata: {len(cache_records)} samples")
        return cache_records

    print("Building synthetic cache...")
    coder = coder.to(DEVICE).eval()
    cache_records = []

    for idx, rec in enumerate(tqdm(records, desc="Caching samples")):
        cube = ensure_float_cube(torch.load(rec["path"], map_location="cpu"))
        cube = torch.clamp(cube * gains.get(rec["source"], 1.0), 0.0, 1.0)
        y, bp, z0 = coder.synthesize(cube, seed=SEED + idx)
        inp = torch.cat([bp, z0], dim=0).float().contiguous()  # [30,512,512]
        item = {
            "input": inp.half(),
            "target": cube.half().contiguous(),
            "measurement": y.half().contiguous(),
            "source": rec["source"],
            "name": rec["name"],
        }
        out_fp = CACHE_DIR / f"sample_{idx:04d}.pt"
        torch.save(item, out_fp)
        cache_records.append({
            "cache_path": str(out_fp),
            "source": rec["source"],
            "name": rec["name"],
        })

        del cube, y, bp, z0, inp, item
        if DEVICE.type == "cuda" and idx % 16 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    with open(cache_meta_path, "w") as f:
        json.dump(cache_records, f, indent=2)
    return cache_records


class HSICacheDataset(Dataset):
    def __init__(self, cache_records, augment=False):
        self.cache_records = cache_records
        self.augment = augment

    def __len__(self):
        return len(self.cache_records)

    def __getitem__(self, idx):
        item = torch.load(self.cache_records[idx]["cache_path"], map_location="cpu")
        inp = item["input"].float()          # [30,H,W]
        target = item["target"].float()      # [29,H,W]
        measurement = item["measurement"].float()  # [1,H,W]

        if self.augment:
            inp, target, measurement = apply_joint_aug(inp, target, measurement)

        return {
            "input": inp,
            "target": target,
            "measurement": measurement,
            "source": item["source"],
            "name": item["name"],
        }


# ============================================================
# MODELS: NAFNET SMALL
# ============================================================
class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(dim=1, keepdim=True)
        var = (x - mu).pow(2).mean(dim=1, keepdim=True)
        x = (x - mu) / torch.sqrt(var + self.eps)
        return x * self.weight + self.bias


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, dw_expand=2, ffn_expand=2, dropout=0.0):
        super().__init__()
        dw_ch = c * dw_expand
        ffn_ch = c * ffn_expand

        self.norm1 = LayerNorm2d(c)
        self.pw1 = nn.Conv2d(c, dw_ch, 1)
        self.dw = nn.Conv2d(dw_ch, dw_ch, 3, padding=1, groups=dw_ch)
        self.sg = SimpleGate()
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_ch // 2, dw_ch // 2, 1),
        )
        self.pw2 = nn.Conv2d(dw_ch // 2, c, 1)
        self.dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))

        self.norm2 = LayerNorm2d(c)
        self.ffn1 = nn.Conv2d(c, ffn_ch, 1)
        self.sg2 = SimpleGate()
        self.ffn2 = nn.Conv2d(ffn_ch // 2, c, 1)
        self.dropout2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x):
        y = self.norm1(x)
        y = self.pw1(y)
        y = self.dw(y)
        y = self.sg(y)
        y = y * self.sca(y)
        y = self.pw2(y)
        y = self.dropout1(y)
        x = x + y * self.beta

        y = self.norm2(x)
        y = self.ffn1(y)
        y = self.sg2(y)
        y = self.ffn2(y)
        y = self.dropout2(y)
        return x + y * self.gamma


class Downsample(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.body = nn.Conv2d(c, c * 2, 2, stride=2)
    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(c, c * 2, 1),
            nn.PixelShuffle(2),
        )
    def forward(self, x):
        return self.body(x)


class NAFNetSmall(nn.Module):
    def __init__(self, in_ch=30, out_ch=29, width=32, enc_blocks=(1, 1, 2), dec_blocks=(1, 1, 1), mid_blocks=4):
        super().__init__()
        self.intro = nn.Conv2d(in_ch, width, 3, padding=1)
        self.ending = nn.Conv2d(width, out_ch, 3, padding=1)

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        ch = width
        for nb in enc_blocks:
            self.encoders.append(nn.Sequential(*[NAFBlock(ch) for _ in range(nb)]))
            self.downs.append(Downsample(ch))
            ch *= 2

        self.middle = nn.Sequential(*[NAFBlock(ch) for _ in range(mid_blocks)])

        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for nb in dec_blocks:
            self.ups.append(Upsample(ch))
            ch //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(ch) for _ in range(nb)]))

    def forward(self, x):
        skips = []
        y = self.intro(x)
        for enc, down in zip(self.encoders, self.downs):
            y = enc(y)
            skips.append(y)
            y = down(y)

        y = self.middle(y)

        for up, dec, skip in zip(self.ups, self.decoders, reversed(skips)):
            y = up(y)
            if y.shape[-2:] != skip.shape[-2:]:
                y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = y + skip
            y = dec(y)

        y = self.ending(y)
        y = torch.sigmoid(y)
        return y


# ============================================================
# MODELS: TINY UFORMER / WINDOW TRANSFORMER FAMILY
# ============================================================
class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=2.0, dropout=0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class WindowAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=4, window_size=8, mlp_ratio=2.0, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def window_partition(self, x):
        # x: [B,C,H,W] -> windows: [B*nW, ws*ws, C]
        B, C, H, W = x.shape
        ws = self.window_size
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        Hp, Wp = x.shape[-2:]
        x = x.view(B, C, Hp // ws, ws, Wp // ws, ws)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        windows = x.view(-1, ws * ws, C)
        return windows, Hp, Wp, pad_h, pad_w

    def window_reverse(self, windows, B, Hp, Wp, pad_h, pad_w):
        ws = self.window_size
        C = windows.shape[-1]
        x = windows.view(B, Hp // ws, Wp // ws, ws, ws, C)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.view(B, C, Hp, Wp)
        if pad_h or pad_w:
            x = x[:, :, :Hp - pad_h, :Wp - pad_w]
        return x

    def forward(self, x):
        B, C, H, W = x.shape
        windows, Hp, Wp, pad_h, pad_w = self.window_partition(x)
        z = self.norm1(windows)
        z, _ = self.attn(z, z, z, need_weights=False)
        windows = windows + z
        windows = windows + self.mlp(self.norm2(windows))
        x = self.window_reverse(windows, B, Hp, Wp, pad_h, pad_w)
        return x


class ConvDown(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.body = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)
    def forward(self, x):
        return self.body(x)


class ConvUp(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, out_ch * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
        )
    def forward(self, x):
        return self.body(x)


class TinyUformer(nn.Module):
    def __init__(self, in_ch=30, out_ch=29, dims=(32, 64, 128), window_size=8):
        super().__init__()
        d1, d2, d3 = dims
        self.in_proj = nn.Conv2d(in_ch, d1, 3, padding=1)

        self.e1 = nn.Sequential(
            WindowAttentionBlock(d1, num_heads=4, window_size=window_size),
            WindowAttentionBlock(d1, num_heads=4, window_size=window_size),
        )
        self.d1 = ConvDown(d1, d2)

        self.e2 = nn.Sequential(
            WindowAttentionBlock(d2, num_heads=4, window_size=window_size),
            WindowAttentionBlock(d2, num_heads=4, window_size=window_size),
        )
        self.d2 = ConvDown(d2, d3)

        self.mid = nn.Sequential(
            WindowAttentionBlock(d3, num_heads=8, window_size=window_size),
            WindowAttentionBlock(d3, num_heads=8, window_size=window_size),
        )

        self.u2 = ConvUp(d3, d2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(d2, d2, 3, padding=1),
            nn.GELU(),
            WindowAttentionBlock(d2, num_heads=4, window_size=window_size),
        )

        self.u1 = ConvUp(d2, d1)
        self.dec1 = nn.Sequential(
            nn.Conv2d(d1, d1, 3, padding=1),
            nn.GELU(),
            WindowAttentionBlock(d1, num_heads=4, window_size=window_size),
        )

        self.out_proj = nn.Conv2d(d1, out_ch, 3, padding=1)

    def forward(self, x):
        x1 = self.in_proj(x)
        s1 = self.e1(x1)
        x2 = self.d1(s1)
        s2 = self.e2(x2)
        x3 = self.d2(s2)
        x3 = self.mid(x3)

        y2 = self.u2(x3)
        if y2.shape[-2:] != s2.shape[-2:]:
            y2 = F.interpolate(y2, size=s2.shape[-2:], mode="bilinear", align_corners=False)
        y2 = self.dec2(y2 + s2)

        y1 = self.u1(y2)
        if y1.shape[-2:] != s1.shape[-2:]:
            y1 = F.interpolate(y1, size=s1.shape[-2:], mode="bilinear", align_corners=False)
        y1 = self.dec1(y1 + s1)

        out = torch.sigmoid(self.out_proj(y1))
        return out


def build_model(name: str) -> nn.Module:
    if name == "nafnet":
        model = NAFNetSmall(in_ch=30, out_ch=29, width=32, enc_blocks=(1, 1, 2), dec_blocks=(1, 1, 1), mid_blocks=4)
    elif name == "uformer":
        model = TinyUformer(in_ch=30, out_ch=29, dims=(32, 64, 128), window_size=8)
    else:
        raise ValueError(f"Unknown model {name}")
    return model


# ============================================================
# TRAIN / VAL
# ============================================================
def make_loaders(cache_records, train_idx, val_idx):
    train_records = [cache_records[i] for i in train_idx]
    val_records = [cache_records[i] for i in val_idx]

    train_ds = HSICacheDataset(train_records, augment=True)
    val_ds = HSICacheDataset(val_records, augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=NUM_WORKERS > 0,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=NUM_WORKERS > 0,
        drop_last=False,
    )
    return train_loader, val_loader


def save_checkpoint(save_path, model, optimizer, scheduler, scaler, epoch, best_val_psnr, history, config):
    module = model.module if isinstance(model, nn.DataParallel) else model
    ckpt = {
        "model_state": module.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "best_val_psnr": best_val_psnr,
        "history": history,
        "config": config,
    }
    torch.save(ckpt, save_path)


@torch.no_grad()
def evaluate(model, loader, coder, use_forward_loss=False):
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    total_sam = 0.0
    total_count = 0

    pbar = tqdm(loader, desc="val", leave=False)
    for batch in pbar:
        inp = batch["input"].to(DEVICE, non_blocking=True)
        target = batch["target"].to(DEVICE, non_blocking=True)
        measurement = batch["measurement"].to(DEVICE, non_blocking=True)

        with autocast(enabled=AMP and DEVICE.type == "cuda"):
            pred = model(inp)
            l1 = F.l1_loss(pred, target)
            sam = sam_loss(pred, target)
            loss = LAMBDA_L1 * l1 + LAMBDA_SAM * sam
            if use_forward_loss:
                y_hat = coder(pred)
                y_hat = y_hat / (y_hat.amax(dim=(-2, -1), keepdim=True) + 1e-8)
                fwd = F.l1_loss(y_hat, measurement)
                loss = loss + LAMBDA_FWD * fwd

        bs = inp.size(0)
        total_loss += loss.item() * bs
        total_psnr += psnr_metric(pred, target).item() * bs
        total_sam += sam_deg_metric(pred, target).item() * bs
        total_count += bs

        pbar.set_postfix(
            loss=f"{total_loss / total_count:.4f}",
            psnr=f"{total_psnr / total_count:.2f}",
            sam=f"{total_sam / total_count:.2f}",
        )

    return {
        "loss": total_loss / max(total_count, 1),
        "psnr": total_psnr / max(total_count, 1),
        "sam_deg": total_sam / max(total_count, 1),
    }


def train_one_experiment(exp_cfg, train_loader, val_loader, coder):
    run_dir = RUNS_DIR / exp_cfg["name"]
    run_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(exp_cfg["model"]).to(DEVICE)
    n_params = count_parameters(model)
    print(f"{exp_cfg['name']} params = {n_params / 1e6:.3f}M")
    if n_params > 6_000_000:
        raise ValueError(f"Model {exp_cfg['name']} has > 6M params")

    if DEVICE.type == "cuda" and N_GPUS > 1 and USE_DATAPARALLEL:
        model = nn.DataParallel(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR * 0.1)
    scaler = GradScaler(enabled=AMP and DEVICE.type == "cuda")

    history = {
        "train_loss": [], "val_loss": [],
        "train_psnr": [], "val_psnr": [],
        "train_sam_deg": [], "val_sam_deg": [],
        "lr": [],
    }

    best_val_psnr = -1e9
    best_epoch = -1
    epochs_no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        total_loss = 0.0
        total_psnr = 0.0
        total_sam = 0.0
        total_count = 0

        pbar = tqdm(train_loader, desc=f"epoch {epoch:03d} train", leave=False)
        for step, batch in enumerate(pbar, start=1):
            inp = batch["input"].to(DEVICE, non_blocking=True)
            target = batch["target"].to(DEVICE, non_blocking=True)
            measurement = batch["measurement"].to(DEVICE, non_blocking=True)

            with autocast(enabled=AMP and DEVICE.type == "cuda"):
                pred = model(inp)
                l1 = F.l1_loss(pred, target)
                sam = sam_loss(pred, target)
                loss = LAMBDA_L1 * l1 + LAMBDA_SAM * sam
                if exp_cfg["use_forward_loss"]:
                    y_hat = coder(pred)
                    y_hat = y_hat / (y_hat.amax(dim=(-2, -1), keepdim=True) + 1e-8)
                    fwd = F.l1_loss(y_hat, measurement)
                    loss = loss + LAMBDA_FWD * fwd
                loss = loss / GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()

            if step % GRAD_ACCUM_STEPS == 0 or step == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            bs = inp.size(0)
            with torch.no_grad():
                pred_det = pred.detach()
                batch_loss = loss.item() * GRAD_ACCUM_STEPS
                batch_psnr = psnr_metric(pred_det, target).item()
                batch_sam = sam_deg_metric(pred_det, target).item()

            total_loss += batch_loss * bs
            total_psnr += batch_psnr * bs
            total_sam += batch_sam * bs
            total_count += bs

            pbar.set_postfix(
                loss=f"{total_loss / total_count:.4f}",
                psnr=f"{total_psnr / total_count:.2f}",
                sam=f"{total_sam / total_count:.2f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

        scheduler.step()

        train_metrics = {
            "loss": total_loss / max(total_count, 1),
            "psnr": total_psnr / max(total_count, 1),
            "sam_deg": total_sam / max(total_count, 1),
        }
        val_metrics = evaluate(model, val_loader, coder, use_forward_loss=exp_cfg["use_forward_loss"])

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_psnr"].append(train_metrics["psnr"])
        history["val_psnr"].append(val_metrics["psnr"])
        history["train_sam_deg"].append(train_metrics["sam_deg"])
        history["val_sam_deg"].append(val_metrics["sam_deg"])
        history["lr"].append(optimizer.param_groups[0]["lr"])

        plot_path = run_dir / "history.png"
        plot_history(history, str(plot_path), title=exp_cfg["name"])

        summary_line = (
            f"[{exp_cfg['name']}] epoch {epoch:03d} | "
            f"train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} | "
            f"train_psnr={train_metrics['psnr']:.2f} val_psnr={val_metrics['psnr']:.2f} | "
            f"train_sam={train_metrics['sam_deg']:.2f} val_sam={val_metrics['sam_deg']:.2f}"
        )
        print(summary_line)

        if val_metrics["psnr"] > best_val_psnr:
            best_val_psnr = val_metrics["psnr"]
            best_epoch = epoch
            epochs_no_improve = 0
            save_checkpoint(
                run_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_val_psnr=best_val_psnr,
                history=history,
                config=exp_cfg,
            )
        else:
            epochs_no_improve += 1

        if SAVE_EVERY_EPOCH:
            save_checkpoint(
                run_dir / "last.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_val_psnr=best_val_psnr,
                history=history,
                config=exp_cfg,
            )

        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"Early stop for {exp_cfg['name']} at epoch {epoch}")
            break

        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    return {
        "experiment": exp_cfg["name"],
        "model": exp_cfg["model"],
        "use_forward_loss": exp_cfg["use_forward_loss"],
        "params": n_params,
        "best_val_psnr": best_val_psnr,
        "best_epoch": best_epoch,
        "final_val_loss": history["val_loss"][-1],
        "final_val_sam_deg": history["val_sam_deg"][-1],
        "run_dir": str(run_dir),
    }


# ============================================================
# MAIN
# ============================================================
records = scan_hsi_files(DATASET_DIRS)
print(pd.DataFrame(records).groupby("source").size().rename("n_files").reset_index())
print(f"total samples = {len(records)}")

if ALIGN_DATASET_MEANS:
    dataset_means, gains, gain_target = compute_dataset_means(records)
else:
    dataset_means = {k: None for k in DATASET_DIRS.keys()}
    gains = {k: 1.0 for k in DATASET_DIRS.keys()}
    gain_target = None

print("dataset means:", dataset_means)
print("dataset gains:", gains)
print("gain target:", gain_target)

coder = SimpleSpectralCoder(
    n_bands=N_BANDS,
    hw=IMG_HW,
    psf_sigma=PSF_SIGMA,
    n_orders=N_ORDERS,
    first_radius=FIRST_RADIUS,
    radius_step=RADIUS_STEP,
    center_weight=PSF_CENTER_WEIGHT,
    order_weight=PSF_ORDER_WEIGHT,
    zero_blur_sigma=ZERO_ORDER_BLUR_SIGMA,
    noise_std=NOISE_STD,
)

cache_records = build_cache(records, gains, coder, rebuild=False)
train_idx, val_idx = stratified_split(cache_records, val_total=VAL_SAMPLES, seed=SEED)
print(f"train={len(train_idx)}, val={len(val_idx)}")
print("train split by source:")
print(pd.DataFrame([cache_records[i] for i in train_idx]).groupby("source").size().rename("n").reset_index())
print("val split by source:")
print(pd.DataFrame([cache_records[i] for i in val_idx]).groupby("source").size().rename("n").reset_index())

train_loader, val_loader = make_loaders(cache_records, train_idx, val_idx)
coder = coder.to(DEVICE).eval()

results = []
for exp_cfg in EXPERIMENTS:
    print("=" * 90)
    print(f"Starting experiment: {exp_cfg['name']}")
    result = train_one_experiment(exp_cfg, train_loader, val_loader, coder)
    results.append(result)

    results_df = pd.DataFrame(results).sort_values("best_val_psnr", ascending=False)
    results_df.to_csv(ROOT_OUT / "results.csv", index=False)
    print(results_df)

    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

print("\nDone.")
print(f"All outputs saved under: {ROOT_OUT}")
print(pd.DataFrame(results).sort_values("best_val_psnr", ascending=False))
