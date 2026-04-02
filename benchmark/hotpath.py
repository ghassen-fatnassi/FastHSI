from __future__ import annotations

from typing import Tuple, Dict
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity, record_function, schedule


# =========================================================
# config
# =========================================================
DEVICE = "cuda"
DTYPE = torch.float32

C = 8
K = 8
IMG_H = 256
IMG_W = 256
CANVAS_H = 2048
CANVAS_W = 2048
LOCAL_PSF_SIZE = 33
LOCAL_SIGMA_BASE = 2.0
CHUNK = 2

BATCHES = [1, 2]
WARMUP_ITERS = 20
STEADY_ITERS = 100

PROFILE_TRACE_DIR = "./traces"
PROFILE_ACTIVE_STEPS = 8
PROFILE_WARMUP_ITERS = 5
DO_PROFILE = True

torch.manual_seed(0)
assert torch.cuda.is_available(), "CUDA required"


# =========================================================
# helpers
# =========================================================
def _next_smooth(n: int) -> int:
    c = n
    while True:
        x = c
        for p in (2, 3, 5):
            while x % p == 0:
                x //= p
        if x == 1:
            return c
        c += 1


def _padarray_3d(image: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
    pad_H = target_hw[0] - image.shape[-2]
    pad_W = target_hw[1] - image.shape[-1]
    if pad_H < 0 or pad_W < 0:
        raise ValueError(f"target_hw {target_hw} smaller than image {image.shape[-2:]}")
    padding = (
        pad_W // 2, pad_W - pad_W // 2,
        pad_H // 2, pad_H - pad_H // 2,
    )
    return F.pad(image, padding, mode="constant", value=0.0)


def cuda_peak_mb() -> float:
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def report_err(tag: str, ref: torch.Tensor, out: torch.Tensor):
    abs_err = (ref - out).abs()
    rel_l2 = abs_err.norm() / ref.norm().clamp_min(1e-12)
    print(
        f"{tag:>24} | rel_l2 {rel_l2.item():.6e} | "
        f"mean_abs {abs_err.mean().item():.6e} | "
        f"max_abs {abs_err.max().item():.6e}"
    )


# =========================================================
# synthetic PSF setup
# =========================================================
def make_spot_coords(
    C: int,
    K: int,
    H: int,
    W: int,
    img_h: int,
    img_w: int,
    device: str = "cuda",
) -> torch.Tensor:
    margin_y = img_h // 2 + 8
    margin_x = img_w // 2 + 8

    ys = torch.linspace(margin_y, H - margin_y - 1, K, device=device)
    xs = torch.linspace(margin_x, W - margin_x - 1, K, device=device)
    base = torch.stack([ys, xs.roll(1)], dim=-1)

    coords = []
    for c in range(C):
        cur = base.clone()
        cur[:, 0] += (c - C // 2) * 7.0
        cur[:, 1] += (c - C // 2) * 11.0
        cur[:, 0].clamp_(margin_y, H - margin_y - 1)
        cur[:, 1].clamp_(margin_x, W - margin_x - 1)
        coords.append(cur)
    return torch.stack(coords, dim=0)  # (C,K,2)


def make_local_psfs(
    C: int,
    K: int,
    S: int,
    sigma_base: float,
    device: str = "cuda",
) -> torch.Tensor:
    ax = torch.arange(S, device=device, dtype=torch.float32) - (S // 2)
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    local = torch.empty((C, K, S, S), device=device, dtype=torch.float32)

    for c in range(C):
        for k in range(K):
            sigma_y = sigma_base + 0.15 * c + 0.10 * k
            sigma_x = sigma_base + 0.12 * c + 0.08 * k + 0.30
            mu_y = -0.15 * ((c % 3) - 1) + 0.04 * k
            mu_x = 0.20 * ((k % 3) - 1) + 0.05 * c

            g = torch.exp(
                -0.5 * (((yy - mu_y) / sigma_y) ** 2 + ((xx - mu_x) / sigma_x) ** 2)
            )
            g /= g.sum().clamp_min(1e-12)
            local[c, k] = g
    return local


def embed_full_canvas_psf(
    coords: torch.Tensor,      # (C,K,2), absolute canvas centers
    local_psfs: torch.Tensor,  # (C,K,S,S)
    H: int,
    W: int,
    img_h: int,
    img_w: int,
    anchor_mode: str = "floor",
) -> torch.Tensor:
    device = coords.device
    C, K, _ = coords.shape
    S = local_psfs.shape[-1]
    half = S // 2

    if anchor_mode == "floor":
        gc_y = (H - img_h) // 2 + (img_h - 1) // 2
        gc_x = (W - img_w) // 2 + (img_w - 1) // 2
    elif anchor_mode == "half":
        gc_y = (H - img_h) // 2 + img_h // 2
        gc_x = (W - img_w) // 2 + img_w // 2
    else:
        raise ValueError(anchor_mode)

    out = torch.zeros((1, C, H, W), device=device, dtype=torch.float32)

    for c in range(C):
        canvas = torch.zeros((H, W), device=device, dtype=torch.float32)
        for k in range(K):
            local = local_psfs[c, k]

            tmp = torch.zeros((H, W), device=device, dtype=torch.float32)
            y0 = H // 2 - half
            x0 = W // 2 - half
            tmp[y0:y0 + S, x0:x0 + S] = local

            base = torch.fft.ifftshift(tmp)

            cy = int(round(coords[c, k, 0].item()))
            cx = int(round(coords[c, k, 1].item()))
            dy = cy - gc_y
            dx = cx - gc_x

            canvas += torch.roll(base, shifts=(dy, dx), dims=(0, 1))

        out[0, c] = canvas

    return out


# =========================================================
# impl 1: old full canvas fft
# =========================================================
def build_old_fft_psf(psf_tensor: torch.Tensor) -> torch.Tensor:
    with record_function("old_full_canvas_fft.build_fft_psf"):
        return torch.fft.fft2(psf_tensor.float(), dim=(-2, -1))


def old_forward_hotpath(x: torch.Tensor, fft_psf: torch.Tensor, chunk: int = 2) -> torch.Tensor:
    with record_function("old_full_canvas_fft.forward"):
        _, C, H, W = fft_psf.shape
        B = x.shape[0]

        summed = torch.zeros((B, H, W), dtype=torch.complex64, device=x.device)

        for i in range(0, C, chunk):
            e = min(i + chunk, C)
            with record_function(f"old_full_canvas_fft.chunk_{i}_{e}"):
                x_padded = _padarray_3d(x[:, i:e], (H, W))
                x_fft = torch.fft.fft2(x_padded, dim=(-2, -1))
                summed += (fft_psf[:, i:e] * x_fft).sum(dim=1)

        y = torch.fft.ifft2(summed, dim=(-2, -1)).real
        return y.clamp_(min=0.0).div_(C).unsqueeze(1)


# =========================================================
# impl 2: mono rfft
# =========================================================
class ForwardMonoRFFT(nn.Module):
    def __init__(self, psf: torch.Tensor, chunk: int = 2):
        super().__init__()
        pH, pW = psf.shape[-2], psf.shape[-1]
        fH, fW = _next_smooth(pH), _next_smooth(pW)
        rW = fW // 2 + 1

        self.psf_hw = (pH, pW)
        self.fft_hw = (fH, fW)
        self.rfft_w = rW
        self.chunk = chunk
        self.crop_h = (fH - pH) // 2
        self.crop_w = (fW - pW) // 2

        with record_function("mono_rfft.build_psf_rfft"):
            psf_pad = _padarray_3d(psf.float(), (fH, fW))
            psf_rfft = torch.fft.rfft2(psf_pad, dim=(-2, -1)).squeeze(0)
            self.register_buffer("psf_rfft", psf_rfft)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with record_function("mono_rfft.forward"):
            B, C, _, _ = x.shape
            pH, pW = self.psf_hw
            fH, fW = self.fft_hw
            ph, pw = self.crop_h, self.crop_w

            acc = torch.zeros((B, fH, self.rfft_w), dtype=torch.complex64, device=x.device)

            for i in range(0, C, self.chunk):
                e = min(i + self.chunk, C)
                with record_function(f"mono_rfft.chunk_{i}_{e}"):
                    xc_pad = _padarray_3d(x[:, i:e], (fH, fW))
                    xc = torch.fft.rfft2(xc_pad, dim=(-2, -1))
                    acc.add_((xc * self.psf_rfft[i:e].unsqueeze(0)).sum(dim=1))

            y = torch.fft.irfft2(acc, s=(fH, fW), dim=(-2, -1))
            y = y[:, ph:ph + pH, pw:pw + pW]
            return y.clamp_(min=0.0).div_(C).unsqueeze(1)


# =========================================================
# impl 3: physical baseline, no loops over channels or psfs
# =========================================================
class ForwardConvScatterNoLoops(nn.Module):
    def __init__(
        self,
        coords: torch.Tensor,        # (C,K,2)
        local_psfs: torch.Tensor,    # (C,K,S,S)
        canvas_hw: Tuple[int, int],
        anchor_mode: str = "floor",
    ):
        super().__init__()

        assert coords.ndim == 3 and coords.shape[-1] == 2
        assert local_psfs.ndim == 4
        assert coords.shape[:2] == local_psfs.shape[:2]

        self.canvas_hw = canvas_hw
        self.anchor_mode = anchor_mode

        C, K, _ = coords.shape
        S = local_psfs.shape[-1]
        assert local_psfs.shape[-2] == S

        self.C = C
        self.K = K
        self.M = C * K
        self.S = S
        self.r = S // 2

        self.register_buffer("coords", coords.float())
        self.register_buffer("local_psfs", local_psfs.float())

        weight = local_psfs.flip(-2, -1).reshape(self.M, 1, S, S)
        self.register_buffer("weight", weight)

        coords_flat = coords.reshape(self.M, 2).float()
        self.register_buffer("coords_flat", coords_flat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with record_function("conv_scatter_noloops.forward"):
            B, C, H, W = x.shape
            canvas_H, canvas_W = self.canvas_hw
            assert C == self.C

            if self.anchor_mode == "floor":
                anchor_y = (H - 1) // 2
                anchor_x = (W - 1) // 2
            elif self.anchor_mode == "half":
                anchor_y = H // 2
                anchor_x = W // 2
            else:
                raise ValueError(self.anchor_mode)

            with record_function("conv_scatter_noloops.expand_input"):
                xr = (
                    x.unsqueeze(2)
                    .expand(B, C, self.K, H, W)
                    .reshape(B, self.M, H, W)
                )

            with record_function("conv_scatter_noloops.grouped_conv2d"):
                yc = F.conv2d(
                    xr,
                    self.weight,
                    padding=self.S - 1,
                    groups=self.M,
                )  # (B,M,Hf,Wf)

            B, M, Hf, Wf = yc.shape

            with record_function("conv_scatter_noloops.index_build"):
                cy = self.coords_flat[:, 0].round().long()
                cx = self.coords_flat[:, 1].round().long()

                y0 = cy - anchor_y - self.r
                x0 = cx - anchor_x - self.r

                py = torch.arange(Hf, device=x.device).view(1, Hf, 1)
                px = torch.arange(Wf, device=x.device).view(1, 1, Wf)

                dst_y = y0.view(M, 1, 1) + py
                dst_x = x0.view(M, 1, 1) + px

                valid = (
                    (dst_y >= 0) & (dst_y < canvas_H) &
                    (dst_x >= 0) & (dst_x < canvas_W)
                )  # (M,Hf,Wf)

                safe_y = torch.where(valid, dst_y, torch.zeros_like(dst_y))
                safe_x = torch.where(valid, dst_x, torch.zeros_like(dst_x))

                linear_idx = safe_y * canvas_W + safe_x
                linear_idx = linear_idx.reshape(1, -1).expand(B, -1)

                vals = torch.where(valid.unsqueeze(0), yc, torch.zeros_like(yc))
                vals = vals.reshape(B, -1)

            with record_function("conv_scatter_noloops.scatter_add"):
                out = torch.zeros((B, canvas_H * canvas_W), device=x.device, dtype=x.dtype)
                out.scatter_add_(dim=1, index=linear_idx, src=vals)
                out = out.view(B, 1, canvas_H, canvas_W)

            return out.clamp_(min=0.0).div_(C)


# =========================================================
# benchmark
# =========================================================
@torch.no_grad()
def benchmark_callable(name: str, fn, *args, warmup: int, iters: int) -> Dict[str, float]:
    for _ in range(warmup):
        _ = fn(*args)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    times_ms = []

    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        _ = fn(*args)
        end.record()
        torch.cuda.synchronize()

        times_ms.append(start.elapsed_time(end))

    t = torch.tensor(times_ms)
    return {
        "mean_ms": t.mean().item(),
        "median_ms": t.median().item(),
        "min_ms": t.min().item(),
        "peak_mb": cuda_peak_mb(),
    }


def print_row(batch: int, name: str, stats: Dict[str, float]):
    print(
        f"B={batch:<2d} | {name:>20} | "
        f"mean {stats['mean_ms']:8.3f} ms | "
        f"per_batch {stats['mean_ms']/batch:8.3f} ms | "
        f"median {stats['median_ms']:8.3f} ms | "
        f"min {stats['min_ms']:8.3f} ms | "
        f"peak {stats['peak_mb']:8.1f} MB"
    )


# =========================================================
# profiler
# =========================================================
def trace_handler_maker(trace_dir: str):
    Path(trace_dir).mkdir(parents=True, exist_ok=True)

    def handler(p):
        step = p.step_num
        chrome_path = os.path.join(trace_dir, f"trace_step_{step}.json")
        mem_path = os.path.join(trace_dir, f"memory_timeline_step_{step}.html")
        stacks_path = os.path.join(trace_dir, f"stacks_step_{step}.txt")

        p.export_chrome_trace(chrome_path)
        try:
            p.export_memory_timeline(mem_path, device=f"cuda:{torch.cuda.current_device()}")
        except Exception:
            pass
        try:
            p.export_stacks(stacks_path, "self_cuda_time_total")
        except Exception:
            pass

        print(f"[profiler] wrote {chrome_path}")
        print(
            p.key_averages(group_by_input_shape=True).table(
                sort_by="self_cuda_time_total",
                row_limit=40,
            )
        )

    return handler


@torch.no_grad()
def profile_one(
    name: str,
    fn,
    *args,
    warmup: int = 5,
    active: int = 8,
    trace_dir: str = "./traces",
):
    for _ in range(warmup):
        _ = fn(*args)
    torch.cuda.synchronize()

    prof_sched = schedule(wait=1, warmup=1, active=active, repeat=1)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=prof_sched,
        on_trace_ready=trace_handler_maker(os.path.join(trace_dir, name)),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
    ) as prof:
        total_steps = 1 + 1 + active
        for step in range(total_steps):
            with record_function(f"profile_step.{name}.{step}"):
                _ = fn(*args)
                torch.cuda.synchronize()
            prof.step()


# =========================================================
# run
# =========================================================
@torch.no_grad()
def main():
    with record_function("setup.make_spot_coords"):
        coords = make_spot_coords(C, K, CANVAS_H, CANVAS_W, IMG_H, IMG_W, DEVICE)

    with record_function("setup.make_local_psfs"):
        local_psfs = make_local_psfs(C, K, LOCAL_PSF_SIZE, LOCAL_SIGMA_BASE, DEVICE)

    with record_function("setup.embed_full_canvas_psf"):
        full_psf = embed_full_canvas_psf(
            coords,
            local_psfs,
            CANVAS_H,
            CANVAS_W,
            IMG_H,
            IMG_W,
            anchor_mode="floor",
        )

    
    with torch.inference_mode():
        fft_psf_old = build_old_fft_psf(full_psf)
        op_rfft = ForwardMonoRFFT(full_psf, chunk=CHUNK).to(DEVICE)
        op_scatter = ForwardConvScatterNoLoops(
            coords,
            local_psfs,
            (CANVAS_H, CANVAS_W),
            anchor_mode="floor",
        ).to(DEVICE)

        # correctness on B=1
        x_check = torch.rand((1, C, IMG_H, IMG_W), device=DEVICE, dtype=DTYPE)
        y_old = old_forward_hotpath(x_check, fft_psf_old, chunk=CHUNK)
        y_rfft = op_rfft(x_check)
        y_scatter = op_scatter(x_check)

    print("correctness vs old")
    report_err("rfft_vs_old", y_old, y_rfft)
    report_err("scatter_vs_old", y_old, y_scatter)
    print()

    all_results = []

    for batch in BATCHES:
        x = torch.rand((batch, C, IMG_H, IMG_W), device=DEVICE, dtype=DTYPE)

        stats_old = benchmark_callable(
            "old_full_canvas_fft",
            old_forward_hotpath,
            x, fft_psf_old, CHUNK,
            warmup=WARMUP_ITERS,
            iters=STEADY_ITERS,
        )
        stats_rfft = benchmark_callable(
            "mono_rfft",
            op_rfft,
            x,
            warmup=WARMUP_ITERS,
            iters=STEADY_ITERS,
        )
        stats_scatter = benchmark_callable(
            "conv_scatter_noloops",
            op_scatter,
            x,
            warmup=WARMUP_ITERS,
            iters=STEADY_ITERS,
        )

        print_row(batch, "old_full_canvas_fft", stats_old)
        print_row(batch, "mono_rfft", stats_rfft)
        print_row(batch, "conv_scatter_noloops", stats_scatter)
        print()

        all_results.append((batch, "old_full_canvas_fft", stats_old))
        all_results.append((batch, "mono_rfft", stats_rfft))
        all_results.append((batch, "conv_scatter_noloops", stats_scatter))

    print("normalized summary")
    for batch, name, stats in all_results:
        print(
            f"B={batch:<2d} | {name:>20} | "
            f"mean_per_batch {stats['mean_ms']/batch:8.3f} ms | "
            f"peak_mb {stats['peak_mb']:8.1f}"
        )

    if DO_PROFILE:
        print("\nprofiling traces...")
        x_prof = torch.rand((1, C, IMG_H, IMG_W), device=DEVICE, dtype=DTYPE)

        profile_one(
            "old_full_canvas_fft",
            old_forward_hotpath,
            x_prof, fft_psf_old, CHUNK,
            warmup=PROFILE_WARMUP_ITERS,
            active=PROFILE_ACTIVE_STEPS,
            trace_dir=PROFILE_TRACE_DIR,
        )

        profile_one(
            "mono_rfft",
            op_rfft,
            x_prof,
            warmup=PROFILE_WARMUP_ITERS,
            active=PROFILE_ACTIVE_STEPS,
            trace_dir=PROFILE_TRACE_DIR,
        )

        profile_one(
            "conv_scatter_noloops",
            op_scatter,
            x_prof,
            warmup=PROFILE_WARMUP_ITERS,
            active=PROFILE_ACTIVE_STEPS,
            trace_dir=PROFILE_TRACE_DIR,
        )


if __name__ == "__main__":
    main()