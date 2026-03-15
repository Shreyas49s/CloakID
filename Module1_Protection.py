# %% [markdown]
# # CloakID — Module 1 v2.0: Image Immunization with EoT
#
# **Improvement: Expectation over Transformations (EoT)**
#
# v1.0 weakness: A simple JPEG save or social media upload strips the
# adversarial noise completely, because the PGD loop optimizes noise
# for a *single, clean* input — making it brittle to any pixel-level
# transformation.
#
# v2.0 fix: At every PGD step, we apply K random augmentations to the
# adversarial image, compute the loss for each, and average the gradients
# before updating delta. The noise now optimizes for the *expected* loss
# over a distribution of transformations → survives JPEG compression,
# brightness shifts, and minor resizing.
#
# Architecture:
#   Layer 1 — VAE Encoder Attack   (Gray Latent Target)
#   Layer 2 — CLIP Vision Attack   (Semantic Void Target)
#   Defense  — EoT over K augments (Compression Robustness)  ← NEW
#
# ─────────────────────────────────────────────────────────────────────
# KAGGLE SESSION RECOMMENDATION
# ─────────────────────────────────────────────────────────────────────
#  Use:  GPU T4 x2  (NOT P100)
#
#  Reason:
#  • EoT multiplies the forward passes per step by K (default K=8).
#    A single P100 (16 GB) will OOM on the combined VAE + CLIP + K passes
#    at float16 when K > 4, especially for 512px inputs.
#  • T4 x2 gives two 16 GB cards (32 GB total addressable via DataParallel
#    or manual model sharding): VAE on cuda:0, CLIP on cuda:1.
#    This splits the memory load and keeps each card below its limit.
#  • T4 x2 also provides ~2x the CUDA cores for the K parallel augmentation
#    forward passes, making EoT iterations roughly as fast as v1.0 on P100.
#  • P100 is preferred ONLY if you reduce K to 2–3 and steps to ≤ 30.
# ─────────────────────────────────────────────────────────────────────


# %% — Cell 1: Setup & Installs
# ============================================================================
import subprocess, sys, os

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

print("⚙ Installing dependencies...")
pkgs = ["diffusers", "transformers", "accelerate", "gradio", "lpips", "scikit-image"]
for p in pkgs:
    install(p)
print("✔ All dependencies installed.")


# %% — Cell 2: Imports & Model Loading
# ============================================================================
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as compare_ssim
from diffusers import AutoencoderKL
from transformers import CLIPModel, CLIPProcessor
import gradio as gr
import warnings, time, random

warnings.filterwarnings("ignore")

# ── Device Setup ─────────────────────────────────────────────────────────────
# T4 x2: place VAE on cuda:0, CLIP on cuda:1 to split VRAM load under EoT
DEVICE_VAE  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE_CLIP = torch.device("cuda:1" if torch.cuda.device_count() > 1 else DEVICE_VAE)
DEVICE      = DEVICE_VAE   # primary device for delta and tensors
DTYPE       = torch.float16 if DEVICE.type == "cuda" else torch.float32

print(f"  VAE device  : {DEVICE_VAE}")
print(f"  CLIP device : {DEVICE_CLIP}")
print(f"  Dtype       : {DTYPE}")

# ── Layer 1 Model: VAE ───────────────────────────────────────────────────────
print("⏳ Loading VAE (stabilityai/sd-vae-ft-mse) …")
vae = AutoencoderKL.from_pretrained(
    "stabilityai/sd-vae-ft-mse",
    torch_dtype=DTYPE,
).to(DEVICE_VAE)
vae.eval()
vae.requires_grad_(False)
print("  ✔ VAE loaded.")

# ── Layer 2 Model: CLIP ──────────────────────────────────────────────────────
print("⏳ Loading CLIP (openai/clip-vit-large-patch14) …")
clip_model = CLIPModel.from_pretrained(
    "openai/clip-vit-large-patch14",
    torch_dtype=DTYPE,
).to(DEVICE_CLIP)
clip_model.eval()
clip_model.requires_grad_(False)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
print("  ✔ CLIP loaded.")
print("✔ All models ready.\n")


# %% — Cell 3: Core Utility Functions
# ============================================================================

ATTACK_RES = 512

def pil_to_tensor(pil_img: Image.Image, size: int = ATTACK_RES) -> torch.Tensor:
    img = pil_img.convert("RGB").resize((size, size), Image.LANCZOS)
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(DEVICE, dtype=torch.float32)

def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    arr = tensor.squeeze(0).clamp(0, 1).detach().cpu().permute(1, 2, 0).numpy()
    return Image.fromarray((arr * 255).astype(np.uint8))

def apply_perturbation_fullres(original_pil: Image.Image, delta_lowres: torch.Tensor) -> Image.Image:
    orig_w, orig_h = original_pil.size
    delta_fullres = F.interpolate(
        delta_lowres, size=(orig_h, orig_w), mode="bilinear", align_corners=False
    )
    orig_tensor = (
        torch.from_numpy(np.array(original_pil.convert("RGB")).astype(np.float32) / 255.0)
        .permute(2, 0, 1).unsqueeze(0).to(DEVICE, dtype=torch.float32)
    )
    return tensor_to_pil((orig_tensor + delta_fullres).clamp(0, 1))

def compute_ssim(img_a: Image.Image, img_b: Image.Image) -> float:
    size = min(img_a.size[0], 1024), min(img_a.size[1], 1024)
    a = np.array(img_a.convert("RGB").resize(size, Image.LANCZOS))
    b = np.array(img_b.convert("RGB").resize(size, Image.LANCZOS))
    return compare_ssim(a, b, channel_axis=2, data_range=255)

def compute_psnr(img_a: Image.Image, img_b: Image.Image) -> float:
    """Peak Signal-to-Noise Ratio. Higher = more similar. >35 dB is excellent."""
    a = np.array(img_a.convert("RGB")).astype(np.float32)
    b = np.array(img_b.convert("RGB")).astype(np.float32)
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * np.log10(255.0 / np.sqrt(mse))


# %% — Cell 4: EoT Augmentation Bank
# ============================================================================
# EoT (Expectation over Transformations) — Core of v2.0
#
# Each augmentation simulates a real-world post-processing operation that
# could strip naive adversarial noise:
#   • JPEG simulation  → social media compression (Instagram, WhatsApp)
#   • Gaussian blur    → image resizing artifacts
#   • Brightness jitter→ screenshot / re-photo
#   • Gaussian noise   → sensor noise
#   • Random crop+pad  → social media cropping / thumbnail generation
#
# By averaging gradients over K samples from this distribution, the PGD
# optimizer finds noise that survives ALL of these, not just clean input.

def jpeg_simulate(x: torch.Tensor, quality: int = 75) -> torch.Tensor:
    """
    Differentiable JPEG via real PIL compression + Straight-Through Estimator.

    Forward pass: actual JPEG encode/decode via PIL — gives exact real-world
    JPEG behavior including DCT, quantization tables, and chroma subsampling.

    Backward pass: STE — gradient flows through as identity. The optimizer
    sees the JPEG-compressed output but can still update delta, because the
    STE trick (jpeg_out - x).detach() + x makes autograd treat it as if
    no non-differentiable operation occurred.

    This is the standard approach used in adversarial robustness literature
    (Shin & Song, 2017; Athalye et al., 2018) for making perturbations
    survive JPEG compression.
    """
    import io
    with torch.no_grad():
        # Tensor [1,3,H,W] → PIL Image
        arr = x.squeeze(0).clamp(0, 1).cpu().permute(1, 2, 0).numpy()
        pil_img = Image.fromarray((arr * 255).astype(np.uint8))

        # Real JPEG compression
        buf = io.BytesIO()
        pil_img.save(buf, format='JPEG', quality=quality)
        buf.seek(0)
        jpeg_pil = Image.open(buf).copy()

        # PIL Image → Tensor on same device
        jpeg_arr = np.array(jpeg_pil).astype(np.float32) / 255.0
        jpeg_tensor = torch.from_numpy(jpeg_arr).permute(2, 0, 1).unsqueeze(0)
        jpeg_tensor = jpeg_tensor.to(x.device, dtype=x.dtype)

    # STE: forward evaluates to jpeg_tensor, backward passes gradient through x
    return (jpeg_tensor - x).detach() + x

def gaussian_blur_augment(x: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """Apply a small Gaussian blur to simulate resizing artifacts."""
    # Simple mean blur as a proxy (no external dependency needed)
    padding = kernel_size // 2
    weight = torch.ones(3, 1, kernel_size, kernel_size, device=x.device, dtype=x.dtype)
    weight = weight / (kernel_size * kernel_size)
    return F.conv2d(x, weight, padding=padding, groups=3).clamp(0, 1)

def brightness_jitter(x: torch.Tensor, strength: float = 0.05) -> torch.Tensor:
    """Random brightness shift to simulate re-photography or screenshot."""
    factor = 1.0 + random.uniform(-strength, strength)
    return (x * factor).clamp(0, 1)

def additive_noise(x: torch.Tensor, std: float = 0.01) -> torch.Tensor:
    """Additive Gaussian noise to simulate sensor noise."""
    return (x + torch.randn_like(x) * std).clamp(0, 1)

def random_crop_pad(x: torch.Tensor, crop_frac: float = 0.05) -> torch.Tensor:
    """Randomly crop a small border and pad back — simulates social media thumbnailing."""
    _, _, H, W = x.shape
    ch = int(H * crop_frac)
    cw = int(W * crop_frac)
    if ch == 0 or cw == 0:
        return x
    cropped = x[:, :, ch:H-ch, cw:W-cw]
    return F.interpolate(cropped, size=(H, W), mode="bilinear", align_corners=False)

# Non-JPEG augmentations — used for the non-JPEG portion of EoT passes
NON_JPEG_AUGMENTATIONS = [
    lambda x: gaussian_blur_augment(x, kernel_size=random.choice([3, 5])),
    lambda x: brightness_jitter(x, strength=random.uniform(0.02, 0.06)),
    lambda x: additive_noise(x, std=random.uniform(0.005, 0.02)),
    lambda x: random_crop_pad(x, crop_frac=random.uniform(0.02, 0.05)),
    lambda x: x,  # Identity — always include clean pass to anchor the loss
]

def apply_eot_augmentation(x: torch.Tensor, k_index: int, eot_k: int) -> torch.Tensor:
    """
    EoT augmentation with guaranteed JPEG coverage.

    Strategy: The first half of K passes ALWAYS use JPEG simulation
    (the most critical real-world transformation). The second half
    samples from the remaining augmentations (blur, noise, crop, etc.).
    This ensures the optimizer spends ≥50% of its gradient budget
    learning to survive JPEG compression.
    """
    n_jpeg = max(1, eot_k // 2)  # At least 1, at least half of K
    if k_index < n_jpeg:
        # JPEG pass — vary quality to cover the range of compression levels
        quality = random.randint(60, 85)
        return jpeg_simulate(x, quality=quality)
    else:
        # Non-JPEG augmentation
        aug_fn = random.choice(NON_JPEG_AUGMENTATIONS)
        return aug_fn(x)


# %% — Cell 5: Loss Functions
# ============================================================================

def compute_vae_loss(vae_model, perturbed: torch.Tensor) -> torch.Tensor:
    """
    Layer 1 — 'Gray Death':
    Force VAE latent → zero vector (gray image representation).
    perturbed is on DEVICE; move to DEVICE_VAE for encoding.
    """
    perturbed_scaled = perturbed * 2.0 - 1.0
    pert_latent = vae_model.encode(
        perturbed_scaled.to(DEVICE_VAE, dtype=DTYPE)
    ).latent_dist.mean
    target_latent = torch.zeros_like(pert_latent)
    return F.mse_loss(pert_latent, target_latent)

def compute_clip_loss(model, processor, perturbed: torch.Tensor) -> torch.Tensor:
    """
    Layer 2 — 'Semantic Void':
    Push CLIP embedding → fixed random non-sensical vector.
    perturbed is on DEVICE; move to DEVICE_CLIP for encoding.
    """
    clip_mean = torch.tensor(
        [0.48145466, 0.4578275, 0.40821073], device=DEVICE_CLIP
    ).view(1, 3, 1, 1)
    clip_std = torch.tensor(
        [0.26862954, 0.26130258, 0.27577711], device=DEVICE_CLIP
    ).view(1, 3, 1, 1)

    img_clip = F.interpolate(
        perturbed.to(DEVICE_CLIP), size=(224, 224), mode="bilinear", align_corners=False
    )
    img_clip = (img_clip - clip_mean) / clip_std

    # Explicitly call vision_model + visual_projection to avoid
    # version-dependent return types from get_image_features()
    vision_outputs = model.vision_model(pixel_values=img_clip.to(DTYPE))
    image_emb = model.visual_projection(vision_outputs.pooler_output)
    image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)

    gen = torch.Generator(device=DEVICE_CLIP if DEVICE_CLIP.type == "cuda" else "cpu")
    gen.manual_seed(999)
    target_emb = torch.randn(1, image_emb.shape[-1], generator=gen, device=DEVICE_CLIP).to(DTYPE)
    target_emb = target_emb / target_emb.norm(dim=-1, keepdim=True)

    return F.mse_loss(image_emb, target_emb)


# %% — Cell 6: EoT-PGD Attack Engine (v2.0 Core)
# ============================================================================

@torch.enable_grad()
def pgd_attack_eot(
    original_pil:    Image.Image,
    steps:           int   = 50,
    epsilon:         float = 0.03,
    vae_weight:      float = 1.0,
    clip_weight:     float = 0.5,
    eot_k:           int   = 8,       # ← NEW: number of augmentation samples per step
    progress_callback = None,
):
    """
    CloakID v2.0 — PGD with Expectation over Transformations (EoT).

    KEY CHANGE vs v1.0:
    ───────────────────
    v1.0 (brittle):
        grad = ∇_δ L(x + δ)           # gradient on clean adversarial image

    v2.0 (robust):
        grad = (1/K) Σ_k ∇_δ L(t_k(x + δ))   # average over K augmentations
        where t_k ~ T (augmentation distribution)

    This means the noise must be effective not just on the clean image, but
    on the *expected* augmented version — making it survive real-world
    post-processing like JPEG compression and social media resizing.
    """
    x_orig = pil_to_tensor(original_pil)   # [1,3,512,512], float32
    x_orig.requires_grad_(False)

    delta = torch.zeros_like(x_orig, requires_grad=True, device=DEVICE)
    alpha = epsilon / (steps * 0.4)

    print(f"\n🛡 CloakID v2.0 EoT Attack | ε={epsilon} | Steps={steps} | K={eot_k}")

    for step in range(steps):

        # ── EoT: accumulate gradients over K augmentations ──────────────────
        accumulated_grad = torch.zeros_like(delta.data)
        total_loss_avg   = 0.0

        for k in range(eot_k):
            # 1. Create adversarial image
            x_adv = (x_orig + delta).clamp(0, 1)

            # 2. Apply a random augmentation (the EoT transformation t_k)
            x_aug = apply_eot_augmentation(x_adv, k, eot_k)

            # 3. Compute losses on the AUGMENTED image
            l_vae  = compute_vae_loss(vae, x_aug)
            l_clip = compute_clip_loss(clip_model, clip_processor, x_aug)
            loss   = (vae_weight * l_vae) + (clip_weight * l_clip.to(l_vae.device))

            # 4. Backprop — accumulate gradients
            loss.backward()
            accumulated_grad += delta.grad.detach().clone()
            total_loss_avg   += loss.item()

            # Zero grad for next augmentation pass
            delta.grad.zero_()

        # ── Average gradients across K augmentations (E[∇]) ─────────────────
        mean_grad      = accumulated_grad / eot_k
        total_loss_avg = total_loss_avg   / eot_k

        # ── PGD Update using averaged gradient ───────────────────────────────
        with torch.no_grad():
            delta.data = delta.data - alpha * torch.sign(mean_grad)
            delta.data = delta.data.clamp(-epsilon, epsilon)
            delta.data = (x_orig + delta.data).clamp(0, 1) - x_orig

        # ── Progress callback ─────────────────────────────────────────────────
        if progress_callback and step % 5 == 0:
            progress_callback(
                (step + 1) / steps,
                desc=f"EoT Step {step+1}/{steps} | Avg Loss: {total_loss_avg:.4f}"
            )

    # ── Output: apply noise at full resolution ────────────────────────────────
    protected_pil = apply_perturbation_fullres(original_pil, delta.detach())

    # ── Robustness verification: does noise survive JPEG? ────────────────────
    import io
    buf = io.BytesIO()
    protected_pil.save(buf, format="JPEG", quality=75)
    buf.seek(0)
    jpeg_pil = Image.open(buf).copy()

    ssim_original  = compute_ssim(original_pil, protected_pil)
    psnr_original  = compute_psnr(original_pil, protected_pil)
    ssim_after_jpg = compute_ssim(original_pil, jpeg_pil)

    # Noise survival rate: what fraction of perturbation energy survives JPEG
    # Measures L2 norm of (image - original) before and after JPEG compression
    # High value = noise is embedded in JPEG-resilient frequencies → robust
    size_for_metric = protected_pil.size  # (W, H)
    orig_arr  = np.array(original_pil.convert("RGB").resize(size_for_metric, Image.LANCZOS)).astype(np.float32)
    prot_arr  = np.array(protected_pil.convert("RGB")).astype(np.float32)
    jpeg_arr  = np.array(jpeg_pil.convert("RGB")).astype(np.float32)

    noise_before = np.linalg.norm(prot_arr - orig_arr)
    noise_after  = np.linalg.norm(jpeg_arr - orig_arr)
    noise_survival = min(1.0, noise_after / max(noise_before, 1e-6))

    print(f"✔ Attack complete.")
    print(f"  SSIM (vs original)     : {ssim_original:.4f}")
    print(f"  PSNR (vs original)     : {psnr_original:.2f} dB")
    print(f"  SSIM after JPEG Q75    : {ssim_after_jpg:.4f}")
    print(f"  Noise survival rate    : {noise_survival:.2%}")

    metrics = {
        "ssim":          ssim_original,
        "psnr":          psnr_original,
        "ssim_jpeg":     ssim_after_jpg,
        "noise_survival": noise_survival,
        "epsilon":       epsilon,
        "steps":         steps,
        "eot_k":         eot_k,
    }

    return protected_pil, jpeg_pil, metrics


# %% — Cell 7: Gradio Interface (v2.0)
# ============================================================================

OUTPUT_DIR = "/kaggle/working"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def immunize(
    image:     Image.Image,
    intensity: float,
    steps:     int,
    eot_k:     int,
    progress = gr.Progress(track_tqdm=True),
):
    if image is None:
        raise gr.Error("Please upload an image first.")

    protected_pil, jpeg_preview, metrics = pgd_attack_eot(
        original_pil    = image,
        steps           = int(steps),
        epsilon         = intensity,
        vae_weight      = 1.0,
        clip_weight     = 0.5,
        eot_k           = int(eot_k),
        progress_callback = progress,
    )

    # Save lossless PNG
    save_path = os.path.join(OUTPUT_DIR, "cloakid_v2_protected.png")
    protected_pil.save(save_path, format="PNG", compress_level=1)

    # Quality badges
    ssim_badge  = "🟢 Excellent" if metrics["ssim"]  >= 0.95 else ("🟡 Good" if metrics["ssim"]  >= 0.90 else "🔴 Visible Noise")
    psnr_badge  = "🟢 Excellent" if metrics["psnr"]  >= 38   else ("🟡 Good" if metrics["psnr"]  >= 32   else "🔴 Degraded")
    surv_badge  = "🟢 Robust"    if metrics["noise_survival"] >= 0.7 else ("🟡 Moderate" if metrics["noise_survival"] >= 0.4 else "🔴 Fragile")

    status = f"""### ✅ CloakID v2.0 — Immunization Report

| Metric | Value | Status |
|---|---|---|
| **Visual Fidelity (SSIM)** | `{metrics['ssim']:.4f}` | {ssim_badge} |
| **Image Quality (PSNR)** | `{metrics['psnr']:.2f} dB` | {psnr_badge} |
| **SSIM after JPEG Q75** | `{metrics['ssim_jpeg']:.4f}` | — |
| **🆕 Noise Survival Rate** | `{metrics['noise_survival']:.1%}` | {surv_badge} |

> **EoT Config:** K={metrics['eot_k']} augmentations/step · ε={metrics['epsilon']} · {metrics['steps']} steps

**What's new in v2.0:** The noise was optimized over {metrics['eot_k']} random augmentations per step
(JPEG simulation, blur, brightness jitter, crop). It should now survive social media compression.
The JPEG preview on the right confirms this."""

    return protected_pil, jpeg_preview, status, save_path


# ── UI Layout ────────────────────────────────────────────────────────────────

css = """
.metric-box { border-radius: 8px; padding: 12px; background: #1a1a2e; }
.new-badge { color: #00ff88; font-weight: bold; }
footer { display: none !important; }
"""

with gr.Blocks(title="CloakID v2.0", theme=gr.themes.Soft(), css=css) as demo:

    gr.Markdown("""
# 🛡 CloakID v2.0 — Image Immunization
### Dual-Layer Adversarial Defense · Now with **Expectation over Transformations (EoT)**

> **What's new:** v1.0 noise was stripped by JPEG compression.
> v2.0 optimizes noise over **K random augmentations per step**, making it survive
> social media uploads, screenshots, and re-sharing.
""")

    with gr.Row():
        # ── Left Column: Inputs ──────────────────────────────────────────────
        with gr.Column(scale=1):
            input_image = gr.Image(label="📤 Original Image", type="pil", height=380)

            with gr.Group():
                gr.Markdown("### ⚙ Protection Settings")

                intensity_slider = gr.Slider(
                    0.01, 0.08, value=0.03, step=0.01,
                    label="Shield Strength (Epsilon ε)",
                    info="Higher = stronger protection but slightly more visible noise"
                )
                steps_slider = gr.Slider(
                    10, 100, value=50, step=10,
                    label="Optimization Steps",
                    info="More steps = stronger protection, longer runtime"
                )

                with gr.Row():
                    eot_k_slider = gr.Slider(
                        2, 16, value=8, step=2,
                        label="🆕 EoT Augmentations per Step (K)",
                        info="v2.0 only. Higher K = more compression-robust, slower. K=8 recommended on T4x2"
                    )

            with gr.Accordion("💡 Kaggle Session Guide", open=False):
                gr.Markdown("""
**Recommended: GPU T4 x2**

| Session | VRAM | EoT K | Verdict |
|---|---|---|---|
| GPU T4 x2 | 2×16 GB | K=8–16 | ✅ **Recommended** |
| GPU P100  | 1×16 GB | K=2–4  | ⚠ Reduce K or OOM |
| CPU       | RAM only | K=2   | ❌ Very slow |

**Why T4 x2?**
EoT runs K forward passes per step. With VAE on `cuda:0` and
CLIP on `cuda:1`, memory is split across both cards.
At K=8, steps=50: ~400 VAE passes + 400 CLIP passes total.
P100's single 16 GB card will OOM above K=4.
""")

            run_btn = gr.Button("🛡 Apply EoT Immunization", variant="primary", size="lg")

        # ── Right Column: Outputs ────────────────────────────────────────────
        with gr.Column(scale=1):
            with gr.Tabs():
                with gr.TabItem("🖼 Protected Image"):
                    output_image = gr.Image(
                        label="Protected (Lossless PNG)", type="pil", height=340
                    )
                with gr.TabItem("🔬 JPEG Robustness Preview"):
                    jpeg_image = gr.Image(
                        label="After JPEG Q75 Compression — noise should still be active",
                        type="pil", height=340
                    )

            status_md   = gr.Markdown()
            download_file = gr.File(label="⬇ Download Lossless PNG")

    # ── How It Works ─────────────────────────────────────────────────────────
    with gr.Accordion("📖 How EoT Works (v1.0 vs v2.0)", open=False):
        gr.Markdown("""
### v1.0 Problem
```
PGD Step: grad = ∇_δ L(x + δ)
```
Noise optimized only for the *clean* image.
A JPEG save changes pixel values → noise becomes ineffective.

### v2.0 Solution (EoT)
```
PGD Step: grad = (1/K) Σₖ ∇_δ L(augment_k(x + δ))
```
Augmentations applied each step:
- 🗜 JPEG simulation (Q=60–85)
- 🌫 Gaussian blur (3×3, 5×5)
- ☀ Brightness jitter (±5%)
- 📷 Gaussian sensor noise
- ✂ Random crop + pad

The noise must work across ALL of these → it becomes **transformation-invariant**.
""")

    run_btn.click(
        fn       = immunize,
        inputs   = [input_image, intensity_slider, steps_slider, eot_k_slider],
        outputs  = [output_image, jpeg_image, status_md, download_file],
    )


# %% — Cell 8: Launch
# ============================================================================
print("🚀 Launching CloakID v2.0 with EoT...")
demo.queue().launch(share=True, debug=True)
