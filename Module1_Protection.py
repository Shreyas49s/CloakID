# %% [markdown]
# # CloakID — Module 1 v5.0: Image Immunization with Multimodal Diffusion Attack
#
# **Upgrade over v4.1: SDXL (text-to-image) → InstructPix2Pix (multimodal)**
# **Model: timbrooks/instruct-pix2pix (fully public, no HF auth required)**
#
# Why InstructPix2Pix is the correct threat model for CloakID:
#   SDXL and SD v1.5 are text-to-image models — they take TEXT as input
#   and generate images from scratch. They are NOT the models attackers use
#   to edit existing photos of real people.
#
#   InstructPix2Pix (IP2P) is a TRUE MULTIMODAL model — it takes BOTH an
#   existing image AND a text instruction simultaneously and edits the image
#   according to the instruction. This is exactly the attack scenario your
#   project synopsis describes:
#     "Make her wear a hijab"  +  [victim's photo]  →  manipulated image
#
#   IP2P directly matches the threat models named in your project:
#     - Generative Identity Theft via text-guided image editing
#     - Latent Diffusion Models used for image manipulation (not generation)
#     - The attacker workflow: download photo → provide edit instruction → done
#
# Key architectural difference from SD v1.5 / SDXL:
#   IP2P U-Net takes 8 input channels instead of 4:
#     channels 0-3  : noisy latent z_t  (same as standard SD)
#     channels 4-7  : image conditioning latent (the INPUT IMAGE to be edited)
#   This concatenation is what makes it multimodal — the U-Net sees BOTH
#   the diffusion noise AND the original image latent at every denoising step.
#
# Key code differences from v4.1 (SDXL):
#   SD_MODEL_ID    : timbrooks/instruct-pix2pix
#   ATTACK_RES     : 1024 -> 512  (IP2P native resolution)
#   Gray target    : (1024,1024) -> (512,512)
#   Text encoder   : dual SDXL -> single CLIPTextModel (IP2P uses SD v1.5 base)
#   Null embed     : [1,77,2048] -> [1,77,768]
#   U-Net channels : 4 -> 8  (image conditioning latent concatenated)
#   added_cond_kwargs: SDXL-specific kwargs REMOVED (not needed for IP2P)
#   compute_diffusion_loss: new image_latent conditioning argument added
#
# Architecture:
#   Layer 1   — VAE Encoder Attack        (Gray Latent Target)
#   Layer 2A  — CLIP Vision Attack        (Semantic Void, legacy pipelines)
#   Layer 2B  — SigLIP Vision Attack      (Semantic Void, modern LDMs)
#   Layer 3   — Diffusion Attack          (Full LDM Pipeline Target) ← NEW
#   Defense   — EoT over K augments       (Compression Robustness)
#
#   Two Modes:
#     Fast Mode         → Layers 1 + 2A + 2B + EoT  (v3.0 behaviour)
#     Max Protection    → Layers 1 + 2A + 2B + 3 + EoT  ← NEW
#
# ─────────────────────────────────────────────────────────────────────
# KAGGLE SESSION RECOMMENDATION
# ─────────────────────────────────────────────────────────────────────
#  Use:  GPU T4 x2  (same as v3.0)
#
#  Memory breakdown (Max Protection mode, IP2P):
#   cuda:0 -> VAE  (~490 MB fp16)
#           + U-Net IP2P (~3.2 GB fp16)  <- same size as SD v1.5 base
#           + Text Encoder (~235 MB fp16)
#           Total cuda:0: ~3.9 GB  OK within 16 GB
#   cuda:1 -> CLIP   (~1.7 GB fp16)
#           + SigLIP (~800 MB fp16)
#           Total cuda:1: ~2.5 GB  OK within 16 GB
#
#  P100 (single 16 GB): Fast Mode only recommended.
#  T4 x2: Both modes fully supported with comfortable headroom.
#
#  Runtime estimate on T4 x2 (steps=80, K=8):
#    Fast Mode        : ~8-12 min  (same as v3.0/v4.0)
#    Max Protection   : ~20-35 min (IP2P similar cost to SD v1.5 per step)
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
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    DDIMScheduler,
)
from transformers import (
    CLIPModel, CLIPProcessor,
    AutoModel, AutoProcessor,
    CLIPTextModel, CLIPTokenizer,
    # CLIPTextModelWithProjection not needed — IP2P uses single text encoder
)
import gradio as gr
import warnings, random, io

warnings.filterwarnings("ignore")

# ── Device Setup ─────────────────────────────────────────────────────────────
DEVICE_VAE    = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE_UNET   = DEVICE_VAE    # U-Net shares cuda:0 with VAE (same latent space)
DEVICE_CLIP   = torch.device("cuda:1" if torch.cuda.device_count() > 1 else DEVICE_VAE)
DEVICE_SIGLIP = DEVICE_CLIP
DEVICE        = DEVICE_VAE
DTYPE         = torch.float16 if DEVICE.type == "cuda" else torch.float32

print(f"  VAE + U-Net device : {DEVICE_VAE}")
print(f"  CLIP + SigLIP      : {DEVICE_CLIP}")
print(f"  Dtype              : {DTYPE}")

SD_MODEL_ID = "timbrooks/instruct-pix2pix"  # fully public, no HF auth needed

# ── Layer 1: VAE ─────────────────────────────────────────────────────────────
print("⏳ Loading VAE (stabilityai/sd-vae-ft-mse) …")
vae = AutoencoderKL.from_pretrained(
    "stabilityai/sd-vae-ft-mse", torch_dtype=DTYPE,
).to(DEVICE_VAE)
vae.eval()
vae.requires_grad_(False)
print("  ✔ VAE loaded.")

# ── Layer 2A: CLIP ────────────────────────────────────────────────────────────
print("⏳ Loading CLIP (openai/clip-vit-large-patch14) …")
clip_model = CLIPModel.from_pretrained(
    "openai/clip-vit-large-patch14", torch_dtype=DTYPE,
).to(DEVICE_CLIP)
clip_model.eval()
clip_model.requires_grad_(False)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# Pre-compute CLIP target vector + normalization constants at load time.
# Previously clip_mean/clip_std were re-created as new tensors on every
# compute_clip_loss() call — that is 1920 unnecessary tensor allocations
# across a steps=80, K=8 run. Pre-computing once here eliminates all of them.
_clip_emb_dim = clip_model.config.projection_dim
_gen_clip = torch.Generator(
    device=DEVICE_CLIP if DEVICE_CLIP.type == "cuda" else "cpu"
)
_gen_clip.manual_seed(999)
_clip_target = torch.randn(
    1, _clip_emb_dim, generator=_gen_clip, device=DEVICE_CLIP
).to(DTYPE)
_clip_target = _clip_target / _clip_target.norm(dim=-1, keepdim=True)
_clip_mean = torch.tensor(
    [0.48145466, 0.4578275, 0.40821073], device=DEVICE_CLIP
).view(1, 3, 1, 1).to(DTYPE)
_clip_std = torch.tensor(
    [0.26862954, 0.26130258, 0.27577711], device=DEVICE_CLIP
).view(1, 3, 1, 1).to(DTYPE)
print(f"  ✔ CLIP loaded (emb_dim={_clip_emb_dim}).")

# ── Layer 2B: SigLIP ──────────────────────────────────────────────────────────
print("⏳ Loading SigLIP (google/siglip-so400m-patch14-384) …")
siglip_model = AutoModel.from_pretrained(
    "google/siglip-so400m-patch14-384", torch_dtype=DTYPE,
).to(DEVICE_SIGLIP)
siglip_model.eval()
siglip_model.requires_grad_(False)
siglip_processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")

_siglip_mean = torch.tensor(
    siglip_processor.image_processor.image_mean, device=DEVICE_SIGLIP
).view(1, 3, 1, 1).to(DTYPE)
_siglip_std = torch.tensor(
    siglip_processor.image_processor.image_std, device=DEVICE_SIGLIP
).view(1, 3, 1, 1).to(DTYPE)

_siglip_emb_dim = siglip_model.config.vision_config.hidden_size
_gen_siglip = torch.Generator(
    device=DEVICE_SIGLIP if DEVICE_SIGLIP.type == "cuda" else "cpu"
)
_gen_siglip.manual_seed(42)
_siglip_target = torch.randn(
    1, _siglip_emb_dim, generator=_gen_siglip, device=DEVICE_SIGLIP
).to(DTYPE)
_siglip_target = _siglip_target / _siglip_target.norm(dim=-1, keepdim=True)
print(f"  ✔ SigLIP loaded (emb_dim={_siglip_emb_dim}).")

# ── Layer 3: Diffusion Attack Components (NEW in v4.0) ───────────────────────
# U-Net, Scheduler, Text Encoder — all needed to run the partial diffusion
# forward pass that the Diffusion Attack backpropagates through.
print("⏳ Loading IP2P U-Net (timbrooks/instruct-pix2pix) …")
unet = UNet2DConditionModel.from_pretrained(
    SD_MODEL_ID, subfolder="unet", torch_dtype=DTYPE,
).to(DEVICE_UNET)
unet.eval()
unet.requires_grad_(False)
# IP2P U-Net has 8 input channels (4 noise + 4 image conditioning)
# Verify this is correct — will print 8 if loaded properly
print(f"  ✔ IP2P U-Net loaded (in_channels={unet.config.in_channels}).")

print("⏳ Loading DDIM Scheduler …")
scheduler = DDIMScheduler.from_pretrained(SD_MODEL_ID, subfolder="scheduler")
scheduler.set_timesteps(50)
print("  ✔ Scheduler loaded.")

# ── IP2P Single Text Encoder + Tokenizer ─────────────────────────────────────
# IP2P is built on SD v1.5 base — uses a single CLIPTextModel (768-dim).
# The text instruction ("make her wear X") is encoded here.
# For our null/unconditional attack we use empty string — this makes the
# Diffusion Attack model-agnostic to whatever instruction an attacker uses.
print("⏳ Loading IP2P Text Encoder (CLIP ViT-L) + Tokenizer …")
sd_tokenizer = CLIPTokenizer.from_pretrained(SD_MODEL_ID, subfolder="tokenizer")
sd_text_enc  = CLIPTextModel.from_pretrained(
    SD_MODEL_ID, subfolder="text_encoder", torch_dtype=DTYPE,
).to(DEVICE_UNET)
sd_text_enc.eval()
sd_text_enc.requires_grad_(False)
print("  ✔ Text encoder loaded.")

# ── Pre-compute null text embedding ──────────────────────────────────────────
# Shape: [1, 77, 768] — same as SD v1.5, single encoder
with torch.no_grad():
    _null_tokens = sd_tokenizer(
        [""], padding="max_length",
        max_length=sd_tokenizer.model_max_length,
        return_tensors="pt",
    ).input_ids.to(DEVICE_UNET)
    _null_emb = sd_text_enc(_null_tokens).last_hidden_state  # [1, 77, 768]
print(f"  ✔ Null text embedding pre-computed (shape={tuple(_null_emb.shape)}).")

# ── Pre-compute gray target for Diffusion Attack ──────────────────────────────
# The Diffusion Attack forces f(x+δ) → x_targ where x_targ is a gray image.
# We encode the gray target ONCE here (not inside the attack loop) and reuse
# it every step — encoding is expensive and the target never changes.
with torch.no_grad():
    _gray_pil    = Image.new("RGB", (512, 512), (128, 128, 128))  # IP2P native 512px
    _gray_arr    = np.array(_gray_pil).astype(np.float32) / 255.0
    _gray_tensor = (
        torch.from_numpy(_gray_arr).permute(2, 0, 1)
        .unsqueeze(0).to(DEVICE_UNET, dtype=DTYPE)
    )
    _gray_tensor_scaled = _gray_tensor * 2.0 - 1.0
    _gray_latent = vae.encode(
        _gray_tensor_scaled.to(DEVICE_VAE)
    ).latent_dist.mean.to(DEVICE_UNET)   # [1, 4, 64, 64] for 512px input
    _VAE_SCALE   = 0.18215
    _gray_latent = _gray_latent * _VAE_SCALE
print(f"  ✔ Gray target latent pre-computed (shape={tuple(_gray_latent.shape)}).")
print("✔ All models ready.\n")


# %% — Cell 3: Core Utility Functions
# ============================================================================

ATTACK_RES = 512   # IP2P native resolution (SDXL uses 1024)

def pil_to_tensor(pil_img: Image.Image, size: int = ATTACK_RES) -> torch.Tensor:
    img = pil_img.convert("RGB").resize((size, size), Image.LANCZOS)
    arr = np.array(img).astype(np.float32) / 255.0
    return (
        torch.from_numpy(arr).permute(2, 0, 1)
        .unsqueeze(0).to(DEVICE, dtype=torch.float32)
    )

def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    arr = tensor.squeeze(0).clamp(0, 1).detach().cpu().permute(1, 2, 0).numpy()
    return Image.fromarray((arr * 255).astype(np.uint8))

def apply_perturbation_fullres(
    original_pil: Image.Image, delta_lowres: torch.Tensor
) -> Image.Image:
    orig_w, orig_h = original_pil.size
    delta_fullres  = F.interpolate(
        delta_lowres, size=(orig_h, orig_w), mode="bilinear", align_corners=False
    )
    orig_tensor = (
        torch.from_numpy(
            np.array(original_pil.convert("RGB")).astype(np.float32) / 255.0
        ).permute(2, 0, 1).unsqueeze(0).to(DEVICE, dtype=torch.float32)
    )
    return tensor_to_pil((orig_tensor + delta_fullres).clamp(0, 1))

def compute_ssim(img_a: Image.Image, img_b: Image.Image) -> float:
    size = min(img_a.size[0], 1024), min(img_a.size[1], 1024)
    a = np.array(img_a.convert("RGB").resize(size, Image.LANCZOS))
    b = np.array(img_b.convert("RGB").resize(size, Image.LANCZOS))
    return compare_ssim(a, b, channel_axis=2, data_range=255)

def compute_psnr(img_a: Image.Image, img_b: Image.Image) -> float:
    a = np.array(img_a.convert("RGB")).astype(np.float32)
    b = np.array(img_b.convert("RGB")).astype(np.float32)
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * np.log10(255.0 / np.sqrt(mse))


# %% — Cell 4: EoT Augmentation Bank (unchanged from v3.0)
# ============================================================================

def jpeg_simulate(x: torch.Tensor, quality: int = 75) -> torch.Tensor:
    """Real PIL JPEG + Straight-Through Estimator."""
    with torch.no_grad():
        arr = x.squeeze(0).clamp(0, 1).cpu().permute(1, 2, 0).numpy()
        pil_img = Image.fromarray((arr * 255).astype(np.uint8))
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        jpeg_arr = np.array(Image.open(buf).copy()).astype(np.float32) / 255.0
        jpeg_tensor = (
            torch.from_numpy(jpeg_arr).permute(2, 0, 1)
            .unsqueeze(0).to(x.device, dtype=x.dtype)
        )
    return (jpeg_tensor - x).detach() + x   # STE

def gaussian_blur_augment(x: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    padding = kernel_size // 2
    w = torch.ones(3, 1, kernel_size, kernel_size, device=x.device, dtype=x.dtype)
    w = w / (kernel_size * kernel_size)
    return F.conv2d(x, w, padding=padding, groups=3).clamp(0, 1)

def brightness_jitter(x: torch.Tensor, strength: float = 0.05) -> torch.Tensor:
    return (x * (1.0 + random.uniform(-strength, strength))).clamp(0, 1)

def additive_noise(x: torch.Tensor, std: float = 0.01) -> torch.Tensor:
    return (x + torch.randn_like(x) * std).clamp(0, 1)

def random_crop_pad(x: torch.Tensor, crop_frac: float = 0.05) -> torch.Tensor:
    _, _, H, W = x.shape
    ch, cw = int(H * crop_frac), int(W * crop_frac)
    if ch == 0 or cw == 0:
        return x
    return F.interpolate(
        x[:, :, ch:H-ch, cw:W-cw], size=(H, W),
        mode="bilinear", align_corners=False
    )

NON_JPEG_AUGMENTATIONS = [
    lambda x: gaussian_blur_augment(x, kernel_size=random.choice([3, 5])),
    lambda x: brightness_jitter(x, strength=random.uniform(0.02, 0.06)),
    lambda x: additive_noise(x, std=random.uniform(0.005, 0.02)),
    lambda x: random_crop_pad(x, crop_frac=random.uniform(0.02, 0.05)),
    lambda x: x,
]

def apply_eot_augmentation(x: torch.Tensor, k_index: int, eot_k: int) -> torch.Tensor:
    # Cap JPEG passes at 2 regardless of K — each PIL JPEG round-trip
    # is a CPU operation (~8 ms). At K=8 the old code ran 4 JPEG passes
    # per step = 320 PIL ops total. Capping at 2 halves this to 160 ops
    # while still providing sufficient JPEG hardening in the gradient.
    n_jpeg = min(2, max(1, eot_k // 2))
    if k_index < n_jpeg:
        return jpeg_simulate(x, quality=random.randint(60, 85))
    return random.choice(NON_JPEG_AUGMENTATIONS)(x)


# %% — Cell 5: Loss Functions
# ============================================================================

def compute_vae_loss(vae_model, perturbed: torch.Tensor) -> torch.Tensor:
    """Layer 1 — Force VAE latent → zero (gray target)."""
    perturbed_scaled = perturbed * 2.0 - 1.0
    pert_latent = vae_model.encode(
        perturbed_scaled.to(DEVICE_VAE, dtype=DTYPE)
    ).latent_dist.mean
    return F.mse_loss(pert_latent, torch.zeros_like(pert_latent))


def compute_clip_loss(model, perturbed: torch.Tensor) -> torch.Tensor:
    """Layer 2A — Push CLIP embedding → pre-computed random target.
    Uses pre-computed _clip_mean, _clip_std, _clip_target from Cell 2.
    Eliminates 3 tensor allocations + 1 Generator creation per call.
    """
    img_clip = F.interpolate(
        perturbed.to(DEVICE_CLIP), size=(224, 224),
        mode="bilinear", align_corners=False
    )
    img_clip = (img_clip.to(DTYPE) - _clip_mean) / _clip_std
    vision_out = model.vision_model(pixel_values=img_clip)
    emb = model.visual_projection(vision_out.pooler_output)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return F.mse_loss(emb, _clip_target)


def compute_siglip_loss(model, perturbed: torch.Tensor) -> torch.Tensor:
    """Layer 2B — Push SigLIP embedding → random target (modern LDMs)."""
    img_siglip = F.interpolate(
        perturbed.to(DEVICE_SIGLIP), size=(384, 384),
        mode="bilinear", align_corners=False
    )
    img_siglip = (img_siglip.to(DTYPE) - _siglip_mean) / _siglip_std
    vision_out = model.vision_model(pixel_values=img_siglip)
    emb = vision_out.pooler_output
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return F.mse_loss(emb, _siglip_target)


def compute_diffusion_loss(
    unet_model,
    vae_model,
    sched,
    perturbed:    torch.Tensor,
    t_partial:    int = 4,
) -> torch.Tensor:
    """
    Layer 3 — Multimodal Diffusion Attack using InstructPix2Pix (v5.0).

    Based on: MIT PhotoGuard (Salman et al., 2023), Equation 5:
        delta_diffusion = argmin || f(x + delta) - x_targ ||^2
    where f is the full multimodal LDM pipeline (IP2P) and x_targ is gray.

    Why IP2P is the correct attack target (vs SDXL in v4.1):
    ─────────────────────────────────────────────────────────
    IP2P is a TRUE MULTIMODAL model — it takes BOTH the input image AND
    a text instruction simultaneously. This directly matches the real
    attacker workflow: provide a victim photo + edit instruction.

    The key architectural difference from standard SD:
    ──────────────────────────────────────────────────
    IP2P U-Net has 8 input channels instead of 4:
        Standard SD:  [z_t: 4ch]                         → U-Net
        IP2P:         [z_t: 4ch | img_latent: 4ch]       → U-Net
                       ↑ noisy    ↑ original image latent
    The image latent (channels 4-7) conditions the generation on the
    input image content — this is what makes it multimodal.

    Attack strategy:
    ────────────────
    We encode the PERTURBED image as BOTH the noisy latent AND the image
    conditioning latent. The U-Net now sees the adversarial image in both
    channels. The loss pushes the final generated output → gray, meaning
    IP2P will fail to produce any coherent edit of the protected image.

    Steps:
    1. Encode perturbed image → clean latent z0
    2. Add noise at mid-timestep → z_t  (noisy channels)
    3. z0 is ALSO used as image conditioning latent (image channels)
    4. Concatenate [z_t (4ch) | z0 (4ch)] → 8-channel U-Net input
    5. Run T_partial denoising steps — backprop flows through all of them
    6. Decode → predicted edited image x_pred
    7. Loss = MSE(x_pred, gray_target)
    """
    # Step 1: Encode perturbed image → clean latent z0
    perturbed_scaled = perturbed * 2.0 - 1.0
    z0 = vae_model.encode(
        perturbed_scaled.to(DEVICE_VAE, dtype=DTYPE)
    ).latent_dist.mean.to(DEVICE_UNET)
    z0 = z0 * _VAE_SCALE

    # Step 2: Add noise at mid-range timestep
    t_idx  = len(sched.timesteps) // 2
    t_step = sched.timesteps[t_idx].to(DEVICE_UNET)
    noise  = torch.randn_like(z0)
    z_t    = sched.add_noise(z0, noise, t_step.unsqueeze(0))

    # Step 3 + 4: Build 8-channel IP2P input
    # IP2P concatenates [noisy_latent | image_conditioning_latent] along channel dim
    # Both use the same perturbed image latent z0 —
    # the U-Net sees the adversarial image as the "source image to edit"
    z_pred = z_t
    timesteps_partial = sched.timesteps[
        len(sched.timesteps) - t_partial :
    ].to(DEVICE_UNET)

    null_emb = _null_emb.to(DEVICE_UNET, dtype=DTYPE)

    # Step 5: T_partial denoising steps with 8-channel multimodal input
    for t in timesteps_partial:
        # Concatenate noisy latent with image conditioning latent → 8 channels
        # This is the core IP2P mechanism — channels 4-7 are the image condition
        unet_input = torch.cat([z_pred, z0], dim=1)  # [1, 8, H/8, W/8]

        noise_pred = unet_model(
            unet_input,                        # 8-channel multimodal input
            t,
            encoder_hidden_states=null_emb,    # null text = model-agnostic attack
        ).sample

        z_pred = sched.step(noise_pred, t, z_pred).prev_sample

    # Step 6: Decode final latent → image space
    z_pred_decoded = z_pred / _VAE_SCALE
    x_pred = vae_model.decode(z_pred_decoded.to(DEVICE_VAE)).sample
    x_pred = (x_pred.clamp(-1, 1) + 1.0) / 2.0   # [-1,1] → [0,1]

    # Step 7: MSE against pre-computed gray target
    if x_pred.shape[-2:] != _gray_tensor.shape[-2:]:
        x_pred = F.interpolate(
            x_pred, size=_gray_tensor.shape[-2:],
            mode="bilinear", align_corners=False
        )

    return F.mse_loss(x_pred, _gray_tensor.to(DEVICE_VAE))


# %% — Cell 6: EoT-PGD Attack Engine (v5.0 Optimized)
# ============================================================================

@torch.enable_grad()
def pgd_attack_eot(
    original_pil:      Image.Image,
    steps:             int   = 50,    # Optimized default: 80 → 50 (save ~37%)
    epsilon:           float = 0.02,
    vae_weight:        float = 1.0,
    clip_weight:       float = 0.3,
    siglip_weight:     float = 0.5,
    diffusion_weight:  float = 0.8,
    t_partial:         int   = 4,
    use_diffusion:     bool  = False,
    eot_k:             int   = 6,     # Optimized default: 8 → 6 (save ~25%)
    progress_callback  = None,
):
    """
    CloakID v5.0 Optimized — Four-loss PGD with EoT.

    Optimizations applied vs original v5.0:
    ────────────────────────────────────────
    1. CLIP norm constants pre-computed at load time (no per-call allocation)
    2. JPEG passes capped at 2 regardless of K (halves CPU PIL overhead)
    3. Stochastic diffusion: U-Net runs on k==0 ONLY (1 pass per step)
       Scale factor = eot_k (not eot_k/2) to normalize gradient correctly
    4. Default steps reduced 80→50, K reduced 8→6 for ~45% faster runs
    5. Demo Mode preset: steps=25, K=4 for ~2-3 min Fast Mode runs

    Runtime estimates on T4 x2 (optimized defaults):
      Fast Mode     (steps=50, K=6)        : ~5-7 min
      Max Protection(steps=50, K=6, T=4)   : ~15-20 min
      Demo Mode     (steps=25, K=4, Fast)  : ~2-3 min
    """
    x_orig = pil_to_tensor(original_pil, size=ATTACK_RES)
    x_orig.requires_grad_(False)

    delta_init = (torch.rand_like(x_orig) * 2 - 1) * (epsilon * 0.1)
    delta = delta_init.clone().detach().requires_grad_(True)
    alpha = epsilon / (steps * 0.8)

    mode_str = "Max Protection (VAE+CLIP+SigLIP+Diffusion)" if use_diffusion \
               else "Fast (VAE+CLIP+SigLIP)"
    print(f"\n🛡 CloakID v5.0 | Mode: {mode_str}")
    print(f"   ε={epsilon} | Steps={steps} | K={eot_k} | T_partial={t_partial}")
    print(f"   Weights → VAE:{vae_weight} CLIP:{clip_weight} "
          f"SigLIP:{siglip_weight} Diff:{diffusion_weight if use_diffusion else 'off'}")

    for step in range(steps):

        accumulated_grad = torch.zeros_like(delta.data)
        total_loss_avg   = 0.0

        for k in range(eot_k):
            x_adv = (x_orig + delta).clamp(0, 1)
            x_aug = apply_eot_augmentation(x_adv, k, eot_k)

            # ── Encoder losses (all K passes — cheap, maintains EoT quality)
            l_vae    = compute_vae_loss(vae, x_aug)
            l_clip   = compute_clip_loss(clip_model, x_aug)
            l_siglip = compute_siglip_loss(siglip_model, x_aug)

            loss = (
                vae_weight    * l_vae
              + clip_weight   * l_clip.to(l_vae.device)
              + siglip_weight * l_siglip.to(l_vae.device)
            )

            # ── Diffusion loss — STOCHASTIC: only on k==0 pass ───────────
            # Running U-Net on 1 of K passes instead of K/K cuts U-Net
            # calls from eot_k per step to 1 per step — the single largest
            # speedup. We scale the loss by eot_k to keep gradient magnitude
            # consistent with a full K-pass average.
            # Bug fix vs provided code: scale = eot_k (not eot_k/2) to
            # correctly normalize against the K accumulated gradients.
            if use_diffusion and k == 0:
                l_diff = compute_diffusion_loss(
                    unet, vae, scheduler, x_aug, t_partial=t_partial
                )
                loss = loss + diffusion_weight * l_diff.to(l_vae.device) * eot_k

            loss.backward()
            accumulated_grad += delta.grad.detach().clone()
            total_loss_avg   += loss.item()
            delta.grad.zero_()

        mean_grad      = accumulated_grad / eot_k
        total_loss_avg = total_loss_avg   / eot_k

        with torch.no_grad():
            delta.data = delta.data - alpha * torch.sign(mean_grad)
            delta.data = delta.data.clamp(-epsilon, epsilon)
            delta.data = (x_orig + delta.data).clamp(0, 1) - x_orig

        if progress_callback and step % 5 == 0:
            diff_str = " | Diff active" if use_diffusion else ""
            progress_callback(
                (step + 1) / steps,
                desc=f"v5.0 Step {step+1}/{steps}{diff_str} | Loss: {total_loss_avg:.4f}"
            )

    # ── Output generation ─────────────────────────────────────────────────
    protected_pil = apply_perturbation_fullres(original_pil, delta.detach())

    # ── JPEG robustness verification ──────────────────────────────────────
    buf = io.BytesIO()
    protected_pil.save(buf, format="JPEG", quality=75)
    buf.seek(0)
    jpeg_pil = Image.open(buf).copy()

    ssim_original  = compute_ssim(original_pil, protected_pil)
    psnr_original  = compute_psnr(original_pil, protected_pil)
    ssim_after_jpg = compute_ssim(original_pil, jpeg_pil)

    orig_arr  = np.array(original_pil.convert("RGB").resize(
        protected_pil.size, Image.LANCZOS)).astype(np.float32)
    prot_arr  = np.array(protected_pil.convert("RGB")).astype(np.float32)
    jpeg_arr  = np.array(jpeg_pil.convert("RGB")).astype(np.float32)
    noise_survival = min(
        1.0,
        np.linalg.norm(jpeg_arr - orig_arr) /
        max(np.linalg.norm(prot_arr - orig_arr), 1e-6)
    )

    print(f"✔ Attack complete.")
    print(f"  SSIM (vs original)   : {ssim_original:.4f}")
    print(f"  PSNR (vs original)   : {psnr_original:.2f} dB")
    print(f"  SSIM after JPEG Q75  : {ssim_after_jpg:.4f}")
    print(f"  Noise survival rate  : {noise_survival:.2%}")

    return protected_pil, jpeg_pil, {
        "ssim":            ssim_original,
        "psnr":            psnr_original,
        "ssim_jpeg":       ssim_after_jpg,
        "noise_survival":  noise_survival,
        "epsilon":         epsilon,
        "steps":           steps,
        "eot_k":           eot_k,
        "clip_weight":     clip_weight,
        "siglip_weight":   siglip_weight,
        "diffusion_weight": diffusion_weight if use_diffusion else 0.0,
        "t_partial":       t_partial if use_diffusion else 0,
        "mode":            "Max Protection" if use_diffusion else "Fast",
    }


# %% — Cell 7: Gradio Interface (v4.0 — v3.0 layout extended)
# ============================================================================

OUTPUT_DIR = "/kaggle/working"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def immunize(
    image:             Image.Image,
    attack_mode:       str,
    intensity:         float,
    steps:             int,
    eot_k:             int,
    clip_weight:       float,
    siglip_weight:     float,
    diffusion_weight:  float,
    t_partial:         int,
    progress = gr.Progress(track_tqdm=True),
):
    if image is None:
        raise gr.Error("Please upload an image first.")

    use_diffusion = (attack_mode == "🔴 Max Protection  (VAE + CLIP + SigLIP + Diffusion)")

    protected_pil, jpeg_preview, metrics = pgd_attack_eot(
        original_pil      = image,
        steps             = int(steps),
        epsilon           = intensity,
        vae_weight        = 1.0,
        clip_weight       = clip_weight,
        siglip_weight     = siglip_weight,
        diffusion_weight  = diffusion_weight,
        t_partial         = int(t_partial),
        use_diffusion     = use_diffusion,
        eot_k             = int(eot_k),
        progress_callback = progress,
    )

    save_path = os.path.join(OUTPUT_DIR, "cloakid_v5_protected.png")
    protected_pil.save(save_path, format="PNG", compress_level=1)

    ssim_badge = (
        "🟢 Excellent"      if metrics["ssim"] >= 0.95 else
        "🟡 Good"           if metrics["ssim"] >= 0.90 else
        "🔴 Visible Noise"
    )
    psnr_badge = (
        "🟢 Excellent"  if metrics["psnr"] >= 38 else
        "🟡 Good"       if metrics["psnr"] >= 32 else
        "🔴 Degraded"
    )
    surv_badge = (
        "🟢 Robust"   if metrics["noise_survival"] >= 0.7 else
        "🟡 Moderate" if metrics["noise_survival"] >= 0.4 else
        "🔴 Fragile"
    )
    ssim_target = (
        "✅ Target Met" if metrics["ssim"] >= 0.90 else "❌ Below 0.90 Target"
    )

    diff_row = (
        f"| **Diffusion Weight** | `{metrics['diffusion_weight']}` "
        f"· T_partial={metrics['t_partial']} | 🔴 Active |\n"
        if use_diffusion else
        "| **Diffusion Attack** | `off` | ⚪ Fast Mode |\n"
    )

    status = f"""### ✅ CloakID v5.0 — Immunization Report  ({metrics['mode']})

| Metric | Value | Status |
|---|---|---|
| **Visual Fidelity (SSIM)** | `{metrics['ssim']:.4f}` | {ssim_badge} · {ssim_target} |
| **Image Quality (PSNR)** | `{metrics['psnr']:.2f} dB` | {psnr_badge} |
| **SSIM after JPEG Q75** | `{metrics['ssim_jpeg']:.4f}` | — |
| **Noise Survival Rate** | `{metrics['noise_survival']:.1%}` | {surv_badge} |
{diff_row}
> **EoT Config:** K={metrics['eot_k']} aug/step · ε={metrics['epsilon']} · {metrics['steps']} steps
> **Ensemble Weights:** VAE=1.0 · CLIP={metrics['clip_weight']} · SigLIP={metrics['siglip_weight']}"""

    return protected_pil, jpeg_preview, status, save_path


# ── UI Layout (v3.0 extended) ─────────────────────────────────────────────────

css = """
.metric-box { border-radius:8px; padding:12px; background:#1a1a2e; }
.new-badge  { color:#00ff88; font-weight:bold; }
footer      { display:none !important; }
"""

with gr.Blocks(title="CloakID v5.0 IP2P", theme=gr.themes.Soft(), css=css) as demo:

    gr.Markdown("""
# 🛡 CloakID v5.0 — Image Immunization (Multimodal)
### Quad-Loss Adversarial Defense · VAE + CLIP + SigLIP + **Multimodal Diffusion Attack (IP2P)**

> **Upgraded in v5.0:** Diffusion Attack now targets **InstructPix2Pix** — a true multimodal
> model that takes image + text instruction simultaneously. This directly matches the real
> attacker workflow: *"make her wear X"* + victim photo → manipulated output.
> Fully public, no HF authentication required.
""")

    with gr.Row():

        # ── Left Column: Inputs ───────────────────────────────────────────────
        with gr.Column(scale=1):
            input_image = gr.Image(
                label="📤 Original Image", type="pil", height=320
            )

            # ── NEW in v4.0: Attack Mode Selector ────────────────────────────
            with gr.Group():
                gr.Markdown("### 🎯 Attack Mode")
                attack_mode = gr.Radio(
                    choices=[
                        "⚡ Fast Mode  (VAE + CLIP + SigLIP)",
                        "🔴 Max Protection  (VAE + CLIP + SigLIP + Diffusion)",
                    ],
                    value="⚡ Fast Mode  (VAE + CLIP + SigLIP)",
                    label="Select Protection Level",
                    info=(
                        "Fast: ~8–12 min · same as v3.0  |  "
                        "Max: ~25–40 min · disrupts full LDM pipeline"
                    ),
                )

            with gr.Group():
                gr.Markdown("### ⚙ Core Settings")
                intensity_slider = gr.Slider(
                    0.01, 0.08, value=0.02, step=0.01,
                    label="Shield Strength (Epsilon ε)",
                    info="0.02 recommended — SSIM ≥ 0.90 verified"
                )
                steps_slider = gr.Slider(
                    10, 120, value=50, step=10,   # optimized default: 80→50
                    label="Optimization Steps",
                    info="50 recommended (balanced). 25 for Demo Mode (~2-3 min)."
                )
                eot_k_slider = gr.Slider(
                    2, 16, value=6, step=2,       # optimized default: 8→6
                    label="EoT Augmentations per Step (K)",
                    info="6 recommended. 4 for Demo Mode."
                )

            with gr.Group():
                gr.Markdown("### ⚡ Quick Presets")
                with gr.Row():
                    demo_btn = gr.Button("⚡ Demo Mode (~2-3 min)", size="sm")
                    balanced_btn = gr.Button("⚖ Balanced (~8-10 min)", size="sm")
                    full_btn = gr.Button("🔴 Full Quality (~20 min)", size="sm")
                gr.Markdown(
                    "_Demo: Fast Mode, steps=25, K=4 · "
                    "Balanced: Fast Mode, steps=50, K=6 · "
                    "Full: Max Protection, steps=50, K=6, T=4_"
                )

            with gr.Group():
                gr.Markdown("### ⚖ Vision Encoder Weights (Layers 2A / 2B)")
                clip_weight_slider = gr.Slider(
                    0.0, 1.0, value=0.3, step=0.1,
                    label="CLIP Weight  (Layer 2A — legacy SD pipelines)",
                )
                siglip_weight_slider = gr.Slider(
                    0.0, 1.0, value=0.5, step=0.1,
                    label="SigLIP Weight  (Layer 2B — Flux / Gemini)",
                )

            # ── NEW in v4.0: Diffusion Attack Controls ────────────────────────
            with gr.Group():
                gr.Markdown("### 🆕 Diffusion Attack Settings (Layer 3 — Max Protection only)")
                diffusion_weight_slider = gr.Slider(
                    0.1, 2.0, value=0.8, step=0.1,
                    label="Diffusion Attack Weight",
                    info=(
                        "Higher = stronger LDM disruption but risks SSIM drop. "
                        "Start at 0.8, reduce if SSIM falls below 0.90."
                    ),
                )
                t_partial_slider = gr.Slider(
                    2, 8, value=4, step=1,
                    label="Partial Denoising Steps (T_partial)",
                    info=(
                        "U-Net steps to backprop through. "
                        "T=4 recommended — T=2 is faster, T=8 is stronger but slower."
                    ),
                )
                gr.Markdown(
                    "> ⚠ **Max Protection runtime:** ~25–40 min on T4 x2. "
                    "Reduce T_partial to 2 for a faster first test run."
                )

            with gr.Accordion("💡 Kaggle Session & Runtime Guide", open=False):
                gr.Markdown("""
**Recommended: GPU T4 x2**

| Session | Fast Mode | Max Protection | Verdict |
|---|---|---|---|
| GPU T4 x2 | ✅ ~8–12 min | ✅ ~20–35 min | ✅ **Recommended** |
| GPU P100  | ✅ ~10–15 min | ❌ OOM risk | ⚠ Fast Mode safer |
| CPU       | ❌ Very slow | ❌ Infeasible | ❌ |

**v5.0 VRAM on T4 x2 (IP2P):**
```
cuda:0 -> VAE (~490 MB) + U-Net IP2P (~3.2 GB) + TextEnc (~235 MB) = ~3.9 GB
cuda:1 -> CLIP (~1.7 GB) + SigLIP (~800 MB)                        = ~2.5 GB
```
Both cards comfortably within 16 GB. IP2P is similar in size to SD v1.5.
""")

            run_btn = gr.Button(
                "🛡 Apply CloakID v5.0 Immunization",
                variant="primary", size="lg"
            )

        # ── Right Column: Outputs ─────────────────────────────────────────────
        with gr.Column(scale=1):
            with gr.Tabs():
                with gr.TabItem("🖼 Protected Image"):
                    output_image = gr.Image(
                        label="Protected PNG (Lossless)", type="pil", height=300
                    )
                with gr.TabItem("🔬 JPEG Robustness Preview"):
                    jpeg_image = gr.Image(
                        label="After JPEG Q75 — noise should survive",
                        type="pil", height=300
                    )
            status_md     = gr.Markdown()
            download_file = gr.File(label="⬇ Download Lossless PNG")

    # ── Architecture Accordion ────────────────────────────────────────────────
    with gr.Accordion("📖 v5.0 Architecture — Optimizations & IP2P", open=False):
        gr.Markdown("""
### Four-Loss Ensemble (Max Protection)
```
Total Loss = 1.0 × L_vae(aug(x+δ))          ← Layer 1: encoder attack
           + 0.3 × L_clip(aug(x+δ))         ← Layer 2A: CLIP semantic
           + 0.5 × L_siglip(aug(x+δ))       ← Layer 2B: SigLIP semantic
           + 0.8 × L_diffusion(aug(x+δ))    ← Layer 3: full LDM pipeline (NEW)
```

### Diffusion Attack — How It Works
```
Perturbed Image x+δ
    ↓  VAE Encoder
Clean Latent z0  (scaled by VAE factor 0.18215)
    ↓  Add noise at mid-timestep t
Noisy Latent z_t
    ↓  U-Net × T_partial steps (backprop flows through here)
    |  null text conditioning — model-agnostic attack
Predicted Latent z_pred
    ↓  VAE Decoder
Predicted Image x_pred
    ↓
Loss = MSE(x_pred, gray_target)   ← Minimize this
```
The optimizer pushes delta so that **InstructPix2Pix** produces a
**gray image** regardless of what edit instruction the attacker provides.

### IP2P vs SDXL vs SD v1.5 — Why IP2P is the Correct Threat Model
| Property | SD v1.5 | SDXL | IP2P (v5.0) |
|---|---|---|---|
| Input modality | Text only | Text only | **Image + Text** |
| Task | Generation | Generation | **Image editing** |
| U-Net channels | 4 | 4 | **8 (multimodal)** |
| Matches attacker workflow | ❌ | ❌ | **✅** |
| Native resolution | 512px | 1024px | 512px |
| Public, no auth | ✅ | ✅ | ✅ |

### IP2P 8-Channel Multimodal U-Net Input
```
Standard SD U-Net:   [z_t: 4ch]              → denoiser → output
IP2P U-Net:          [z_t: 4ch | z0: 4ch]   → denoiser → edited image
                      ^ noisy    ^ image conditioning latent
```
CloakID injects adversarial delta into BOTH the noisy latent AND the
image conditioning channels — disrupting the edit at the deepest level.

### Version History
| Version | Key Addition | All Metrics Green |
|---|---|---|
| v1.0 | VAE + CLIP | ❌ |
| v2.0 | + EoT | ❌ SSIM < 0.90 |
| v3.0 | + SigLIP | ✅ SSIM 0.9575, Survival 100% |
| v4.0 | + Diffusion Attack (SD v1.5) | ✅ SSIM 0.9588 |
| v4.1 | Diffusion → SDXL | ✅ (text-to-image) |
| **v5.0** | **Diffusion → IP2P (multimodal)** | **TBD after testing** |
""")

    # ── Preset button handlers ────────────────────────────────────────────────
    demo_btn.click(
        fn=lambda: (
            "⚡ Fast Mode  (VAE + CLIP + SigLIP)", 0.02, 25, 4, 0.3, 0.5, 0.8, 2
        ),
        outputs=[
            attack_mode, intensity_slider, steps_slider, eot_k_slider,
            clip_weight_slider, siglip_weight_slider,
            diffusion_weight_slider, t_partial_slider,
        ]
    )
    balanced_btn.click(
        fn=lambda: (
            "⚡ Fast Mode  (VAE + CLIP + SigLIP)", 0.02, 50, 6, 0.3, 0.5, 0.8, 4
        ),
        outputs=[
            attack_mode, intensity_slider, steps_slider, eot_k_slider,
            clip_weight_slider, siglip_weight_slider,
            diffusion_weight_slider, t_partial_slider,
        ]
    )
    full_btn.click(
        fn=lambda: (
            "🔴 Max Protection  (VAE + CLIP + SigLIP + Diffusion)", 0.02, 50, 6, 0.3, 0.5, 0.8, 4
        ),
        outputs=[
            attack_mode, intensity_slider, steps_slider, eot_k_slider,
            clip_weight_slider, siglip_weight_slider,
            diffusion_weight_slider, t_partial_slider,
        ]
    )

    run_btn.click(
        fn      = immunize,
        inputs  = [
            input_image,
            attack_mode,
            intensity_slider,
            steps_slider,
            eot_k_slider,
            clip_weight_slider,
            siglip_weight_slider,
            diffusion_weight_slider,
            t_partial_slider,
        ],
        outputs = [output_image, jpeg_image, status_md, download_file],
    )


# %% — Cell 8: Launch
# ============================================================================
print("🚀 Launching CloakID v5.0 Optimized (VAE+CLIP+SigLIP+IP2P / Demo~2min, Full~20min)...")
demo.queue().launch(share=True, debug=True)
