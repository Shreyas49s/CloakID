# =============================================================================
# CloakID Module 1 — Streamlit Version (Attractive UI)
# =============================================================================

# ---------------------------
# CELL 1 — Install Dependencies
# ---------------------------
import subprocess, sys, os

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

print("Installing dependencies...")
pkgs = ["diffusers", "transformers", "accelerate", "streamlit", "lpips",
        "scikit-image", "pyngrok", "torch", "numpy", "pillow"]
for p in pkgs:
    install(p)
print("All dependencies installed.")


# ---------------------------
# CELL 2 — Write Streamlit App
# ---------------------------

app_code = r'''
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as compare_ssim
from diffusers import AutoencoderKL
from transformers import CLIPModel, CLIPProcessor
import streamlit as st
import warnings, os, io

warnings.filterwarnings("ignore")

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CloakID — Image Immunization",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Injected CSS (Cyberpunk Dark Theme) ──────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600&display=swap');

:root {
    --neon-cyan: #00f5ff;
    --neon-green: #39ff14;
    --dark-bg: #020b18;
    --card-bg: #041225;
    --border: #0a3a5c;
    --text-main: #cce8f4;
    --text-dim: #5b8fa8;
    --accent: #ff6b35;
}

/* Global */
html, body, [class*="css"] {
    font-family: 'Rajdhani', sans-serif;
    background-color: var(--dark-bg) !important;
    color: var(--text-main) !important;
}

.stApp {
    background: var(--dark-bg) !important;
    background-image:
        radial-gradient(ellipse at 10% 20%, rgba(0,245,255,0.04) 0%, transparent 50%),
        radial-gradient(ellipse at 90% 80%, rgba(57,255,20,0.03) 0%, transparent 50%),
        repeating-linear-gradient(
            0deg,
            transparent,
            transparent 80px,
            rgba(0,245,255,0.015) 80px,
            rgba(0,245,255,0.015) 81px
        ),
        repeating-linear-gradient(
            90deg,
            transparent,
            transparent 80px,
            rgba(0,245,255,0.015) 80px,
            rgba(0,245,255,0.015) 81px
        ) !important;
}

/* Hero Header */
.hero-block {
    text-align: center;
    padding: 3.5rem 1rem 2rem;
    position: relative;
}
.hero-tag {
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.75rem;
    letter-spacing: 0.4em;
    color: var(--neon-cyan);
    text-transform: uppercase;
    margin-bottom: 0.8rem;
    opacity: 0.8;
}
.hero-title {
    font-family: 'Orbitron', monospace;
    font-size: clamp(2.8rem, 6vw, 5rem);
    font-weight: 900;
    background: linear-gradient(135deg, #00f5ff 0%, #39ff14 50%, #00f5ff 100%);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: shimmer 4s linear infinite;
    line-height: 1.1;
    margin: 0;
    text-shadow: none;
}
@keyframes shimmer {
    0% { background-position: 0% center; }
    100% { background-position: 200% center; }
}
.hero-sub {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.1rem;
    color: var(--text-dim);
    margin-top: 1rem;
    letter-spacing: 0.05em;
    font-weight: 300;
}
.hero-divider {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 1rem;
    margin: 2rem auto;
    max-width: 600px;
}
.hero-divider-line {
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--neon-cyan), transparent);
}
.hero-divider-dot {
    width: 6px; height: 6px;
    background: var(--neon-cyan);
    border-radius: 50%;
    box-shadow: 0 0 10px var(--neon-cyan);
    animation: pulse 2s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(0.6); }
}

/* Stat Badges */
.badge-row {
    display: flex;
    justify-content: center;
    gap: 1.5rem;
    flex-wrap: wrap;
    margin-bottom: 2.5rem;
}
.badge {
    background: rgba(0,245,255,0.06);
    border: 1px solid rgba(0,245,255,0.2);
    border-radius: 4px;
    padding: 0.4rem 1rem;
    font-family: 'Orbitron', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    color: var(--neon-cyan);
    text-transform: uppercase;
}

/* Cards */
.cyber-card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.8rem;
    margin-bottom: 1.2rem;
    position: relative;
    overflow: hidden;
}
.cyber-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: linear-gradient(180deg, var(--neon-cyan), transparent);
}
.cyber-card-title {
    font-family: 'Orbitron', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.25em;
    color: var(--neon-cyan);
    text-transform: uppercase;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.cyber-card-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* Section Headers */
.section-label {
    font-family: 'Orbitron', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.3em;
    color: var(--neon-cyan);
    text-transform: uppercase;
    margin-bottom: 0.5rem;
    opacity: 0.8;
}

/* Metric Cards */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.8rem;
    margin-top: 1.2rem;
}
.metric-box {
    background: rgba(0,245,255,0.04);
    border: 1px solid rgba(0,245,255,0.15);
    border-radius: 6px;
    padding: 1rem;
    text-align: center;
}
.metric-value {
    font-family: 'Orbitron', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--neon-green);
    line-height: 1;
}
.metric-label {
    font-size: 0.7rem;
    color: var(--text-dim);
    letter-spacing: 0.1em;
    margin-top: 0.3rem;
    text-transform: uppercase;
}

/* Success Alert */
.success-alert {
    background: rgba(57,255,20,0.06);
    border: 1px solid rgba(57,255,20,0.3);
    border-radius: 6px;
    padding: 1.2rem 1.5rem;
    margin: 1rem 0;
    font-family: 'Rajdhani', sans-serif;
    font-size: 1rem;
    color: var(--neon-green);
    display: flex;
    align-items: center;
    gap: 0.8rem;
}
.warn-alert {
    background: rgba(255,107,53,0.06);
    border: 1px solid rgba(255,107,53,0.3);
    border-radius: 6px;
    padding: 1.2rem 1.5rem;
    margin: 1rem 0;
    font-family: 'Rajdhani', sans-serif;
    font-size: 1rem;
    color: var(--accent);
}

/* Streamlit Widget Overrides */
div[data-testid="stFileUploader"] {
    background: rgba(0,245,255,0.03) !important;
    border: 1px dashed rgba(0,245,255,0.25) !important;
    border-radius: 8px !important;
    padding: 1rem !important;
    transition: border-color 0.3s;
}
div[data-testid="stFileUploader"]:hover {
    border-color: rgba(0,245,255,0.5) !important;
}

.stSlider > div > div > div > div {
    background: var(--neon-cyan) !important;
}
.stSlider > div > div > div {
    background: rgba(0,245,255,0.2) !important;
}

.stTextInput input, .stNumberInput input {
    background: rgba(4,18,37,0.9) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-main) !important;
    font-family: 'Rajdhani', sans-serif !important;
    border-radius: 4px !important;
}
.stTextInput input:focus, .stNumberInput input:focus {
    border-color: var(--neon-cyan) !important;
    box-shadow: 0 0 0 1px var(--neon-cyan) !important;
}

/* Button */
.stButton > button {
    font-family: 'Orbitron', monospace !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.2em !important;
    text-transform: uppercase !important;
    background: transparent !important;
    border: 1px solid var(--neon-cyan) !important;
    color: var(--neon-cyan) !important;
    padding: 0.85rem 2rem !important;
    border-radius: 4px !important;
    width: 100% !important;
    transition: all 0.3s ease !important;
    position: relative !important;
    overflow: hidden !important;
}
.stButton > button:hover {
    background: rgba(0,245,255,0.1) !important;
    box-shadow: 0 0 20px rgba(0,245,255,0.3), inset 0 0 20px rgba(0,245,255,0.05) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}

/* Spinner */
.stSpinner > div {
    border-top-color: var(--neon-cyan) !important;
}

/* Expander */
.streamlit-expanderHeader {
    background: rgba(0,245,255,0.04) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.2em !important;
    color: var(--neon-cyan) !important;
}

/* Image */
.stImage img {
    border-radius: 6px !important;
    border: 1px solid var(--border) !important;
}

/* Labels */
label {
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.08em !important;
    color: var(--text-dim) !important;
    text-transform: uppercase !important;
}

/* Download button */
.stDownloadButton > button {
    font-family: 'Orbitron', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.15em !important;
    background: rgba(57,255,20,0.08) !important;
    border: 1px solid rgba(57,255,20,0.4) !important;
    color: var(--neon-green) !important;
    border-radius: 4px !important;
    width: 100% !important;
    padding: 0.75rem !important;
}
.stDownloadButton > button:hover {
    background: rgba(57,255,20,0.15) !important;
    box-shadow: 0 0 15px rgba(57,255,20,0.2) !important;
}

/* Progress */
.stProgress > div > div {
    background: linear-gradient(90deg, var(--neon-cyan), var(--neon-green)) !important;
    border-radius: 2px !important;
}

/* Hide streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0 !important; max-width: 1200px; }
</style>
""", unsafe_allow_html=True)


# ── Device Setup ─────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float16 if DEVICE.type == "cuda" else torch.float32
ATTACK_RES = 512
OUTPUT_DIR = "/kaggle/working" if os.path.exists("/kaggle") else "."
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Load Models ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse", torch_dtype=DTYPE
    ).to(DEVICE)
    vae.eval()
    vae.requires_grad_(False)

    clip_model = CLIPModel.from_pretrained(
        "openai/clip-vit-large-patch14", torch_dtype=DTYPE
    ).to(DEVICE)
    clip_model.eval()
    clip_model.requires_grad_(False)

    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    return vae, clip_model, clip_processor


# ── Utility Functions ─────────────────────────────────────────────────────────
def pil_to_tensor(pil_img, size=ATTACK_RES):
    img = pil_img.convert("RGB").resize((size, size), Image.LANCZOS)
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2,0,1).unsqueeze(0).to(DEVICE, dtype=torch.float32)

def tensor_to_pil(tensor):
    arr = tensor.squeeze(0).clamp(0,1).detach().cpu().permute(1,2,0).numpy()
    return Image.fromarray((arr*255).astype(np.uint8))

def apply_perturbation_fullres(original_pil, delta_lowres):
    orig_w, orig_h = original_pil.size
    delta_fullres = F.interpolate(delta_lowres, size=(orig_h, orig_w),
                                   mode="bilinear", align_corners=False)
    orig_tensor = (
        torch.from_numpy(np.array(original_pil.convert("RGB")).astype(np.float32)/255.0)
        .permute(2,0,1).unsqueeze(0).to(DEVICE, dtype=torch.float32)
    )
    return tensor_to_pil((orig_tensor + delta_fullres).clamp(0,1))

def compute_ssim(img_a, img_b):
    size = min(img_a.size[0],1024), min(img_a.size[1],1024)
    a = np.array(img_a.convert("RGB").resize(size, Image.LANCZOS))
    b = np.array(img_b.convert("RGB").resize(size, Image.LANCZOS))
    return compare_ssim(a, b, channel_axis=2, data_range=255)

def compute_vae_loss(vae_model, perturbed):
    perturbed_scaled = perturbed * 2.0 - 1.0
    pert_latent = vae_model.encode(perturbed_scaled.to(DTYPE)).latent_dist.mean
    target_latent = torch.zeros_like(pert_latent)
    return F.mse_loss(pert_latent, target_latent)

def compute_clip_loss(model, processor, perturbed):
    clip_mean = torch.tensor([0.48145466,0.4578275,0.40821073], device=DEVICE).view(1,3,1,1)
    clip_std  = torch.tensor([0.26862954,0.26130258,0.27577711], device=DEVICE).view(1,3,1,1)
    img_clip = F.interpolate(perturbed, size=(224,224), mode="bilinear", align_corners=False)
    img_clip = (img_clip - clip_mean) / clip_std

    # get_image_features returns a plain tensor, but vision_model returns
    # BaseModelOutputWithPooling — use vision_model + projection instead
    vision_outputs = model.vision_model(pixel_values=img_clip.to(DTYPE))
    pooled = vision_outputs.pooler_output          # shape: [1, hidden_dim]
    image_emb = model.visual_projection(pooled)    # shape: [1, projection_dim]
    image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)

    gen = torch.Generator(device="cpu")
    gen.manual_seed(999)
    target_emb = torch.randn(1, image_emb.shape[-1], generator=gen).to(DEVICE, dtype=DTYPE)
    target_emb = target_emb / target_emb.norm(dim=-1, keepdim=True)
    return F.mse_loss(image_emb, target_emb)

def eot_single_transform(x):
    """
    Apply ONE random transformation to x (in-place style, no list).
    Called once per EOT sample inside the loop to avoid holding
    all augmented tensors in VRAM simultaneously.
    """
    t = x.clone()

    # 1. Random brightness shift +/-8%
    brightness = 1.0 + (torch.rand(1).item() - 0.5) * 0.16
    t = (t * brightness).clamp(0, 1)

    # 2. Random contrast jitter +/-10%
    contrast = 1.0 + (torch.rand(1).item() - 0.5) * 0.20
    mean = t.mean(dim=[2, 3], keepdim=True)
    t = ((t - mean) * contrast + mean).clamp(0, 1)

    # 3. Mild Gaussian noise (simulates JPEG artifacts)
    noise_std = torch.rand(1).item() * 0.012
    t = (t + torch.randn_like(t) * noise_std).clamp(0, 1)

    # 4. Random resize-and-restore (simulates compression)
    scale = 0.88 + torch.rand(1).item() * 0.12   # 88%-100%
    h, w  = t.shape[2], t.shape[3]
    new_h, new_w = max(32, int(h * scale)), max(32, int(w * scale))
    t = F.interpolate(t, size=(new_h, new_w), mode="bilinear", align_corners=False)
    t = F.interpolate(t, size=(h, w),         mode="bilinear", align_corners=False)

    return t


@torch.enable_grad()
def pgd_attack(original_pil, steps=50, epsilon=0.04,
               vae_weight=1.0, clip_weight=0.5,
               eot_samples=4, progress_bar=None):
    """
    PGD + EOT (Expectation Over Transformations).
    KEY FIX: Each EOT sample is forward-passed and backwarded individually
    (gradient accumulation), so only ONE computational graph lives in VRAM
    at a time instead of all eot_samples graphs simultaneously.
    This cuts peak VRAM usage by ~(eot_samples - 1)x.
    """
    vae, clip_model, clip_processor = load_models()

    # Free any cached memory before starting
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    x_orig = pil_to_tensor(original_pil)
    x_orig.requires_grad_(False)
    delta = torch.zeros_like(x_orig, requires_grad=True, device=DEVICE)
    alpha = epsilon / (steps * 0.4)
    last_loss = 0.0

    for step in range(steps):
        # ── EOT: gradient accumulation — ONE sample at a time ─────────────
        # CRITICAL: x_adv is recomputed INSIDE each EOT sample, not outside.
        # If we compute x_adv once and reuse it, PyTorch frees its saved
        # intermediates after the first .backward(), causing:
        # "Trying to backward through the graph a second time"
        # Recomputing x_adv each iteration creates a fresh graph every time.
        accumulated_loss = 0.0

        for _ in range(eot_samples):
            # Fresh graph per sample — this is the key fix
            x_adv_i = (x_orig + delta).clamp(0, 1)
            t_img   = eot_single_transform(x_adv_i)

            l_vae  = compute_vae_loss(vae, t_img)
            l_clip = compute_clip_loss(clip_model, clip_processor, t_img)
            sample_loss = (vae_weight * l_vae + clip_weight * l_clip) / eot_samples

            # backward() immediately — frees this graph from VRAM
            sample_loss.backward()
            accumulated_loss += sample_loss.item()

            del x_adv_i, t_img, l_vae, l_clip, sample_loss
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
        # ──────────────────────────────────────────────────────────────────

        with torch.no_grad():
            grad = delta.grad.detach()
            delta.data = delta.data - alpha * torch.sign(grad)
            delta.data = delta.data.clamp(-epsilon, epsilon)
            delta.data = (x_orig + delta.data).clamp(0, 1) - x_orig
        delta.grad.zero_()

        last_loss = accumulated_loss

        if progress_bar is not None:
            progress_bar.progress(
                (step + 1) / steps,
                text=f"⚡ Optimizing shield... Step {step+1}/{steps} | EOT Loss: {last_loss:.4f}"
            )

    # Final cleanup before generating output
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    protected_pil = apply_perturbation_fullres(original_pil, delta.detach())
    ssim_val = compute_ssim(original_pil, protected_pil)
    return protected_pil, {"ssim": ssim_val, "epsilon": epsilon, "steps": steps}


# ── HERO SECTION ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-block">
    <div class="hero-tag">◈ Adversarial Defense System</div>
    <h1 class="hero-title">CLOAK ID</h1>
    <p class="hero-sub">Dual-Layer Adversarial Immunization · VAE + CLIP Neural Defense</p>
    <div class="hero-divider">
        <div class="hero-divider-line"></div>
        <div class="hero-divider-dot"></div>
        <div class="hero-divider-line"></div>
    </div>
</div>
""", unsafe_allow_html=True)

device_label = "CUDA GPU" if DEVICE.type == "cuda" else "CPU"
st.markdown(f"""
<div class="badge-row">
    <span class="badge">◈ Module 01 — Immunization</span>
    <span class="badge">⬡ Engine: {device_label}</span>
    <span class="badge">◈ PGD Attack</span>
    <span class="badge">⬡ VAE + CLIP Defense</span>
</div>
""", unsafe_allow_html=True)

# ── MAIN LAYOUT ───────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown('<div class="cyber-card-title">◈ Input Image</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drop your image file here",
        type=["png", "jpg", "jpeg", "webp"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        original_image = Image.open(uploaded_file)
        st.image(original_image, use_container_width=True, caption="")
        w, h = original_image.size
        st.markdown(f"""
        <div style="display:flex; gap:0.8rem; margin-top:0.5rem;">
            <span style="font-family:'Orbitron',monospace; font-size:0.6rem; 
                         color:#5b8fa8; letter-spacing:0.15em;">
                RES {w}×{h}
            </span>
            <span style="font-family:'Orbitron',monospace; font-size:0.6rem;
                         color:#5b8fa8; letter-spacing:0.15em;">
                MODE RGB
            </span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="height:280px; display:flex; flex-direction:column; align-items:center;
                    justify-content:center; border:1px dashed rgba(0,245,255,0.15);
                    border-radius:8px; color:#5b8fa8;">
            <div style="font-size:2.5rem; margin-bottom:0.8rem; opacity:0.4;">◈</div>
            <div style="font-family:'Orbitron',monospace; font-size:0.65rem;
                        letter-spacing:0.2em; text-align:center;">
                NO IMAGE LOADED<br>
                <span style="opacity:0.5; font-size:0.55rem;">PNG · JPG · WEBP</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Settings
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="cyber-card-title">⚙ Protection Parameters</div>', unsafe_allow_html=True)

    epsilon = st.slider(
        "Shield Strength (Epsilon)",
        min_value=0.01, max_value=0.08, value=0.04, step=0.01,
        help="Higher = stronger protection, more visible noise"
    )
    steps = st.slider(
        "Optimization Steps",
        min_value=10, max_value=100, value=50, step=10,
        help="More steps = better protection, slower processing"
    )

    with st.expander("⬡ ADVANCED LAYER WEIGHTS"):
        adv_col1, adv_col2 = st.columns(2)
        with adv_col1:
            vae_weight = st.slider("VAE Layer Weight", 0.1, 2.0, 1.0, 0.1)
        with adv_col2:
            clip_weight = st.slider("CLIP Layer Weight", 0.1, 2.0, 0.5, 0.1)

    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("◈  APPLY IMMUNIZATION", use_container_width=True)


with col_right:
    st.markdown('<div class="cyber-card-title">◈ Protected Output</div>', unsafe_allow_html=True)

    result_placeholder = st.empty()
    status_placeholder = st.empty()
    metrics_placeholder = st.empty()
    download_placeholder = st.empty()

    # Default state
    result_placeholder.markdown("""
    <div style="height:280px; display:flex; flex-direction:column; align-items:center;
                justify-content:center; border:1px dashed rgba(0,245,255,0.08);
                border-radius:8px; color:#5b8fa8;">
        <div style="font-size:2.5rem; margin-bottom:0.8rem; opacity:0.3;">🛡</div>
        <div style="font-family:'Orbitron',monospace; font-size:0.65rem;
                    letter-spacing:0.2em; text-align:center; opacity:0.6;">
            AWAITING IMMUNIZATION<br>
            <span style="opacity:0.5; font-size:0.55rem;">Protected image will appear here</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if run_btn:
        if uploaded_file is None:
            status_placeholder.markdown("""
            <div class="warn-alert">⚠ No image loaded. Please upload an image first.</div>
            """, unsafe_allow_html=True)
        else:
            result_placeholder.empty()
            progress_bar = result_placeholder.progress(0, text="⚡ Initializing CloakID engine...")

            # Load models info
            status_placeholder.markdown("""
            <div style="font-family:'Rajdhani',sans-serif; font-size:0.85rem;
                        color:#5b8fa8; letter-spacing:0.05em; margin-top:0.5rem;">
                ◈ Loading VAE + CLIP neural models...
            </div>
            """, unsafe_allow_html=True)

            try:
                # Load models (cached)
                load_models()

                # Free unused VRAM before attack begins
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                status_placeholder.markdown("""
                <div style="font-family:'Rajdhani',sans-serif; font-size:0.85rem;
                            color:#5b8fa8; letter-spacing:0.05em; margin-top:0.5rem;">
                    ◈ Models ready. Running PGD optimization...
                </div>
                """, unsafe_allow_html=True)

                protected_pil, metrics = pgd_attack(
                    original_pil=original_image,
                    steps=int(steps),
                    epsilon=float(epsilon),
                    vae_weight=float(vae_weight),
                    clip_weight=float(clip_weight),
                    progress_bar=progress_bar
                )

                # Show result
                result_placeholder.image(protected_pil, use_container_width=True)

                # SSIM quality
                ssim = metrics["ssim"]
                quality_label = "EXCELLENT" if ssim >= 0.95 else "GOOD" if ssim >= 0.90 else "DEGRADED"
                quality_color = "#39ff14" if ssim >= 0.95 else "#00f5ff" if ssim >= 0.90 else "#ff6b35"

                status_placeholder.markdown(f"""
                <div class="success-alert">
                    ✓ Immunization complete — shield successfully applied
                </div>
                """, unsafe_allow_html=True)

                metrics_placeholder.markdown(f"""
                <div class="metric-grid">
                    <div class="metric-box">
                        <div class="metric-value" style="color:{quality_color};">{ssim:.4f}</div>
                        <div class="metric-label">SSIM Score</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value">{epsilon}</div>
                        <div class="metric-label">Epsilon</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value">{steps}</div>
                        <div class="metric-label">Steps</div>
                    </div>
                </div>
                <div style="text-align:center; margin-top:0.8rem;">
                    <span style="font-family:'Orbitron',monospace; font-size:0.6rem;
                                 color:{quality_color}; letter-spacing:0.2em;">
                        VISUAL FIDELITY: {quality_label}
                    </span>
                </div>
                """, unsafe_allow_html=True)

                # Download
                buf = io.BytesIO()
                protected_pil.save(buf, format="PNG", compress_level=1)
                buf.seek(0)
                download_placeholder.download_button(
                    label="⬇ DOWNLOAD PROTECTED IMAGE (PNG)",
                    data=buf,
                    file_name="cloakid_protected.png",
                    mime="image/png",
                    use_container_width=True
                )

            except Exception as e:
                result_placeholder.empty()
                status_placeholder.markdown(f"""
                <div class="warn-alert">❌ Error during immunization: {str(e)}</div>
                """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 3rem 1rem 2rem;
            border-top: 1px solid rgba(0,245,255,0.08); margin-top:3rem;">
    <div style="font-family:'Orbitron',monospace; font-size:0.55rem;
                letter-spacing:0.4em; color:#1a3a4a; text-transform:uppercase;
                margin-bottom:1rem;">
        CloakID · Phase 01 · Adversarial Immunization Engine · VAE + CLIP · EOT
    </div>
    <div style="font-family:'Rajdhani',sans-serif; font-size:0.95rem;
                color:#2a5a72; max-width:680px; margin:0 auto; line-height:1.7;
                font-weight:300; letter-spacing:0.03em;">
        CloakID is a privacy-first adversarial defense system that embeds imperceptible 
        noise into personal images — silently disrupting how AI models perceive and 
        manipulate them, so your identity stays yours.
    </div>
</div>
""", unsafe_allow_html=True)
'''

with open("app.py", "w") as f:
    f.write(app_code)

print("✅ app.py written successfully")


# ---------------------------
# CELL 3 — Launch with ngrok
# ---------------------------
import subprocess, threading, time, sys
from pyngrok import ngrok, conf

# ✅ PASTE YOUR NGROK AUTHTOKEN HERE
NGROK_AUTH_TOKEN = "3BHYmJsobAuAkrrXRdA05dU4dQS_7w4TgN3xCFA7RCCASCm2G"

conf.get_default().auth_token = NGROK_AUTH_TOKEN

def run_streamlit():
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", "app.py",
        "--server.port", "8501",
        "--server.headless", "true",
        "--server.enableCORS", "false",
        "--server.enableXsrfProtection", "false"
    ])

thread = threading.Thread(target=run_streamlit, daemon=True)
thread.start()

time.sleep(6)

public_url = ngrok.connect(8502)
print("=" * 55)
print(f"  ✅  CloakID is LIVE at: {public_url}")
print("=" * 55)
print("  👆  Click the link above to open your app!")
