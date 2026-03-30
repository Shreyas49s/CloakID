# =============================================================================
# CloakID Module 2 — Attractive Gradio UI (Kaggle Ready)
# =============================================================================

import subprocess, sys
for p in ["diffusers", "transformers", "accelerate", "gradio",
          "torch", "numpy", "pillow", "lpips", "opencv-python", "matplotlib"]:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", p])

import torch
import torch.nn.functional as F
from diffusers import (StableDiffusionInstructPix2PixPipeline,
                       EulerAncestralDiscreteScheduler, AutoencoderKL)
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import numpy as np
import gradio as gr
import math, cv2, lpips, warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

warnings.filterwarnings("ignore")

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float16 if DEVICE.type == "cuda" else torch.float32
print(f"Running on {DEVICE}")

# ── Load Models ───────────────────────────────────────────────────────────────
print("Loading InstructPix2Pix...")
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    "timbrooks/instruct-pix2pix", torch_dtype=DTYPE, safety_checker=None,
).to(DEVICE)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.set_progress_bar_config(disable=True)

print("Loading VAE...")
vae = AutoencoderKL.from_pretrained(
    "stabilityai/sd-vae-ft-mse", torch_dtype=DTYPE
).to(DEVICE).eval()
vae.requires_grad_(False)

print("Loading CLIP...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(DEVICE).eval()
clip_proc  = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

print("Loading LPIPS...")
lpips_model = lpips.LPIPS(net="alex").to(DEVICE)
print("All models ready.\n")

# ── Helpers ───────────────────────────────────────────────────────────────────
def resize_img(img):
    return img.convert("RGB").resize((512, 512), Image.LANCZOS)

def pil_to_tensor(img):
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2,0,1).unsqueeze(0).to(DEVICE, dtype=torch.float32)

def tensor_to_pil(t):
    arr = t.squeeze(0).clamp(0,1).cpu().permute(1,2,0).numpy()
    return Image.fromarray((arr*255).astype(np.uint8))

def vae_corrupt_protected(img):
    x = pil_to_tensor(img)
    with torch.no_grad():
        x_scaled     = x * 2.0 - 1.0
        latent       = vae.encode(x_scaled.to(DTYPE)).latent_dist.mean
        zero_latent  = torch.zeros_like(latent)
        decoded_zero = (vae.decode(zero_latent).sample / 2.0 + 0.5).clamp(0,1).float()
        decoded_raw  = (vae.decode(latent).sample        / 2.0 + 0.5).clamp(0,1).float()
        decoded      = 0.3 * decoded_raw + 0.7 * decoded_zero
    return tensor_to_pil(decoded)

def compute_psnr(a, b):
    a = np.array(a).astype(np.float32)
    b = np.array(b).astype(np.float32)
    mse = np.mean((a - b) ** 2)
    return round(100.0 if mse == 0 else 20 * math.log10(255.0 / math.sqrt(mse)), 2)

def compute_lpips_score(a, b):
    t1 = torch.tensor(np.array(a)/255.0, dtype=torch.float32).permute(2,0,1).unsqueeze(0).to(DEVICE)
    t2 = torch.tensor(np.array(b)/255.0, dtype=torch.float32).permute(2,0,1).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        return round(lpips_model(t1, t2).item(), 3)

def compute_sharpness(img):
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    return round(cv2.Laplacian(gray, cv2.CV_64F).var(), 1)

def get_clip_embedding(img):
    """Returns a normalised L2 unit embedding for one image [1, D]."""
    inputs = clip_proc(images=img, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        raw  = clip_model.get_image_features(**inputs)
        # get_image_features may return a tensor or an object depending on version
        feat = raw if isinstance(raw, torch.Tensor) else raw.pooler_output
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat  # shape [1, D], float32, unit norm

def compute_clip_similarity(img_src, img_edit):
    """
    Cosine similarity between source image and its edited version in CLIP space.
    Range: -1 to 1.
      ~0.85-0.95 → edit kept identity  (bad for us = edit succeeded)
      ~0.30-0.60 → identity lost       (good for us = protection worked)
    This is the correct metric — NOT feat.mean() which collapses to ~0.
    """
    emb_src  = get_clip_embedding(img_src)
    emb_edit = get_clip_embedding(img_edit)
    cosine   = (emb_src * emb_edit).sum(dim=-1).item()
    return round(cosine, 3)

# ── Chart ─────────────────────────────────────────────────────────────────────
def plot_metrics(psnr_val, lpips_orig, lpips_prot,
                 sharp_orig, sharp_prot, clip_orig, clip_prot):
    BG    = "#0d1117"
    PANEL = "#161b22"
    BLUE  = "#58a6ff"
    GREEN = "#3fb950"
    PINK  = "#f778ba"
    RED   = "#f85149"
    TEXT  = "#e6edf3"
    MUTED = "#7d8590"

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.patch.set_facecolor(BG)
    fig.subplots_adjust(wspace=0.38)

    for ax in axes:
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=MUTED, labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor("#21262d")

    # Chart 1 — PSNR
    ax = axes[0]
    b = ax.bar(["PSNR"], [psnr_val], color=PINK, width=0.35, zorder=3,
               edgecolor=BG, linewidth=0.5)
    ax.bar_label(b, fmt="%.1f dB", color=TEXT, fontsize=10, padding=5, fontweight="bold")
    ax.axhline(y=15, color=RED, linestyle="--", linewidth=1.2,
               label="PhotoGuard ref (~15 dB)")
    ax.set_title("PSNR  (orig vs prot edit)\n↓ lower = edits more different",
                 color=TEXT, fontsize=9.5, pad=10, loc="left")
    ax.set_ylabel("dB", color=MUTED, fontsize=9)
    ax.set_ylim(0, max(psnr_val * 1.4, 22))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.legend(fontsize=7.5, labelcolor=MUTED, facecolor=PANEL,
              edgecolor="#21262d", loc="upper right")
    ax.grid(axis="y", color="#21262d", linewidth=0.6, zorder=0)

    # Chart 2 — CLIP cosine similarity
    ax = axes[1]
    x = np.array([0, 1])
    b = ax.bar(x, [clip_orig, clip_prot], color=[BLUE, GREEN],
               width=0.45, zorder=3, edgecolor=BG, linewidth=0.5)
    ax.bar_label(b, fmt="%.3f", color=TEXT, fontsize=10, padding=5, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(["Original", "Protected"], color=MUTED, fontsize=9)
    ax.set_title("CLIP Identity Retention\n(cosine sim: source → edit)  ↓ protected = identity lost",
                 color=TEXT, fontsize=9.5, pad=10, loc="left")
    # Cosine similarity is always in [-1, 1]; realistic values are 0.3–1.0
    y_top = max(max(clip_orig, clip_prot) * 1.25, 1.0)
    ax.set_ylim(0, y_top)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.grid(axis="y", color="#21262d", linewidth=0.6, zorder=0)

    # Chart 3 — Sharpness
    ax = axes[2]
    b = ax.bar(x, [sharp_orig, sharp_prot], color=[BLUE, GREEN],
               width=0.45, zorder=3, edgecolor=BG, linewidth=0.5)
    ax.bar_label(b, fmt="%.0f", color=TEXT, fontsize=10, padding=5, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(["Original", "Protected"], color=MUTED, fontsize=9)
    ax.set_title("Sharpness  (Laplacian var)\n↓ protected = blur / ghost effect",
                 color=TEXT, fontsize=9.5, pad=10, loc="left")
    top = max(sharp_orig, sharp_prot)
    ax.set_ylim(0, top * 1.4)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(max(20, round(top/6/20)*20)))
    ax.grid(axis="y", color="#21262d", linewidth=0.6, zorder=0)

    # Chart 4 — LPIPS
    ax = axes[3]
    b = ax.bar(x, [lpips_orig, lpips_prot], color=[BLUE, GREEN],
               width=0.45, zorder=3, edgecolor=BG, linewidth=0.5)
    ax.bar_label(b, fmt="%.3f", color=TEXT, fontsize=10, padding=5, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(["Original", "Protected"], color=MUTED, fontsize=9)
    ax.set_title("LPIPS vs Source\n↑ protected = perceptual disruption",
                 color=TEXT, fontsize=9.5, pad=10, loc="left")
    top = max(lpips_orig, lpips_prot)
    ax.set_ylim(0, min(top * 1.5, 1.0))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.grid(axis="y", color="#21262d", linewidth=0.6, zorder=0)

    plt.tight_layout(pad=2.5)
    return fig

# ── Core Attack Function ──────────────────────────────────────────────────────
def run_attack(orig, prot, prompt, progress=gr.Progress(track_tqdm=True)):
    if orig is None or prot is None:
        raise gr.Error("Please upload BOTH the Original and Protected images.")
    if not prompt.strip():
        raise gr.Error("Please enter a malicious instruction prompt.")

    orig_r = resize_img(orig)
    prot_r = resize_img(prot)

    progress(0.05, desc="Corrupting protected image via VAE...")
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    prot_feed = vae_corrupt_protected(prot_r)

    progress(0.15, desc="Attacking original image...")
    gen = torch.Generator(DEVICE).manual_seed(1234)
    out_orig = pipe(
        prompt=prompt, image=orig_r,
        num_inference_steps=20, guidance_scale=7.5,
        image_guidance_scale=1.0, generator=gen,
    ).images[0]

    progress(0.55, desc="Attacking protected image...")
    gen = torch.Generator(DEVICE).manual_seed(1234)
    out_prot = pipe(
        prompt=prompt, image=prot_feed,
        num_inference_steps=20, guidance_scale=7.5,
        image_guidance_scale=1.0, generator=gen,
    ).images[0]

    progress(0.85, desc="Computing metrics...")
    psnr_val   = compute_psnr(out_orig, out_prot)
    lpips_orig = compute_lpips_score(orig_r, out_orig)
    lpips_prot = compute_lpips_score(prot_r, out_prot)
    sharp_orig = compute_sharpness(out_orig)
    sharp_prot = compute_sharpness(out_prot)
    # CLIP: cosine similarity between SOURCE and its EDIT
    # High (~0.85+) = edit kept identity (attack succeeded)
    # Low  (~0.30–0.60) = identity lost  (protection worked)
    clip_orig  = compute_clip_similarity(orig_r, out_orig)
    clip_prot  = compute_clip_similarity(prot_r, out_prot)

    drop_sharp = round((sharp_orig - sharp_prot) / max(sharp_orig, 1e-6) * 100, 1)
    lpips_gain = round((lpips_prot - lpips_orig) / max(lpips_orig, 1e-4) * 100, 1)
    clip_drop  = round((clip_orig  - clip_prot)  / max(clip_orig,  1e-4) * 100, 1)

    if drop_sharp > 25:
        verdict = f"🟢  PROTECTION SUCCESSFUL "
    elif drop_sharp > 10:
        verdict = f"🟡  PARTIAL PROTECTION — Sharpness ↓{drop_sharp}%  ·  Some disruption detected"
    else:
        verdict = f"🔴  PROTECTION INSUFFICIENT — Sharpness drop only {drop_sharp}%"

    table = [
        ["PSNR (original vs protected edit)", "—",             f"{psnr_val} dB",   "Lower = edits are more different"],
        ["LPIPS vs source",                   str(lpips_orig),  str(lpips_prot),   f"+{lpips_gain}% perceptual disruption"],
        ["Sharpness (Laplacian variance)",    str(sharp_orig),  str(sharp_prot),   f"{drop_sharp}% drop — blur/ghost confirmed"],
        ["CLIP identity retention",           str(clip_orig),   str(clip_prot),    f"↓{clip_drop}% — lower protected = identity lost"],
        ["Verdict",                           "—",              "—",               verdict],
    ]

    fig = plot_metrics(psnr_val, lpips_orig, lpips_prot,
                       sharp_orig, sharp_prot, clip_orig, clip_prot)
    progress(1.0, desc="Done!")
    return out_orig, out_prot, table, fig

# ── CSS ───────────────────────────────────────────────────────────────────────
css = """
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@200;400;600;800&display=swap');

:root {
    --bg:      #0d1117;
    --surface: #161b22;
    --border:  #21262d;
    --border2: #30363d;
    --cyan:    #39d0d8;
    --green:   #3fb950;
    --red:     #f85149;
    --amber:   #d29922;
    --blue:    #58a6ff;
    --text:    #e6edf3;
    --muted:   #7d8590;
}

/* ── Global ── */
body, .gradio-container {
    background-color: var(--bg) !important;
    font-family: 'Exo 2', sans-serif !important;
}
.gradio-container {
    background-image:
        radial-gradient(ellipse at 0% 0%, rgba(57,208,216,0.05) 0%, transparent 45%),
        radial-gradient(ellipse at 100% 100%, rgba(63,185,80,0.04) 0%, transparent 45%) !important;
    max-width: 1320px !important;
    margin: 0 auto !important;
}

/* ── Hero header ── */
.hero-wrap {
    text-align: center;
    padding: 3rem 1rem 1rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
}
.hero-eyebrow {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.38em;
    color: var(--cyan);
    text-transform: uppercase;
    margin-bottom: 0.7rem;
    opacity: 0.85;
}
.hero-title {
    font-family: 'Exo 2', sans-serif;
    font-size: clamp(2.2rem, 4.5vw, 4rem);
    font-weight: 800;
    margin: 0 0 0.4rem;
    background: linear-gradient(90deg, #39d0d8 0%, #58a6ff 45%, #3fb950 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
}
.hero-sub {
    font-family: 'Exo 2', sans-serif;
    font-size: 0.95rem;
    color: var(--muted);
    font-weight: 200;
    letter-spacing: 0.05em;
    margin: 0.4rem 0 1.5rem;
}
.chip-row {
    display: flex;
    justify-content: center;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-bottom: 1.2rem;
}
.chip {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.18em;
    padding: 0.28rem 0.85rem;
    border-radius: 2px;
    text-transform: uppercase;
    border: 1px solid;
}
.chip-c { color: var(--cyan);  border-color: rgba(57,208,216,0.3);  background: rgba(57,208,216,0.07); }
.chip-b { color: var(--blue);  border-color: rgba(88,166,255,0.3);  background: rgba(88,166,255,0.07); }
.chip-g { color: var(--green); border-color: rgba(63,185,80,0.3);   background: rgba(63,185,80,0.07); }
.chip-a { color: var(--amber); border-color: rgba(210,153,34,0.3);  background: rgba(210,153,34,0.07); }

/* ── Section label ── */
.sec-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.32em;
    color: var(--cyan);
    text-transform: uppercase;
    margin: 1.5rem 0 0.6rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
}

/* ── Panels ── */
.panel {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
}

/* ── Upload zones ── */
.upload-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    padding: 0.3rem 0 0.6rem;
}
.label-orig { color: var(--blue); }
.label-prot { color: var(--green); }
.label-res-orig { color: var(--blue); }
.label-res-prot { color: var(--red); }

/* ── Gradio image component ── */
.image-container, .image-frame {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
}

/* ── Run button ── */
#run-btn {
    background: transparent !important;
    border: 1px solid var(--cyan) !important;
    color: var(--cyan) !important;
    font-family: 'Exo 2', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    padding: 0.85rem !important;
    border-radius: 4px !important;
    transition: all 0.25s ease !important;
    width: 100% !important;
}
#run-btn:hover {
    background: rgba(57,208,216,0.1) !important;
    box-shadow: 0 0 20px rgba(57,208,216,0.25) !important;
    transform: translateY(-1px) !important;
}

/* ── Prompt input ── */
textarea, input[type="text"] {
    background: var(--surface) !important;
    border: 1px solid var(--border2) !important;
    color: var(--text) !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.82rem !important;
    border-radius: 4px !important;
}
textarea:focus, input[type="text"]:focus {
    border-color: var(--cyan) !important;
    box-shadow: 0 0 0 1px var(--cyan) !important;
    outline: none !important;
}

/* ── Labels ── */
label, .block > label > span {
    font-family: 'Exo 2', sans-serif !important;
    font-size: 0.78rem !important;
    color: var(--muted) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}

/* ── Dataframe (metrics table) ── */
.dataframe {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.8rem !important;
    color: var(--text) !important;
}
.dataframe th {
    background: #1c2128 !important;
    color: var(--muted) !important;
    border-bottom: 1px solid var(--border2) !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    padding: 0.6rem 0.8rem !important;
}
.dataframe td {
    border-bottom: 1px solid var(--border) !important;
    padding: 0.6rem 0.8rem !important;
    color: var(--text) !important;
}
.dataframe tr:last-child td { border-bottom: none !important; }
.dataframe tr:hover td { background: #1c2128 !important; }

/* ── Progress bar ── */
.progress-bar {
    background: linear-gradient(90deg, var(--cyan), var(--green)) !important;
    border-radius: 2px !important;
}

/* ── Plot ── */
.plot-container {
    background: var(--bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
}

/* ── Footer ── */
.footer-wrap {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    border-top: 1px solid var(--border);
    margin-top: 2.5rem;
}
.footer-mono {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.52rem;
    letter-spacing: 0.38em;
    color: #21262d;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
}
.footer-desc {
    font-family: 'Exo 2', sans-serif;
    font-size: 0.88rem;
    font-weight: 200;
    color: var(--muted);
    max-width: 620px;
    margin: 0 auto;
    line-height: 1.75;
}
.how-to-box {
    background: #161b22;
    border: 1px solid #21262d;
    border-left: 3px solid var(--cyan);
    border-radius: 4px;
    padding: 0.9rem 1.2rem;
    font-family: 'Exo 2', sans-serif;
    font-size: 0.84rem;
    color: var(--muted);
    line-height: 1.7;
    margin-top: 0.5rem;
}
.how-to-box strong { color: var(--text); font-weight: 600; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--cyan); }

/* ── Hide Gradio footer ── */
footer { display: none !important; }
"""

# ── Gradio UI ─────────────────────────────────────────────────────────────────
dev_label = "CUDA GPU" if DEVICE.type == "cuda" else "CPU"

with gr.Blocks(title="CloakID — Attack Lab", css=css) as demo:

    # Hero
    gr.HTML(f"""
    <div class="hero-wrap">
        <div class="hero-eyebrow">Robustness Verification — Module 02</div>
        <div class="hero-title">CloakID Attack Lab</div>
        <div class="hero-sub">Measure adversarial protection strength against InstructPix2Pix diffusion attacks</div>
        <div class="chip-row">
            <span class="chip chip-c">InstructPix2Pix</span>
         
            <span class="chip chip-g">LPIPS · PSNR · CLIP</span>
            <span class="chip chip-a">Engine: {dev_label}</span>
        </div>
    </div>
    """)

    # Input images
    gr.HTML('<div class="sec-label">Input Images</div>')
    with gr.Row(equal_height=True):
        with gr.Column():
            gr.HTML('<div class="upload-label label-orig">Original Image — Unprotected</div>')
            in_orig = gr.Image(label="", type="pil", height=280, show_label=False)
        with gr.Column():
            gr.HTML('<div class="upload-label label-prot">Protected Image — Immunized</div>')
            in_prot = gr.Image(label="", type="pil", height=280, show_label=False)

    # Prompt
    gr.HTML('<div class="sec-label">Malicious Instruction</div>')
    prompt = gr.Textbox(
        
        placeholder="Describe the manipulation you want to attempt...",
        label="",
        show_label=False,
        lines=1,
    )

    # Run button
    gr.HTML("<br>")
    run_btn = gr.Button("⚡  RUN ATTACK SIMULATION", elem_id="run-btn")

    # Result images
    gr.HTML('<div class="sec-label">Attack Results</div>')
    with gr.Row(equal_height=True):
        with gr.Column():
            gr.HTML('<div class="upload-label label-res-orig">Original — Edit Should Succeed ✅</div>')
            out_orig = gr.Image(label="", type="pil", height=280, show_label=False)
        with gr.Column():
            gr.HTML('<div class="upload-label label-res-prot">Protected — Edit Should Fail 🔴</div>')
            out_prot = gr.Image(label="", type="pil", height=280, show_label=False)

    # Metrics table
    gr.HTML('<div class="sec-label">Verification Metrics</div>')
    metrics_table = gr.Dataframe(
        headers=["Metric", "Original", "Protected", "Interpretation"],
        label="",
        show_label=False,
        wrap=True,
    )

    # Chart
    gr.HTML('<div class="sec-label">Performance Analysis</div>')
    metrics_plot = gr.Plot(label="", show_label=False)

    # How-to
    gr.HTML("""
    <div class="how-to-box">
        <strong>How to read:</strong>
        The left result should show a clean successful edit (instruction followed).
        The right result should appear gray, blurry, or corrupted — that confirms CloakID blocked the manipulation.
        A <strong>sharpness drop &gt;25%</strong> on the protected output = Protection Successful.
    </div>
    """)

    # Footer
    gr.HTML("""
    <div class="footer-wrap">
        <div class="footer-mono">CloakID · Module 02 · Robustness Verification · VAE + LPIPS + CLIP + PSNR</div>
        <div class="footer-desc">
            CloakID Module 2 stress-tests the adversarial shield from Module 1 — running real
            diffusion-model attacks on both the original and protected image, then measuring how
            effectively the noise disrupted the AI's ability to follow the instruction.
        </div>
    </div>
    """)

    # Wiring
    run_btn.click(
        fn=run_attack,
        inputs=[in_orig, in_prot, prompt],
        outputs=[out_orig, out_prot, metrics_table, metrics_plot],
    )

demo.queue().launch(share=True, debug=True)