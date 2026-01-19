import gradio as gr
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from diffusers import AutoencoderKL
from PIL import Image
import numpy as np
import os

# 1. SETUP DEVICE
# Hugging Face ZeroGPU uses 'cuda', CPU Basic uses 'cpu'
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f" CloakID is running on: {device}")

# 2. LOAD MODELS
print(" Loading VAE Model...")
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
vae.requires_grad_(False)
print(" Model Loaded.")

# 3. HELPER FUNCTIONS
def preprocess(image):
    image = image.resize((512, 512), resample=Image.BILINEAR)
    tensor = T.ToTensor()(image)
    tensor = tensor * 2.0 - 1.0 
    return tensor.unsqueeze(0).to(device)

def deprocess(tensor):
    tensor = tensor.detach().cpu().squeeze(0)
    tensor = (tensor + 1.0) / 2.0
    tensor = torch.clamp(tensor, 0, 1)
    return T.ToPILImage()(tensor)

# 4. DEFENSE ENGINE (PGD ATTACK)
def cloak_image(original_tensor, epsilon, steps):
    # Setup perturbation
    delta = torch.zeros_like(original_tensor, requires_grad=True).to(device)
    target_latents = torch.zeros((1, 4, 64, 64)).to(device) # Gray Target
    
    # Optimizer settings
    step_size = 0.01
    
    for i in range(steps):
        adv_image = original_tensor + delta
        
        # VAE Encoding
        current_latents = vae.encode(adv_image).latent_dist.mean
        
        # Loss
        loss = F.mse_loss(current_latents, target_latents)
        
        # Gradient Step
        if delta.grad is not None:
            delta.grad.zero_()
        loss.backward()
        
        with torch.no_grad():
            # Update Noise
            delta.data = delta.data - step_size * torch.sign(delta.grad)
            
            # Projection (Clamp)
            delta.data = torch.clamp(delta.data, -epsilon, epsilon)
            adv_check = original_tensor + delta.data
            adv_check = torch.clamp(adv_check, -1, 1)
            delta.data = adv_check - original_tensor
            
    return (original_tensor + delta).detach()

# 5. GRADIO WRAPPER
def process_and_protect(input_image, intensity, steps):
    if input_image is None:
        return None, None

    # Status update for user
    print("Starting protection...")
    
    # Preprocess
    input_pil = Image.fromarray(input_image).convert("RGB")
    clean_tensor = preprocess(input_pil)
    
    # Attack
    protected_tensor = cloak_image(clean_tensor, epsilon=intensity, steps=int(steps))
    
    # Verification
    with torch.no_grad():
        broken_latents = vae.encode(protected_tensor).latent_dist.sample()
        broken_recon = vae.decode(broken_latents).sample
    
    # Postprocess
    final_protected = deprocess(protected_tensor)
    final_proof = deprocess(broken_recon)
    
    return final_protected, final_proof

# 6. UI LAYOUT
with gr.Blocks(title="CloakID Project", theme=gr.themes.Soft()) as demo:
    gr.Markdown("#  CloakID: Adversarial Defense")
    gr.Markdown("MCA Main Project | Phase 1 Prototype")
    gr.Markdown("Upload a photo to immunize it against Latent Diffusion Models (Deepfakes).")
    
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(label="Original Photo")
            eps_slider = gr.Slider(0.01, 0.1, value=0.03, step=0.01, label="Intensity (Epsilon)")
            steps_slider = gr.Slider(20, 100, value=40, step=10, label="Steps (Lower = Faster)")
            run_btn = gr.Button("Protect Identity", variant="primary")
        
        with gr.Column():
            output_cloaked = gr.Image(label="Protected Image")
            output_proof = gr.Image(label="AI Verification (Noise)")
            
    run_btn.click(
        process_and_protect, 
        inputs=[input_img, eps_slider, steps_slider], 
        outputs=[output_cloaked, output_proof]
    )

demo.launch()