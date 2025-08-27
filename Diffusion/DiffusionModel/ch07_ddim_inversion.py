# -*- coding: utf-8 -*-
# @project: LargeDemo
# @filename: chapter7_ddim_inversion.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/8/25 15:00

# https://chat.deepseek.com/a/chat/s/b4368206-436a-491f-9f53-501b32df0686

# stable_diffusion_utils.py

import torch
import requests
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from io import BytesIO
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from torchvision import transforms as tfms
from diffusers import StableDiffusionPipeline, DDIMScheduler, StableDiffusionControlNetPipeline, ControlNetModel, \
    UniPCMultistepScheduler
from diffusers.utils import load_image as diffusers_load_image
import cv2
import numpy as np
from controlnet_aux import OpenposeDetector

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global pipe variable to avoid reloading multiple times
_pipe = None
_controlnet_pipe = None


def setup_environment():
    """Install required packages (commented out as they should be pre-installed)"""
    # !pip install -q transformers diffusers accelerate
    # !pip install opencv-contrib-python
    # !pip install controlnet_aux
    pass


def load_image(url, size=None):
    """Load image from URL and resize if needed"""
    response = requests.get(url, timeout=10)
    img = Image.open(BytesIO(response.content)).convert('RGB')
    if size is not None:
        img = img.resize(size)
    return img


def initialize_pipeline(model_name="runwayml/stable-diffusion-v1-5", use_ddim=True):
    """Initialize the Stable Diffusion pipeline"""
    global _pipe
    if _pipe is None:
        _pipe = StableDiffusionPipeline.from_pretrained(model_name).to(device)
        if use_ddim:
            _pipe.scheduler = DDIMScheduler.from_config(_pipe.scheduler.config)
    return _pipe


def initialize_controlnet_pipeline(controlnet_type="canny"):
    """Initialize ControlNet pipeline"""
    global _controlnet_pipe

    if controlnet_type == "canny":
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-canny",
            torch_dtype=torch.float16
        )
    elif controlnet_type == "openpose":
        controlnet = ControlNetModel.from_pretrained(
            "fusing/stable-diffusion-v1-5-controlnet-openpose",
            torch_dtype=torch.float16
        )
    else:
        raise ValueError(f"Unsupported controlnet_type: {controlnet_type}")

    model_id = "runwayml/stable-diffusion-v1-5"
    _controlnet_pipe = StableDiffusionControlNetPipeline.from_pretrained(
        model_id,
        controlnet=controlnet,
        torch_dtype=torch.float16,
    )
    _controlnet_pipe.scheduler = UniPCMultistepScheduler.from_config(_controlnet_pipe.scheduler.config)
    _controlnet_pipe.enable_model_cpu_offload()

    try:
        _controlnet_pipe.enable_xformers_memory_efficient_attention()
    except:
        print("XFormers not available, skipping...")

    return _controlnet_pipe


def sample_image(prompt, negative_prompt='', guidance_scale=3.5, num_inference_steps=30):
    """Sample an image using the pipeline"""
    pipe = initialize_pipeline()
    result = pipe(prompt, negative_prompt=negative_prompt, guidance_scale=guidance_scale,
                  num_inference_steps=num_inference_steps)
    return result.images[0]


def plot_alpha_timesteps():
    """Plot alpha (alphas_cumprod) over timesteps"""
    pipe = initialize_pipeline()
    timesteps = pipe.scheduler.timesteps.cpu()
    alphas = pipe.scheduler.alphas_cumprod[timesteps]
    plt.plot(timesteps, alphas, label='alpha_t')
    plt.legend()
    plt.show()


@torch.no_grad()
def custom_sample(prompt, start_step=0, start_latents=None,
                  guidance_scale=3.5, num_inference_steps=30,
                  num_images_per_prompt=1, do_classifier_free_guidance=True,
                  negative_prompt='', device=device):
    """Custom sampling function with manual update step"""
    pipe = initialize_pipeline()

    # Encode prompt
    text_embeddings = pipe._encode_prompt(
        prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
    )

    # Set num inference steps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)

    # Create a random starting point if we don't have one already
    if start_latents is None:
        start_latents = torch.randn(1, 4, 64, 64, device=device)
        start_latents *= pipe.scheduler.init_noise_sigma

    latents = start_latents.clone()

    for i in tqdm(range(start_step, num_inference_steps)):
        t = pipe.scheduler.timesteps[i]

        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        # predict the noise residual
        noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # Manual update step
        prev_t = max(1, t.item() - (1000 // num_inference_steps))  # t-1
        alpha_t = pipe.scheduler.alphas_cumprod[t.item()]
        alpha_t_prev = pipe.scheduler.alphas_cumprod[prev_t]
        predicted_x0 = (latents - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
        direction_pointing_to_xt = (1 - alpha_t_prev).sqrt() * noise_pred
        latents = alpha_t_prev.sqrt() * predicted_x0 + direction_pointing_to_xt

    # Post-processing
    images = pipe.decode_latents(latents)
    images = pipe.numpy_to_pil(images)

    return images


def encode_image_to_latent(image, pipe=None):
    """Encode an image to latent space"""
    if pipe is None:
        pipe = initialize_pipeline()

    with torch.no_grad():
        latent = pipe.vae.encode(tfms.functional.to_tensor(image).unsqueeze(0).to(device) * 2 - 1)
    return 0.18215 * latent.latent_dist.sample()


@torch.no_grad()
def invert_latents(start_latents, prompt, guidance_scale=3.5, num_inference_steps=80,
                   num_images_per_prompt=1, do_classifier_free_guidance=True,
                   negative_prompt='', device=device):
    """Invert latents to find noise representation"""
    pipe = initialize_pipeline()

    # Encode prompt
    text_embeddings = pipe._encode_prompt(
        prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
    )

    # latents are now the specified start latents
    latents = start_latents.clone()

    # We'll keep a list of the inverted latents as the process goes on
    intermediate_latents = []

    # Set num inference steps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)

    # Reversed timesteps
    timesteps = reversed(pipe.scheduler.timesteps)

    for i in tqdm(range(1, num_inference_steps), total=num_inference_steps - 1):
        # We'll skip the final iteration
        if i >= num_inference_steps - 1:
            continue

        t = timesteps[i]

        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        # predict the noise residual
        noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        current_t = max(0, t.item() - (1000 // num_inference_steps))  # t
        next_t = t  # t+1
        alpha_t = pipe.scheduler.alphas_cumprod[current_t]
        alpha_t_next = pipe.scheduler.alphas_cumprod[next_t]

        # Inverted update step
        latents = (latents - (1 - alpha_t).sqrt() * noise_pred) * (alpha_t_next.sqrt() / alpha_t.sqrt()) + (
                    1 - alpha_t_next).sqrt() * noise_pred

        # Store
        intermediate_latents.append(latents)

    return torch.cat(intermediate_latents)


def decode_latents(latents, pipe=None):
    """Decode latents to image"""
    if pipe is None:
        pipe = initialize_pipeline()

    with torch.no_grad():
        im = pipe.decode_latents(latents.unsqueeze(0))
    return pipe.numpy_to_pil(im)[0]


def edit_image(input_image, input_image_prompt, edit_prompt, num_steps=100,
               start_step=30, guidance_scale=3.5):
    """Edit an image using inversion and sampling"""
    # Encode image to latent
    l = encode_image_to_latent(input_image)

    # Invert to get noise representation
    inverted_latents = invert_latents(l, input_image_prompt, num_inference_steps=num_steps)

    # Sample with new prompt starting from inverted latents
    final_im = custom_sample(
        edit_prompt,
        start_latents=inverted_latents[-(start_step + 1)][None],
        start_step=start_step,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale
    )[0]

    return final_im


def create_canny_image(image_path_or_url, low_threshold=100, high_threshold=200):
    """Create canny edge image from input"""
    if image_path_or_url.startswith('http'):
        image = load_image(image_path_or_url)
    else:
        image = Image.open(image_path_or_url)

    image_np = np.array(image)
    image_edges = cv2.Canny(image_np, low_threshold, high_threshold)
    image_edges = image_edges[:, :, None]
    image_edges = np.concatenate([image_edges, image_edges, image_edges], axis=2)
    return Image.fromarray(image_edges)


def create_pose_image(image_path_or_url):
    """Create pose estimation image from input"""
    if image_path_or_url.startswith('http'):
        image = load_image(image_path_or_url)
    else:
        image = Image.open(image_path_or_url)

    model = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
    return model(image)


def image_grid(imgs, rows, cols):
    """Create a grid of images"""
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def controlnet_generate(control_image, prompt, controlnet_type="canny",
                        negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
                        num_inference_steps=20, generator_seed=2):
    """Generate images using ControlNet"""
    pipe = initialize_controlnet_pipeline(controlnet_type)

    if isinstance(prompt, str):
        prompt = [prompt]

    generators = [torch.Generator(device="cpu").manual_seed(generator_seed + i)
                  for i in range(len(prompt))]

    output = pipe(
        prompt,
        control_image,
        negative_prompt=[negative_prompt] * len(prompt),
        num_inference_steps=num_inference_steps,
        generator=generators,
    )

    return output.images


def demo_basic_sampling():
    """Demo basic sampling functionality"""
    print("Running basic sampling demo...")
    prompt = 'Beautiful DSLR Photograph of a penguin on the beach, golden hour'
    negative_prompt = 'blurry, ugly, stock photo'
    im = sample_image(prompt, negative_prompt=negative_prompt)
    return im.resize((256, 256))


def demo_custom_sampling():
    """Demo custom sampling functionality"""
    print("Running custom sampling demo...")
    negative_prompt = 'blurry, ugly, stock photo'
    result = custom_sample('Watercolor painting of a beach sunset',
                           negative_prompt=negative_prompt,
                           num_inference_steps=50)
    return result[0].resize((256, 256))


def demo_inversion_editing():
    """Demo image inversion and editing"""
    print("Running inversion and editing demo...")

    # Load sample image
    input_image = load_image(
        'https://images.pexels.com/photos/8306128/pexels-photo-8306128.jpeg',
        size=(512, 512)
    )

    # Edit the image
    edited_image = edit_image(
        input_image,
        'A puppy on the grass',
        'an old grey dog on the grass',
        num_steps=50,
        start_step=10
    )

    return edited_image


def demo_controlnet_canny():
    """Demo ControlNet with Canny edges"""
    print("Running ControlNet Canny demo...")

    # Load and process image
    image_url = "https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png"
    canny_image = create_canny_image(image_url)

    # Generate images
    prompt = ", best quality, extremely detailed"
    prompts = [t + prompt for t in ["Sandra Oh", "Kim Kardashian", "rihanna", "taylor swift"]]

    generated_images = controlnet_generate(canny_image, prompts, controlnet_type="canny")

    return image_grid(generated_images, 2, 2)


def demo_controlnet_openpose():
    """Demo ControlNet with OpenPose"""
    print("Running ControlNet OpenPose demo...")

    # Load yoga pose images
    urls = ["yoga1.jpeg", "yoga2.jpeg", "yoga3.jpeg", "yoga4.jpeg"]
    image_urls = [
        "https://huggingface.co/datasets/YiYiXu/controlnet-testing/resolve/main/" + url
        for url in urls
    ]

    # Create pose images
    pose_images = [create_pose_image(url) for url in image_urls]

    # Generate images
    prompt = "super-hero character, best quality, extremely detailed"
    generated_images = controlnet_generate(pose_images, [prompt] * 4, controlnet_type="openpose")

    return image_grid(generated_images, 2, 2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Stable Diffusion Utilities")
    parser.add_argument("--demo", type=str, choices=[
        "basic_sampling",
        "custom_sampling",
        "inversion_editing",
        "controlnet_canny",
        "controlnet_openpose",
        "all"
    ], default="basic_sampling", help="Which demo to run")

    parser.add_argument("--show", action="store_true", help="Show the result image")
    parser.add_argument("--save", type=str, help="Save the result to file")

    args = parser.parse_args()

    # Initialize pipeline
    initialize_pipeline()

    # Run selected demo
    if args.demo == "basic_sampling" or args.demo == "all":
        result = demo_basic_sampling()
    elif args.demo == "custom_sampling":
        result = demo_custom_sampling()
    elif args.demo == "inversion_editing":
        result = demo_inversion_editing()
    elif args.demo == "controlnet_canny":
        result = demo_controlnet_canny()
    elif args.demo == "controlnet_openpose":
        result = demo_controlnet_openpose()

    # Show or save result
    if args.show:
        result.show()

    if args.save:
        result.save(args.save)
        print(f"Image saved to {args.save}")