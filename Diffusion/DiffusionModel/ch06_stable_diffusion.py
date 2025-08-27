# -*- coding: utf-8 -*-
# @project: LargeDemo
# @filename: chapter6_stable_diffusion.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/8/25 15:00
# https://chat.deepseek.com/a/chat/s/d1a19b8f-2f0b-4578-b917-c4c6fa1936dc

# stable_diffusion_demo.py

import torch
import requests
from PIL import Image
from io import BytesIO
from matplotlib import pyplot as plt
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionDepth2ImgPipeline,
    LMSDiscreteScheduler
)


def setup_environment():
    """安装必要的依赖包"""
    # 注意：在实际运行前，请确保已安装这些包
    # !pip install -Uq diffusers ftfy accelerate
    # !pip install -Uq git+https://github.com/huggingface/transformers
    pass


def get_device():
    """设置设备"""
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")
    return device


def download_image(url):
    """下载图像"""
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGB")


def load_base_pipeline(model_id="stabilityai/stable-diffusion-2-1-base", device=None):
    """加载基础Stable Diffusion管线"""
    if device is None:
        device = get_device()

    pipe = StableDiffusionPipeline.from_pretrained(model_id).to(device)
    return pipe


def run_text_to_image(pipe, prompt, negative_prompt=None, height=480, width=640,
                      guidance_scale=8, num_inference_steps=35, seed=42):
    """运行文本到图像生成"""
    generator = torch.Generator(device=pipe.device).manual_seed(seed)

    pipe_output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator
    )

    return pipe_output.images[0]


def compare_guidance_scales(pipe, prompt, cfg_scales=[1.1, 8, 12], image_size=480):
    """比较不同guidance scale的效果"""
    fig, axs = plt.subplots(1, len(cfg_scales), figsize=(16, 5))

    for i, ax in enumerate(axs):
        im = pipe(
            prompt,
            height=image_size,
            width=image_size,
            guidance_scale=cfg_scales[i],
            num_inference_steps=35,
            generator=torch.Generator(device=pipe.device).manual_seed(42)
        ).images[0]

        ax.imshow(im)
        ax.set_title(f'CFG Scale {cfg_scales[i]}')

    plt.show()


def demonstrate_pipeline_components(pipe):
    """演示管线的各个组件"""
    print("Pipeline components:", list(pipe.components.keys()))

    # 演示VAE编码解码
    images = torch.rand(1, 3, 512, 512).to(pipe.device) * 2 - 1
    print("Input images shape:", images.shape)

    with torch.no_grad():
        latents = 0.18215 * pipe.vae.encode(images).latent_dist.mean
    print("Encoded latents shape:", latents.shape)

    with torch.no_grad():
        decoded_images = pipe.vae.decode(latents / 0.18215).sample
    print("Decoded images shape:", decoded_images.shape)

    # 演示文本编码
    input_ids = pipe.tokenizer(["A painting of a flooble"])['input_ids']
    print("Input ID -> decoded token")
    for input_id in input_ids[0]:
        print(f"{input_id} -> {pipe.tokenizer.decode(input_id)}")

    input_ids = torch.tensor(input_ids).to(pipe.device)
    with torch.no_grad():
        text_embeddings = pipe.text_encoder(input_ids)['last_hidden_state']
    print("Text embeddings shape:", text_embeddings.shape)

    # 使用管线的_encode_prompt方法
    text_embeddings = pipe._encode_prompt("A painting of a flooble", pipe.device, 1, False, '')
    print("Encoded prompt shape:", text_embeddings.shape)

    # 演示UNet
    timestep = pipe.scheduler.timesteps[0]
    latents = torch.randn(1, 4, 64, 64).to(pipe.device)
    text_embeddings = torch.randn(1, 77, 1024).to(pipe.device)

    with torch.no_grad():
        unet_output = pipe.unet(latents, timestep, text_embeddings).sample
    print('UNet output shape:', unet_output.shape)

    # 绘制噪声调度
    plt.plot(pipe.scheduler.alphas_cumprod, label=r'$\bar{\alpha}$')
    plt.xlabel('Timestep (high noise to low noise ->)')
    plt.title('Noise schedule')
    plt.legend()
    plt.show()


def change_scheduler(pipe):
    """更换调度器"""
    pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
    print('Scheduler config:', pipe.scheduler)

    # 使用新调度器生成图像
    result = pipe(
        prompt="Palette knife painting of an winter cityscape",
        height=480,
        width=480,
        generator=torch.Generator(device=pipe.device).manual_seed(42)
    )

    return result.images[0]


def manual_sampling_process(pipe, prompt, negative_prompt, guidance_scale=8, num_inference_steps=30):
    """手动实现采样过程"""
    generator = torch.Generator(device=pipe.device).manual_seed(42)

    # 编码提示
    text_embeddings = pipe._encode_prompt(prompt, pipe.device, 1, True, negative_prompt)

    # 创建随机起点
    latents = torch.randn((1, 4, 64, 64), device=pipe.device, generator=generator)
    latents *= pipe.scheduler.init_noise_sigma

    # 准备调度器
    pipe.scheduler.set_timesteps(num_inference_steps, device=pipe.device)

    # 循环采样时间步
    for i, t in enumerate(pipe.scheduler.timesteps):
        # 扩展潜在变量用于分类器自由引导
        latent_model_input = torch.cat([latents] * 2)

        # 应用调度器所需的缩放
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        # 使用UNet预测噪声残差
        with torch.no_grad():
            noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # 执行引导
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # 计算前一个噪声样本 x_t -> x_t-1
        latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

    # 解码潜在变量为图像
    with torch.no_grad():
        image = pipe.decode_latents(latents.detach())

    return pipe.numpy_to_pil(image)[0]


def run_img2img(init_image, prompt="An oil painting of a man on a bench", strength=0.6):
    """运行图像到图像转换"""
    device = get_device()
    model_id = "stabilityai/stable-diffusion-2-1-base"
    img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id).to(device)

    result_image = img2img_pipe(
        prompt=prompt,
        image=init_image,
        strength=strength,
    ).images[0]

    # 显示结果
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].imshow(init_image)
    axs[0].set_title('Input Image')
    axs[1].imshow(result_image)
    axs[1].set_title('Result')
    plt.show()

    return result_image


def run_inpainting(init_image, mask_image, prompt="A small robot, high resolution, sitting on a park bench"):
    """运行图像修复"""
    device = get_device()
    pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting")
    pipe = pipe.to(device)

    image = pipe(prompt=prompt, image=init_image, mask_image=mask_image).images[0]

    # 显示结果
    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    axs[0].imshow(init_image)
    axs[0].set_title('Input Image')
    axs[1].imshow(mask_image)
    axs[1].set_title('Mask')
    axs[2].imshow(image)
    axs[2].set_title('Result')
    plt.show()

    return image


def run_depth2img():
    """运行深度到图像转换（需要合适的模型）"""
    device = get_device()
    pipe = StableDiffusionDepth2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-2-depth")
    pipe = pipe.to(device)
    print("Depth2Img pipeline loaded")
    return pipe


def main():
    """主函数，演示各种功能"""
    device = get_device()

    # 下载示例图像
    img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
    mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"

    init_image = download_image(img_url).resize((512, 512))
    mask_image = download_image(mask_url).resize((512, 512))

    # 加载基础管线
    pipe = load_base_pipeline(device=device)

    # 演示文本到图像生成
    print("=== Text to Image Generation ===")
    result = run_text_to_image(
        pipe,
        "Palette knife painting of an autumn cityscape",
        negative_prompt="Oversaturated, blurry, low quality"
    )
    result.show()

    # 比较不同guidance scale
    print("=== Comparing Guidance Scales ===")
    compare_guidance_scales(pipe, "A collie with a pink hat")

    # 演示管线组件
    print("=== Pipeline Components ===")
    demonstrate_pipeline_components(pipe)

    # 更换调度器
    print("=== Changing Scheduler ===")
    result = change_scheduler(pipe)
    result.show()

    # 手动采样过程
    print("=== Manual Sampling Process ===")
    result = manual_sampling_process(
        pipe,
        "Beautiful picture of a wave breaking",
        "zoomed in, blurry, oversaturated, warped"
    )
    result.show()

    # 图像到图像转换
    print("=== Image to Image ===")
    run_img2img(init_image)

    # 图像修复
    print("=== Inpainting ===")
    run_inpainting(init_image, mask_image)

    # 深度到图像（仅加载，需要进一步使用）
    print("=== Depth to Image ===")
    depth_pipe = run_depth2img()


if __name__ == "__main__":
    # 可以选择运行特定的功能，而不是全部运行
    import argparse

    parser = argparse.ArgumentParser(description='Stable Diffusion Demo')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['all', 'text2img', 'guidance', 'components',
                                 'scheduler', 'manual', 'img2img', 'inpainting', 'depth2img'],
                        help='运行模式')

    args = parser.parse_args()

    if args.mode == 'all':
        main()
    else:
        device = get_device()

        if args.mode == 'text2img':
            pipe = load_base_pipeline(device=device)
            result = run_text_to_image(
                pipe,
                "Palette knife painting of an autumn cityscape",
                negative_prompt="Oversaturated, blurry, low quality"
            )
            result.show()

        elif args.mode == 'guidance':
            pipe = load_base_pipeline(device=device)
            compare_guidance_scales(pipe, "A collie with a pink hat")

        elif args.mode == 'components':
            pipe = load_base_pipeline(device=device)
            demonstrate_pipeline_components(pipe)

        elif args.mode == 'scheduler':
            pipe = load_base_pipeline(device=device)
            result = change_scheduler(pipe)
            result.show()

        elif args.mode == 'manual':
            pipe = load_base_pipeline(device=device)
            result = manual_sampling_process(
                pipe,
                "Beautiful picture of a wave breaking",
                "zoomed in, blurry, oversaturated, warped"
            )
            result.show()

        elif args.mode == 'img2img':
            img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
            init_image = download_image(img_url).resize((512, 512))
            run_img2img(init_image)

        elif args.mode == 'inpainting':
            img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
            mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"
            init_image = download_image(img_url).resize((512, 512))
            mask_image = download_image(mask_url).resize((512, 512))
            run_inpainting(init_image, mask_image)

        elif args.mode == 'depth2img':
            run_depth2img()
