# -*- coding: utf-8 -*-
# @project: LargeDemo
# @filename: chapter4_diffusers.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/8/25 14:58
# https://chat.deepseek.com/a/chat/s/78e6febf-0fc7-42fe-a967-b5519235553b

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
from huggingface_hub import notebook_login, get_full_repo_name, HfApi, create_repo, ModelCard
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import os


def setup_environment():
    """安装必要的依赖包"""
    # 实际使用时取消注释
    # %pip install -qq -U diffusers datasets transformers accelerate ftfy pyarrow==9.0.0
    # !sudo apt -qq install git-lfs
    # !git config --global credential.helper store
    pass


def login_to_huggingface():
    """登录Hugging Face Hub"""
    notebook_login()


def get_device():
    """获取可用设备"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def show_images(x):
    """Given a batch of images x, make a grid and convert to PIL"""
    x = x * 0.5 + 0.5  # Map from (-1, 1) back to (0, 1)
    grid = torchvision.utils.make_grid(x)
    grid_im = grid.detach().cpu().permute(1, 2, 0).clip(0, 1) * 255
    grid_im = Image.fromarray(np.array(grid_im).astype(np.uint8))
    return grid_im


def make_grid(images, size=64):
    """Given a list of PIL images, stack them together into a line for easy viewing"""
    output_im = Image.new("RGB", (size * len(images), size))
    for i, im in enumerate(images):
        output_im.paste(im.resize((size, size)), (i * size, 0))
    return output_im


def load_pretrained_model(model_id="sd-dreambooth-library/mr-potato-head", device=None):
    """加载预训练的Stable Diffusion模型"""
    if device is None:
        device = get_device()

    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
    return pipe


def generate_image_with_prompt(pipe, prompt="an abstract oil painting of sks mr potato head by picasso"):
    """使用提示词生成图像"""
    image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
    return image


def load_butterfly_pipeline(device=None):
    """加载蝴蝶图像的预训练管道"""
    if device is None:
        device = get_device()

    butterfly_pipeline = DDPMPipeline.from_pretrained("johnowhitaker/ddpm-butterflies-32px").to(device)
    return butterfly_pipeline


def generate_butterflies(butterfly_pipeline, batch_size=8):
    """生成蝴蝶图像"""
    images = butterfly_pipeline(batch_size=batch_size).images
    return make_grid(images)


def prepare_dataset(dataset_name="huggan/smithsonian_butterflies_subset", image_size=32, batch_size=64):
    """准备训练数据集"""
    dataset = load_dataset(dataset_name, split="train")

    # 定义数据增强
    preprocess = transforms.Compose([
        transforms.Resize((image_size, image_size)),  # Resize
        transforms.RandomHorizontalFlip(),  # Randomly flip (data augmentation)
        transforms.ToTensor(),  # Convert to tensor (0, 1)
        transforms.Normalize([0.5], [0.5]),  # Map to (-1, 1)
    ])

    def transform(examples):
        images = [preprocess(image.convert("RGB")) for image in examples["image"]]
        return {"images": images}

    dataset.set_transform(transform)

    # 创建数据加载器
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataset, train_dataloader


def visualize_batch(train_dataloader, device=None):
    """可视化数据批次"""
    if device is None:
        device = get_device()

    xb = next(iter(train_dataloader))["images"].to(device)[:8]
    print("X shape:", xb.shape)
    return show_images(xb).resize((8 * 64, 64), resample=Image.NEAREST)


def setup_noise_scheduler(scheduler_type="default"):
    """设置噪声调度器"""
    if scheduler_type == "default":
        noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    elif scheduler_type == "cosine":
        noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2')
    elif scheduler_type == "low_noise":
        noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_start=0.001, beta_end=0.004)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    return noise_scheduler


def visualize_noise_schedule(noise_scheduler):
    """可视化噪声调度计划"""
    plt.plot(noise_scheduler.alphas_cumprod.cpu() ** 0.5, label=r"${\sqrt{\bar{\alpha}_t}}$")
    plt.plot((1 - noise_scheduler.alphas_cumprod.cpu()) ** 0.5, label=r"$\sqrt{(1 - \bar{\alpha}_t)}$")
    plt.legend(fontsize="x-large")
    plt.show()


def add_noise_to_batch(xb, noise_scheduler, device=None):
    """向批次添加噪声"""
    if device is None:
        device = get_device()

    timesteps = torch.linspace(0, 999, 8).long().to(device)
    noise = torch.randn_like(xb)
    noisy_xb = noise_scheduler.add_noise(xb, noise, timesteps)
    print("Noisy X shape", noisy_xb.shape)
    return show_images(noisy_xb).resize((8 * 64, 64), resample=Image.NEAREST)


def create_unet_model(image_size=32, device=None):
    """创建UNet模型"""
    if device is None:
        device = get_device()

    model = UNet2DModel(
        sample_size=image_size,  # the target image resolution
        in_channels=3,  # the number of input channels, 3 for RGB images
        out_channels=3,  # the number of output channels
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        block_out_channels=(64, 128, 128, 256),  # More channels -> more parameters
        down_block_types=(
            "DownBlock2D",  # a regular ResNet downsampling block
            "DownBlock2D",
            "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
            "AttnDownBlock2D",
        ),
        up_block_types=(
            "AttnUpBlock2D",
            "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
            "UpBlock2D",
            "UpBlock2D",  # a regular ResNet upsampling block
        ),
    )
    model.to(device)
    return model


def test_model_prediction(model, noisy_xb, timesteps):
    """测试模型预测"""
    with torch.no_grad():
        model_prediction = model(noisy_xb, timesteps).sample
    print("Model prediction shape:", model_prediction.shape)
    return model_prediction


def train_model(model, train_dataloader, noise_scheduler, num_epochs=30, learning_rate=4e-4, device=None):
    """训练模型"""
    if device is None:
        device = get_device()

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    losses = []

    for epoch in range(num_epochs):
        for step, batch in enumerate(train_dataloader):
            clean_images = batch["images"].to(device)
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.num_train_timesteps, (bs,), device=clean_images.device
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            # Get the model prediction
            noise_pred = model(noisy_images, timesteps, return_dict=False)[0]

            # Calculate the loss
            loss = F.mse_loss(noise_pred, noise)
            loss.backward()
            losses.append(loss.item())

            # Update the model parameters with the optimizer
            optimizer.step()
            optimizer.zero_grad()

        if (epoch + 1) % 5 == 0:
            loss_last_epoch = sum(losses[-len(train_dataloader):]) / len(train_dataloader)
            print(f"Epoch:{epoch + 1}, loss: {loss_last_epoch}")

    return losses


def plot_training_loss(losses):
    """绘制训练损失"""
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].plot(losses)
    axs[1].plot(np.log(losses))
    plt.show()


def create_pipeline(model, noise_scheduler):
    """创建生成管道"""
    return DDPMPipeline(unet=model, scheduler=noise_scheduler)


def generate_with_pipeline(image_pipe):
    """使用管道生成图像"""
    pipeline_output = image_pipe()
    return pipeline_output.images[0]


def save_pipeline(image_pipe, save_path="my_pipeline"):
    """保存管道"""
    image_pipe.save_pretrained(save_path)
    return save_path


def manual_sampling_loop(model, noise_scheduler, num_samples=8, image_size=32, device=None):
    """手动采样循环"""
    if device is None:
        device = get_device()

    # Random starting point
    sample = torch.randn(num_samples, 3, image_size, image_size).to(device)

    for i, t in enumerate(noise_scheduler.timesteps):
        # Get model pred
        with torch.no_grad():
            residual = model(sample, t).sample

        # Update sample with step
        sample = noise_scheduler.step(residual, t, sample).prev_sample

    return show_images(sample)


def upload_to_hub(model_name, pipeline_path="my_pipeline"):
    """上传模型到Hugging Face Hub"""
    hub_model_id = get_full_repo_name(model_name)

    # 创建仓库
    create_repo(hub_model_id)
    api = HfApi()

    # 上传文件
    api.upload_folder(
        folder_path=f"{pipeline_path}/scheduler", path_in_repo="", repo_id=hub_model_id
    )
    api.upload_folder(
        folder_path=f"{pipeline_path}/unet", path_in_repo="", repo_id=hub_model_id
    )
    api.upload_file(
        path_or_fileobj=f"{pipeline_path}/model_index.json",
        path_in_repo="model_index.json",
        repo_id=hub_model_id,
    )

    # 创建模型卡片
    content = f"""
---
license: mit
tags:
- pytorch
- diffusers
- unconditional-image-generation
- diffusion-models-class
---

# Model Card for Unit 1 of the [Diffusion Models Class 🧨](https://github.com/huggingface/diffusion-models-class)

This model is a diffusion model for unconditional image generation of cute 🦋.

## Usage
from diffusers import DDPMPipeline

pipeline = DDPMPipeline.from_pretrained('{hub_model_id}')
image = pipeline().images[0]
image
"""

    card = ModelCard(content)
    card.push_to_hub(hub_model_id)

    return hub_model_id


def load_from_hub(hub_model_id):
    """从Hub加载模型"""
    image_pipe = DDPMPipeline.from_pretrained(hub_model_id)
    pipeline_output = image_pipe()
    return pipeline_output.images[0]


def train_with_script(model_name="sd-class-butterflies-64", dataset_name="huggan/smithsonian_butterflies_subset",
                      resolution=64, train_batch_size=32, num_epochs=50, learning_rate=1e-4):
    """使用训练脚本训练模型"""
    # 这里应该是调用外部训练脚本的逻辑
    # 实际使用时取消注释
    """
    !accelerate launch train_unconditional.py \\
        --dataset_name="{dataset_name}" \\
        --resolution={resolution} \\
        --output_dir={model_name} \\
        --train_batch_size={train_batch_size} \\
        --num_epochs={num_epochs} \\
        --gradient_accumulation_steps=1 \\
        --learning_rate={learning_rate} \\
        --lr_warmup_steps=500 \\
        --mixed_precision="no"
    """
    print(f"Training model {model_name} with resolution {resolution}")
    return model_name


def main():
    """主函数，用于演示各个功能模块"""
    import argparse

    parser = argparse.ArgumentParser(description="Diffusion Model Training and Inference")
    parser.add_argument("--mode", choices=["setup", "login", "demo", "train", "generate", "upload"],
                        default="demo", help="运行模式")
    parser.add_argument("--model_id", default="sd-dreambooth-library/mr-potato-head",
                        help="预训练模型ID")
    parser.add_argument("--prompt", default="an abstract oil painting of sks mr potato head by picasso",
                        help="生成提示词")
    parser.add_argument("--image_size", type=int, default=32, help="图像大小")
    parser.add_argument("--batch_size", type=int, default=64, help="批次大小")
    parser.add_argument("--epochs", type=int, default=30, help="训练轮数")
    parser.add_argument("--model_name", default="sd-class-butterflies-32", help="模型名称")

    args = parser.parse_args()
    device = get_device()

    if args.mode == "setup":
        setup_environment()

    elif args.mode == "login":
        login_to_huggingface()

    elif args.mode == "demo":
        # 演示预训练模型
        print("Loading pretrained model...")
        pipe = load_pretrained_model(args.model_id, device)

        print("Generating image with prompt...")
        image = generate_image_with_prompt(pipe, args.prompt)
        image.show()

        print("Loading butterfly pipeline...")
        butterfly_pipe = load_butterfly_pipeline(device)
        butterfly_grid = generate_butterflies(butterfly_pipe)
        butterfly_grid.show()

    elif args.mode == "train":
        # 训练新模型
        print("Preparing dataset...")
        dataset, train_dataloader = prepare_dataset(image_size=args.image_size, batch_size=args.batch_size)

        print("Setting up noise scheduler...")
        noise_scheduler = setup_noise_scheduler("cosine")

        print("Creating UNet model...")
        model = create_unet_model(args.image_size, device)

        print("Training model...")
        losses = train_model(model, train_dataloader, noise_scheduler, num_epochs=args.epochs)

        print("Plotting training loss...")
        plot_training_loss(losses)

        print("Creating pipeline...")
        image_pipe = create_pipeline(model, noise_scheduler)

        print("Saving pipeline...")
        save_path = save_pipeline(image_pipe, f"{args.model_name}_pipeline")
        print(f"Pipeline saved to {save_path}")

    elif args.mode == "generate":
        # 生成图像
        print("Loading pipeline...")
        image_pipe = DDPMPipeline.from_pretrained(f"{args.model_name}_pipeline").to(device)

        print("Generating images...")
        image = generate_with_pipeline(image_pipe)
        image.show()

    elif args.mode == "upload":
        # 上传到Hub
        print("Uploading to Hugging Face Hub...")
        hub_id = upload_to_hub(args.model_name, f"{args.model_name}_pipeline")
        print(f"Model uploaded to: {hub_id}")

        print("Testing loaded model...")
        test_image = load_from_hub(hub_id)
        test_image.show()


if __name__ == "__main__":
    main()
