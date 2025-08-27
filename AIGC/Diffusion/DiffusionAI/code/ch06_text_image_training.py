# -*- coding: utf-8 -*-
# @project: LargeDemo
# @filename: ch6_text_image_training.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/8/26 8:50
# https://chat.deepseek.com/a/chat/s/da103a55-3e45-48f3-9ba0-38c63d71f42e

# text_to_image_training.py

import torch
import torchvision
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    DDPMScheduler,
    StableDiffusionPipeline,
    DiffusionPipeline,
)
from matplotlib import pyplot as plt
import argparse

# 全局常量
REPO_ID = "lansinuote/diffusion.4.text_to_image.book"
CHECKPOINT = "CompVis/stable-diffusion-v1-4"


def load_and_preprocess_dataset(dataset_name="m1guelpf/nouns", num_samples=1000):
    """加载并预处理数据集"""
    print("Loading dataset...")
    dataset = load_dataset(dataset_name, split="train")
    dataset = dataset.shuffle(seed=0).select(range(num_samples))
    return dataset


def get_transform():
    """定义图像预处理变换"""
    return torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(512, interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
            torchvision.transforms.CenterCrop(512),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5], [0.5]),
        ]
    )


def tokenize_text(tokenizer, texts, max_length=77):
    """对文本进行编码"""
    return tokenizer.batch_encode_plus(
        texts,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).input_ids


def setup_models(checkpoint=CHECKPOINT, freeze_vae_and_encoder=True):
    """加载并设置模型"""
    print("Loading models...")
    encoder = CLIPTextModel.from_pretrained(checkpoint, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(checkpoint, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(checkpoint, subfolder="unet")

    if freeze_vae_and_encoder:
        vae.requires_grad_(False)
        encoder.requires_grad_(False)

    return encoder, vae, unet


def print_model_sizes(encoder, vae, unet):
    """打印模型参数量"""
    print(f"Encoder: {sum(p.numel() for p in encoder.parameters()) / 1e4:.4f}万参数")
    print(f"VAE: {sum(p.numel() for p in vae.parameters()) / 1e4:.4f}万参数")
    print(f"UNet: {sum(p.numel() for p in unet.parameters()) / 1e4:.4f}万参数")


def setup_training_tools():
    """设置训练工具（调度器、优化器、损失函数）"""
    scheduler = DDPMScheduler.from_pretrained(CHECKPOINT, subfolder="scheduler")
    optimizer = torch.optim.AdamW(
        unet.parameters(), lr=1e-5, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-8
    )
    criterion = torch.nn.MSELoss()
    return scheduler, optimizer, criterion


def compute_loss(data, encoder, vae, unet, scheduler, criterion, device="cuda"):
    """计算损失函数"""
    # 文字编码
    out_encoder = encoder(data["input_ids"])[0]

    # 抽取图像特征图
    out_vae = vae.encode(data["pixel_values"]).latent_dist.sample()
    out_vae = out_vae * 0.18215  # vae.config.scaling_factor

    # 随机噪声
    noise = torch.randn_like(out_vae)

    # 添加噪声
    noise_step = torch.randint(0, scheduler.num_train_timesteps, (1,)).long().to(device)
    out_vae_noise = scheduler.add_noise(out_vae, noise, noise_step)

    # 预测噪声
    out_unet = unet(out_vae_noise, noise_step, out_encoder).sample

    # 计算MSE损失
    return criterion(out_unet, noise)


def train_model(
    encoder, vae, unet, train_loader, scheduler, optimizer, criterion, num_epochs=150, device="cuda"
):
    """训练模型"""
    encoder.to(device)
    vae.to(device)
    unet.to(device)
    unet.train()

    loss_sum = 0
    for epoch in range(num_epochs):
        for i, data in enumerate(train_loader):
            # 移动数据到设备
            data = {k: v.to(device) for k, v in data.items()}

            # 计算损失并反向传播
            loss = compute_loss(data, encoder, vae, unet, scheduler, criterion, device) / 4
            loss.backward()
            loss_sum += loss.item()

            # 每4个batch更新一次参数
            if (i + 1) % 4 == 0:
                torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

        # 每10个epoch打印一次损失
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss_sum:.4f}")
            loss_sum = 0

    # 保存模型
    pipeline = StableDiffusionPipeline.from_pretrained(
        CHECKPOINT, text_encoder=encoder, vae=vae, unet=unet
    )
    pipeline.save_pretrained("./save")
    print("Model saved to ./save")


def test_model(pipeline, device="cuda"):
    """测试模型生成效果"""
    pipeline = pipeline.to(device)

    texts = [
        "a pixel art character with square orange glasses, a whale-shaped head and a teal-colored body on a warm background",
        "a pixel art character with square green and blue glasses, a hanger-shaped head and a red-colored body on a warm background",
        "a pixel art character with square black sunglasses, a wallsafe-shaped head and a purple-colored body on a cool background",
        "a pixel art character with square green and blue glasses, a mushroom-shaped head and a bluegrey-colored body on a cool background",
        "a pixel art character with square pink and red glasses, a chipboard-shaped head and a blue-colored body on a cool background",
        "a pixel art character with square pink and red glasses, a spaghetti-shaped head and a darkpink-colored body on a cool background",
    ]

    images = [pipeline(text).images[0] for text in texts]

    plt.figure(figsize=(20, 10))
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.imshow(images[i])
        plt.axis("off")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Stable Diffusion Text-to-Image Training")
    parser.add_argument("--mode", type=str, choices=["train", "test", "test_pretrained", "test_online"],
                        default="test", help="运行模式")
    parser.add_argument("--checkpoint_path", type=str, default="./save", help="模型检查点路径")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.mode == "train":
        # 训练模式
        dataset = load_and_preprocess_dataset()
        transform = get_transform()
        tokenizer = CLIPTokenizer.from_pretrained(CHECKPOINT, subfolder="tokenizer")

        def collate_fn(data):
            pixel_values = transform(data["image"]).unsqueeze(0)
            input_ids = tokenize_text(tokenizer, [data["text"]])
            return {"pixel_values": pixel_values, "input_ids": input_ids}

        dataset.set_transform(collate_fn)
        train_loader = DataLoader(dataset, shuffle=True, batch_size=1)

        encoder, vae, unet = setup_models()
        print_model_sizes(encoder, vae, unet)

        scheduler, optimizer, criterion = setup_training_tools()
        train_model(encoder, vae, unet, train_loader, scheduler, optimizer, criterion)

    elif args.mode == "test":
        # 测试训练后的模型
        pipeline = DiffusionPipeline.from_pretrained(args.checkpoint_path, safety_checker=None)
        test_model(pipeline, device)

    elif args.mode == "test_pretrained":
        # 测试原始预训练模型
        pipeline = DiffusionPipeline.from_pretrained(CHECKPOINT, safety_checker=None)
        test_model(pipeline, device)

    elif args.mode == "test_online":
        # 测试在线模型
        pipeline = DiffusionPipeline.from_pretrained(REPO_ID, safety_checker=None)
        test_model(pipeline, device)


if __name__ == "__main__":
    main()
