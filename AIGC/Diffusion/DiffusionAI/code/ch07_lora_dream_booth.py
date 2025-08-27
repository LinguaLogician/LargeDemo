# -*- coding: utf-8 -*-
# @project: LargeDemo
# @filename: ch7_lora_dream_booth.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/8/26 9:04
# https://chat.deepseek.com/a/chat/s/dd857ce1-cb61-41c4-9f60-d818ae60eb84

import torch
import torchvision
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    DDPMScheduler,
    DiffusionPipeline,
    LoRACrossAttnProcessor,
)
from datasets import load_dataset
from matplotlib import pyplot as plt
from typing import Dict, List, Optional
import argparse

# ==================== 全局常量 ====================
REPO_ID = "lansinuote/diffusion.3.dream_booth"
CHECKPOINT = "runwayml/stable-diffusion-v1-5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==================== 数据加载与预处理 ====================
def load_and_preprocess_data(
        repo_id: str, checkpoint: str
) -> torch.utils.data.DataLoader:
    """加载数据集并进行预处理，返回 DataLoader"""
    dataset = load_dataset(path=repo_id, split="train")

    # 数据增强
    compose = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(
                512, interpolation=torchvision.transforms.InterpolationMode.BILINEAR
            ),
            torchvision.transforms.RandomCrop(512),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5], [0.5]),
        ]
    )

    # 文字编码
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, subfolder="tokenizer", use_fast=False
    )

    def transform_function(data):
        # 图像编码
        pixel_values = compose(data["image"]).unsqueeze(dim=0)

        # 文字编码
        tokens = tokenizer(
            data["text"],
            truncation=True,
            padding="max_length",
            max_length=77,
            return_tensors="pt",
        )

        return {
            "pixel_values": pixel_values,
            "input_ids": tokens.input_ids,
            "attention_mask": tokens.attention_mask,
        }

    dataset = dataset.with_transform(transform_function)

    # 创建 DataLoader
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, collate_fn=None
    )

    return loader


# ==================== 模型加载与配置 ====================
def load_models(checkpoint: str):
    """加载预训练模型并冻结参数"""
    encoder = CLIPTextModel.from_pretrained(checkpoint, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(checkpoint, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(checkpoint, subfolder="unet")

    # 冻结参数
    vae.requires_grad_(False)
    encoder.requires_grad_(False)
    unet.requires_grad_(False)

    return encoder, vae, unet


def setup_lora_layers(unet: UNet2DConditionModel):
    """为UNet设置LoRA注意力处理器"""
    processors = {}

    for name in unet.attn_processors.keys():
        cross_attention_dim = 768  # unet.config.cross_attention_dim
        if name.endswith("attn1.processor"):
            cross_attention_dim = None

        hidden_size = 1280  # unet.config.block_out_channels[-1]

        if name.startswith("up_blocks"):
            block_id = int(name[10])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]

        if name.startswith("down_blocks"):
            block_id = int(name[12])
            hidden_size = unet.config.block_out_channels[block_id]

        processors[name] = LoRACrossAttnProcessor(hidden_size, cross_attention_dim)

    unet.set_attn_processor(processors)
    return torch.nn.ModuleList(unet.attn_processors.values())


# ==================== 训练相关函数 ====================
def compute_loss(
        data: Dict[str, torch.Tensor],
        encoder: CLIPTextModel,
        vae: AutoencoderKL,
        unet: UNet2DConditionModel,
        scheduler: DDPMScheduler,
        criterion: torch.nn.Module,
) -> torch.Tensor:
    """计算扩散模型的损失"""
    # 编码文字
    out_encoder = encoder(
        input_ids=data["input_ids"], attention_mask=data["attention_mask"]
    )[0]

    # 计算特征图
    out_vae = vae.encode(data["pixel_values"]).latent_dist.sample().detach()
    out_vae = out_vae * 0.18215  # vae.config.scaling_factor

    # 随机噪声
    noise = torch.randn_like(out_vae)

    # 随机噪声步
    noise_step = torch.randint(0, 1000, (1,), device=out_vae.device).long()

    # 添加噪声
    out_vae_noise = scheduler.add_noise(out_vae, noise, noise_step)

    # 从噪声图中预测噪声
    out_unet = unet(out_vae_noise, noise_step, out_encoder).sample

    return criterion(out_unet, noise)


def train_model(
        loader: torch.utils.data.DataLoader,
        encoder: CLIPTextModel,
        vae: AutoencoderKL,
        unet: UNet2DConditionModel,
        lora_layers: torch.nn.ModuleList,
        save_path: str = "./save",
        num_epochs: int = 200,
):
    """训练LoRA模型"""
    # 初始化工具
    scheduler = DDPMScheduler.from_pretrained(CHECKPOINT, subfolder="scheduler")
    optimizer = torch.optim.AdamW(
        lora_layers.parameters(),
        lr=1e-4,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        eps=1e-8,
    )
    criterion = torch.nn.MSELoss()

    # 移动到设备
    unet.to(DEVICE)
    vae.to(DEVICE)
    encoder.to(DEVICE)
    unet.train()

    loss_sum = 0
    for epoch in range(num_epochs):
        for data in loader:
            # 移动数据到设备
            data = {k: v.to(DEVICE) for k, v in data.items()}

            # 计算损失
            loss = compute_loss(data, encoder, vae, unet, scheduler, criterion)

            # 反向传播和优化
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_layers.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            loss_sum += loss.item()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss_sum}")
            loss_sum = 0

    # 保存训练结果
    unet.save_attn_procs(save_path)
    print(f"Training completed. Model saved to {save_path}")


# ==================== 测试相关函数 ====================
def test_model(use_trained_lora: bool = False, lora_path: str = "./save"):
    """测试模型生成效果"""
    pipeline = DiffusionPipeline.from_pretrained(CHECKPOINT, safety_checker=None)

    if use_trained_lora:
        pipeline.unet.load_attn_procs(lora_path)
        print("Using trained LoRA weights")
    else:
        print("Using original model")

    pipeline = pipeline.to(DEVICE)

    texts = [
        "A photo of little dog in a bucket",
        "A photo of little dog swimming",
        "A photo of little dog sleeping",
        "A photo of little dog in a doghouse",
    ]

    images = []
    for text in texts:
        with torch.autocast(DEVICE):
            result = pipeline(text)
            images.append(result.images[0])

    # 显示结果
    plt.figure(figsize=(10, 5))
    for i in range(4):
        plt.subplot(1, 4, i + 1)
        plt.imshow(images[i])
        plt.axis("off")
        plt.title(texts[i][:20] + "..." if len(texts[i]) > 20 else texts[i])

    plt.tight_layout()
    plt.show()


# ==================== 主程序 ====================
def main():
    parser = argparse.ArgumentParser(description="DreamBooth with LoRA Training")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "test_original", "test_trained"],
        required=True,
        help="Run mode: train, test_original, or test_trained",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default="./save",
        help="Path to save/load LoRA weights",
    )
    parser.add_argument(
        "--epochs", type=int, default=200, help="Number of training epochs"
    )

    args = parser.parse_args()

    if args.mode == "train":
        print("Loading data...")
        loader = load_and_preprocess_data(REPO_ID, CHECKPOINT)

        print("Loading models...")
        encoder, vae, unet = load_models(CHECKPOINT)

        print("Setting up LoRA layers...")
        lora_layers = setup_lora_layers(unet)

        print("Starting training...")
        train_model(loader, encoder, vae, unet, lora_layers, args.lora_path, args.epochs)

    elif args.mode == "test_original":
        print("Testing original model...")
        test_model(use_trained_lora=False)

    elif args.mode == "test_trained":
        print("Testing trained model...")
        test_model(use_trained_lora=True, lora_path=args.lora_path)


if __name__ == "__main__":
    main()