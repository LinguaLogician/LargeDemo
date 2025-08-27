# -*- coding: utf-8 -*-
# @project: LargeDemo
# @filename: ch5_dream_booth.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/8/26 8:31
# https://chat.deepseek.com/a/chat/s/dd81bfe2-6c70-49d1-a3e8-f0fe24c0bed6

import torch
import torchvision
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    DDPMScheduler,
    DiffusionPipeline,
)
from matplotlib import pyplot as plt
from PIL import Image
import argparse

# 全局常量
REPO_ID = "lansinuote/diffusion.3.dream_booth"
CHECKPOINT = "CompVis/stable-diffusion-v1-4"


def load_local_dataset():
    """加载本地数据集"""
    images = [
        {
            "image": Image.open("images/%d.jpeg" % i),
            "text": "a photo of little dog",
        }
        for i in range(5)
    ]
    return Dataset.from_list(images)


def load_online_dataset():
    """加载在线数据集"""
    dataset = load_dataset(path=REPO_ID, split="train")
    return dataset, dataset[0]


def preprocess_dataset(dataset):
    """数据集预处理"""
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
        CHECKPOINT, subfolder="tokenizer", use_fast=False
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

    return dataset.with_transform(transform_function)


def create_data_loader(dataset, batch_size=1, shuffle=True):
    """创建数据加载器"""
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=None
    )


def load_models():
    """加载模型"""
    encoder = CLIPTextModel.from_pretrained(CHECKPOINT, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(CHECKPOINT, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(CHECKPOINT, subfolder="unet")

    # 冻结参数
    vae.requires_grad_(False)
    encoder.requires_grad_(False)

    return encoder, vae, unet


def print_model_size(name, model):
    """打印模型大小"""
    print(name, sum(i.numel() for i in model.parameters()) / 10000)


def setup_training_tools():
    """初始化训练工具"""
    scheduler = DDPMScheduler.from_pretrained(CHECKPOINT, subfolder="scheduler")
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=5e-6,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        eps=1e-8,
    )
    criterion = torch.nn.MSELoss()
    return scheduler, optimizer, criterion


def get_loss(data, encoder, vae, unet, scheduler, criterion):
    """计算损失"""
    # 编码文字 [1, 77] -> [1, 77, 768]
    out_encoder = encoder(
        input_ids=data["input_ids"], attention_mask=data["attention_mask"]
    )[0]

    # 计算特征图 [1, 3, 512, 512] -> [1, 4, 64, 64]
    out_vae = vae.encode(data["pixel_values"]).latent_dist.sample()
    out_vae = out_vae * 0.18215  # vae.config.scaling_factor

    # 随机噪声 [1, 4, 64, 64]
    noise = torch.randn_like(out_vae)

    # 随机噪声步
    noise_step = torch.randint(
        0, scheduler.config.num_train_timesteps, (1,), device=data["input_ids"].device
    ).long()

    # 添加噪声
    out_vae_noise = scheduler.add_noise(out_vae, noise, noise_step)

    # 从噪声图中预测噪声
    out_unet = unet(out_vae_noise, noise_step, out_encoder).sample

    return criterion(out_unet, noise)


def train_model(encoder, vae, unet, scheduler, optimizer, criterion, data_loader):
    """训练模型"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    unet.to(device)
    vae.to(device)
    encoder.to(device)
    unet.train()

    loss_sum = 0
    for epoch in range(200):
        for data in data_loader:
            # 移动数据到设备
            for k in data.keys():
                data[k] = data[k].to(device)

            loss = get_loss(data, encoder, vae, unet, scheduler, criterion)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            loss_sum += loss.item()

        if epoch % 10 == 0:
            print(epoch, loss_sum)
            loss_sum = 0

    # 保存训练后的模型
    DiffusionPipeline.from_pretrained(
        CHECKPOINT, unet=unet, text_encoder=encoder
    ).save_pretrained("./save")


def test_pipeline(pipeline_path, safety_checker=None):
    """测试管道"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = DiffusionPipeline.from_pretrained(
        pipeline_path, safety_checker=safety_checker
    ).to(device)

    texts = [
        "A photo of little dog in a bucket",
        "A photo of little dog swimming",
        "A photo of little dog sleeping",
        "A photo of little dog in a doghouse",
    ]

    images = [pipeline(text).images[0] for text in texts]

    plt.figure(figsize=(20, 10))
    for i in range(4):
        plt.subplot(1, 4, i + 1)
        plt.imshow(images[i])
        plt.axis("off")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行Diffusion模型的不同模块")
    parser.add_argument(
        "--module",
        type=str,
        choices=[
            "load_data",
            "preprocess",
            "train",
            "test_before",
            "test_after",
            "test_online",
        ],
        required=True,
        help="选择要运行的模块",
    )
    args = parser.parse_args()

    if args.module == "load_data":
        # 加载数据集
        dataset = load_local_dataset()
        print("本地数据集:", dataset)
        online_dataset, sample = load_online_dataset()
        print("在线数据集:", online_dataset)
        print("样本数据:", sample)

    elif args.module == "preprocess":
        # 预处理数据集
        dataset = load_local_dataset()
        processed_dataset = preprocess_dataset(dataset)
        print("预处理后的数据集:", processed_dataset)
        # 打印一条样本的形状和类型
        sample = processed_dataset[0]
        for k, v in sample.items():
            print(k, v.shape, v.dtype)

    elif args.module == "train":
        # 训练模型
        dataset = load_local_dataset()
        processed_dataset = preprocess_dataset(dataset)
        data_loader = create_data_loader(processed_dataset)
        encoder, vae, unet = load_models()
        scheduler, optimizer, criterion = setup_training_tools()
        train_model(encoder, vae, unet, scheduler, optimizer, criterion, data_loader)

    elif args.module == "test_before":
        # 测试训练前的模型
        test_pipeline(CHECKPOINT, safety_checker=None)

    elif args.module == "test_after":
        # 测试训练后的模型
        test_pipeline("./save", safety_checker=None)

    elif args.module == "test_online":
        # 测试在线模型
        test_pipeline(REPO_ID, safety_checker=None)