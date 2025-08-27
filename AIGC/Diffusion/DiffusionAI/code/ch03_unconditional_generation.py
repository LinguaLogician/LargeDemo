# -*- coding: utf-8 -*-
# @project: LargeDemo
# @filename: ch3_unconditional_generation.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/8/25 22:05

import torch
import torchvision
from torch.utils.data import DataLoader
from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline
from diffusers.optimization import get_scheduler
from datasets import load_dataset
from matplotlib import pyplot as plt
import argparse

# 全局常量
REPO_ID = 'lansinuote/diffusion.1.unconditional'


def define_model():
    """定义并初始化UNet2D模型"""
    model = UNet2DModel(
        sample_size=64,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256, 512, 512),
        down_block_types=(
            'DownBlock2D',
            'DownBlock2D',
            'DownBlock2D',
            'DownBlock2D',
            'AttnDownBlock2D',
            'DownBlock2D',
        ),
        up_block_types=(
            'UpBlock2D',
            'AttnUpBlock2D',
            'UpBlock2D',
            'UpBlock2D',
            'UpBlock2D',
            'UpBlock2D',
        ),
    )
    return model


def get_data_transforms():
    """获取数据预处理转换"""
    compose = torchvision.transforms.Compose([
        torchvision.transforms.Resize(
            64, interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
        torchvision.transforms.RandomCrop(64),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.5], [0.5])
    ])
    return compose


def load_and_preprocess_data():
    """加载并预处理数据集"""
    dataset = load_dataset(path=REPO_ID, split='train')

    compose = get_data_transforms()

    def transform_function(data):
        image = [compose(img) for img in data['image']]
        return {'image': image}

    dataset.set_transform(transform_function)
    return dataset


def create_data_loader(dataset, batch_size=16):
    """创建数据加载器"""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    return loader


def setup_training_components(model, loader, num_epochs=100):
    """设置训练所需的组件"""
    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule='linear',
        prediction_type='epsilon'
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        betas=(0.95, 0.999),
        weight_decay=1e-6,
        eps=1e-8
    )

    scheduler_lr = get_scheduler(
        'cosine',
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(loader) * num_epochs
    )

    criterion = torch.nn.MSELoss()

    return scheduler, optimizer, scheduler_lr, criterion


def compute_loss(image, model, scheduler, criterion, device):
    """计算损失函数"""
    # 随机噪声
    noise = torch.randn(image.shape).to(device)

    # 随机噪声步数
    noise_step = torch.randint(0, 1000, (image.shape[0],), device=device).long()

    # 添加噪声到图像
    image_noise = scheduler.add_noise(image, noise, noise_step)

    # 预测噪声
    out = model(image_noise, noise_step).sample

    # 计算MSE损失
    return criterion(out, noise)


def train_model(model, loader, num_epochs=10):
    """训练模型"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.train()

    scheduler, optimizer, scheduler_lr, criterion = setup_training_components(model, loader, num_epochs)

    loss_sum = 0
    for epoch in range(num_epochs):
        for i, data in enumerate(loader):
            images = data['image'].to(device)
            loss = compute_loss(images, model, scheduler, criterion, device)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler_lr.step()
            optimizer.zero_grad()

            loss_sum += loss.item()

        if epoch % 1 == 0:
            print(f"Epoch {epoch}, Loss: {loss_sum}")
            loss_sum = 0

    # 保存模型
    pipeline = DDPMPipeline(unet=model, scheduler=scheduler)
    pipeline.save_pretrained('./save')
    print("Model saved to './save'")

    return model


def test_model(pipeline_path=None, model=None, scheduler=None):
    """测试模型生成效果"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if pipeline_path:
        pipeline = DDPMPipeline.from_pretrained(pipeline_path)
    elif model and scheduler:
        pipeline = DDPMPipeline(unet=model, scheduler=scheduler)
    else:
        raise ValueError("Either pipeline_path or both model and scheduler must be provided")

    pipeline = pipeline.to(device)

    # 生成图像
    images = pipeline(
        batch_size=8,
        num_inference_steps=1000,
        output_type='numpy'
    ).images

    pipeline.to('cpu')
    torch.cuda.empty_cache()

    # 转换并显示图像
    images = (images * 255).round().astype('uint8')

    plt.figure(figsize=(10, 5))
    for i in range(8):
        plt.subplot(2, 4, i + 1)
        plt.imshow(images[i])
        plt.axis('off')

    plt.show()


def test_untrained_model():
    """测试未训练的模型"""
    model = define_model()
    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule='linear',
        prediction_type='epsilon'
    )
    test_model(model=model, scheduler=scheduler)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Diffusion Model Training and Testing')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['train', 'test_untrained', 'test_trained', 'test_pretrained'],
                        help='Mode to run: train, test_untrained, test_trained, or test_pretrained')
    parser.add_argument('--model_path', type=str, default='./save',
                        help='Path to trained model (for test_trained mode)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')

    args = parser.parse_args()

    if args.mode == 'train':
        print("Loading and preprocessing data...")
        dataset = load_and_preprocess_data()
        loader = create_data_loader(dataset)

        print("Initializing model...")
        model = define_model()

        print("Starting training...")
        train_model(model, loader, args.epochs)

    elif args.mode == 'test_untrained':
        print("Testing untrained model...")
        test_untrained_model()

    elif args.mode == 'test_trained':
        print(f"Testing trained model from {args.model_path}...")
        test_model(pipeline_path=args.model_path)

    elif args.mode == 'test_pretrained':
        print("Testing pretrained model from Hugging Face...")
        test_model(pipeline_path=REPO_ID)


if __name__ == "__main__":
    main()
