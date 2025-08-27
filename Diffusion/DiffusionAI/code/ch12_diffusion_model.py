# -*- coding: utf-8 -*-
# @project: LargeDemo
# @filename: ch12_diffusion_model.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/8/26 10:13
# https://chat.deepseek.com/a/chat/s/58c9a158-1773-4180-85eb-634af1b167f8

import torch
import torchvision
from datasets import load_dataset
from transformers import PreTrainedModel, PretrainedConfig
from matplotlib import pyplot as plt
import argparse

# -------------------- 数据加载与预处理 --------------------
def get_dataset():
    # 加载数据集
    dataset = load_dataset('lansinuote/gen.5.flower.book', split='train')

    # 删除多余的字段
    dataset = dataset.remove_columns(['cls'])

    # 图像数据预处理
    compose = torchvision.transforms.Compose([
        torchvision.transforms.Resize(64),
        torchvision.transforms.ToTensor(),
    ])

    def f(data):
        image = compose(data['image'][0]).unsqueeze(dim=0)
        return {'image': image}

    dataset = dataset.with_transform(f)

    # 为了加速数据遍历的效率,把全体数据载入内存以加速IO
    dataset_tensor = torch.empty(len(dataset), 3, 64, 64)

    for i in range(len(dataset)):
        dataset_tensor[i] = dataset[i]['image']

    return dataset_tensor

def get_dataloader(dataset, batch_size=64, shuffle=True, drop_last=True):
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last
    )
    return loader

# -------------------- 可视化工具 --------------------
def show(images):
    images = images.to('cpu').detach()[:50]
    images = images.permute(0, 2, 3, 1)

    plt.figure(figsize=(20, 10))

    for i in range(len(images)):
        plt.subplot(5, 10, i + 1)
        plt.imshow(images[i])
        plt.axis('off')

    plt.show()

# -------------------- 噪声调度 --------------------
def schedule(time, method='offset_cosine'):
    if method == 'linear':
        t = 1 - (1e-4 + time * (0.02 - 1e-4))
        t = torch.cumprod(t, dim=0)
        return (1 - t).sqrt(), t.sqrt()

    if method == 'cosine':
        # 1.5707963267948966 = pi/2
        t = time * 1.5707963267948966
        return t.sin(), t.cos()

    if method == 'offset_cosine':
        # 0.3175604292915215 = acos(0.95)
        # 1.2332345639299847 = acos(0.02) - acos(0.95)
        t = 0.3175604292915215 + time * 1.2332345639299847
        return t.sin(), t.cos()

# -------------------- 模型组件 --------------------
class Combine(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 6.907755278982137 = log(1000)
        t = torch.linspace(0.0, 6.907755278982137, 16).exp()
        t *= 2
        # 3.141592653589793 = pi
        t *= 3.141592653589793
        self.register_buffer('t', t)

        self.upsample = torch.nn.UpsamplingNearest2d(size=(64, 64))
        self.cnn = torch.nn.Conv2d(3, 32, kernel_size=1, stride=1, padding=0)

    def get_var(self, var):
        # [b, 1] -> [b, 16]
        var = self.t * var
        # [b, 16+16] -> [b, 32]
        var = torch.cat((var.sin(), var.cos()), dim=1)
        # [b, 32] -> [b, 32, 1, 1]
        var = var.unsqueeze(dim=-1).unsqueeze(dim=-1)
        # [b, 32, 1, 1] -> [b, 32, 64, 64]
        var = self.upsample(var)
        return var

    def get_image(self, image):
        # [b, 3, 64, 64] -> [b, 32, 64, 64]
        image = self.cnn(image)
        return image

    def forward(self, image, var):
        # image -> [b, 3, 64, 64]
        # var -> [b, 1, 1, 1]
        # [b, 1, 1, 1] -> [b, 1]
        var = var.squeeze(dim=-1).squeeze(dim=-1)
        # [b, 1] -> [b, 32, 64, 64]
        var = self.get_var(var)
        # [b, 3, 64, 64] -> [b, 32, 64, 64]
        image = self.get_image(image)
        # [b, 32+32, 64, 64] -> [b, 64, 64, 64]
        combine = torch.cat((image, var), dim=1)
        return combine

class Residual(torch.nn.Module):
    def __init__(self, channel_in, channel_out):
        super().__init__()
        self.cnn = torch.nn.Conv2d(channel_in, channel_out, kernel_size=1, stride=1, padding=0)
        self.s = torch.nn.Sequential(
            torch.nn.BatchNorm2d(channel_in),
            torch.nn.Conv2d(channel_in, channel_out, kernel_size=3, stride=1, padding=1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(channel_out, channel_out, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        return self.cnn(x) + self.s(x)

class UNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.down = torch.nn.ModuleList([
            Residual(64, 32),
            Residual(32, 32),
            torch.nn.AvgPool2d(2),
            Residual(32, 64),
            Residual(64, 64),
            torch.nn.AvgPool2d(2),
            Residual(64, 96),
            Residual(96, 96),
            torch.nn.AvgPool2d(2),
        ])

        self.middle = torch.nn.ModuleList([
            Residual(96, 128),
            Residual(128, 128),
        ])

        self.up = torch.nn.ModuleList([
            torch.nn.UpsamplingBilinear2d(scale_factor=2),
            Residual(224, 96),
            Residual(192, 96),
            torch.nn.UpsamplingBilinear2d(scale_factor=2),
            Residual(160, 64),
            Residual(128, 64),
            torch.nn.UpsamplingBilinear2d(scale_factor=2),
            Residual(96, 32),
            Residual(64, 32),
        ])

        self.out = torch.nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0)

    def forward(self, image):
        out = []
        for i in self.down:
            image = i(image)
            if type(i) == Residual:
                out.append(image)

        for i in self.middle:
            image = i(image)

        for i in self.up:
            if type(i) == Residual:
                p = out.pop()
                image = torch.cat((image, p), dim=1)
            image = i(image)

        image = self.out(image)
        return image

class Diffusion(PreTrainedModel):
    config_class = PretrainedConfig

    def __init__(self, config):
        super().__init__(config)
        self.norm = torch.nn.BatchNorm2d(3, affine=False)
        self.unet = UNet()
        self.combine = Combine()

    def forward(self, image):
        # image -> [b, 3, 64, 64]
        b = image.shape[0]
        # 对图像正则化处理
        image = self.norm(image)
        # 随机噪声
        noise = torch.randn(b, 3, 64, 64, device=image.device)
        # 随机系数
        noise_r, image_r = schedule(torch.rand(b, 1, 1, 1, device=image.device))
        # 合并图像和噪声
        image = image * image_r + noise * noise_r
        # 合并噪声图和噪声系数
        combine = self.combine(image, noise_r**2)
        # 从噪声图中预测出噪声
        pred_noise = self.unet(combine)
        return noise, pred_noise

# -------------------- 生成与训练 --------------------
def generate(diffusion_model, n, device):
    # 从随机噪声开始生成
    image = torch.randn(n, 3, 64, 64, device=device)

    # 生成20个step
    for i in range(20):
        time = torch.full(size=(n, 1, 1, 1),
                          fill_value=(20 - i) / 20,
                          dtype=torch.float32,
                          device=device)

        # 随机系数
        noise_r, image_r = schedule(time)

        # 合并噪声图和噪声系数
        combine = diffusion_model.combine(image, noise_r**2)

        # 从噪声图中预测出噪声
        pred_noise = diffusion_model.unet(combine)

        # 根据预测的噪声还原图像
        pred_image = (image - noise_r * pred_noise) / image_r

        # 再次计算随机系数
        time = time - (1 / 20)
        noise_r, image_r = schedule(time)

        # 重新向图像中添加噪声,以进行下一个step的计算
        image = image_r * pred_image + noise_r * pred_noise

    # 根据正则化数据反正则化图像
    mean = diffusion_model.norm.running_mean.reshape(1, 3, 1, 1)
    std = (diffusion_model.norm.running_var**0.5).reshape(1, 3, 1, 1)

    pred_image = mean + pred_image * std
    pred_image = pred_image.clip(0.0, 1.0)

    return pred_image

def train_model(diffusion_model, loader, num_epochs=2000, save_path='save/diffusion.model'):
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.AdamW(diffusion_model.parameters(), lr=2e-4, weight_decay=1e-4)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    diffusion_model.to(device)
    diffusion_model.train()

    for epoch in range(num_epochs):
        for i, data in enumerate(loader):
            noise, pred_noise = diffusion_model(data.to(device))
            loss = criterion(noise, pred_noise)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
            diffusion_model.eval()
            show(generate(diffusion_model, 10, device))
            diffusion_model.train()

    torch.save(diffusion_model.to('cpu'), save_path)
    print(f"Model saved to {save_path}")

# -------------------- 主程序入口 --------------------
def main():
    parser = argparse.ArgumentParser(description='Diffusion Model Training and Generation')
    parser.add_argument('--mode', type=str, choices=['train', 'generate', 'test_pretrained'],
                        default='generate', help='Mode to run: train, generate, or test_pretrained')
    parser.add_argument('--num_images', type=int, default=50, help='Number of images to generate')
    parser.add_argument('--model_path', type=str, default='save/diffusion.model', help='Path to model for generation')
    args = parser.parse_args()

    if args.mode == 'train':
        print("Loading dataset...")
        dataset = get_dataset()
        loader = get_dataloader(dataset)
        print("Initializing model...")
        diffusion = Diffusion(PretrainedConfig())
        print("Starting training...")
        train_model(diffusion, loader)

    elif args.mode == 'generate':
        print(f"Loading model from {args.model_path}...")
        diffusion = torch.load(args.model_path)
        diffusion.eval()
        print(f"Generating {args.num_images} images...")
        with torch.no_grad():
            images = generate(diffusion, args.num_images, 'cpu')
            show(images)

    elif args.mode == 'test_pretrained':
        print("Loading pretrained model...")
        diffusion = Diffusion.from_pretrained('lansinuote/gen.9.diffusion.book')
        diffusion.eval()
        print(f"Generating {args.num_images} images...")
        with torch.no_grad():
            images = generate(diffusion, args.num_images, 'cpu')
            show(images)

if __name__ == '__main__':
    main()
