# -*- coding: utf-8 -*-
# @project: LargeDemo
# @filename: ch10_dcgan_model.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/8/26 9:23
# https://chat.deepseek.com/a/chat/s/499d98a0-bff8-4103-8034-d8edaf96ce85

import torch
import torchvision
from datasets import load_dataset
from matplotlib import pyplot as plt
from transformers import PreTrainedModel, PretrainedConfig
import torch.nn as nn
import torch.optim as optim


class DCGAN:
    def __init__(self, device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.cls = self.build_cls_model()
        self.gen = self.build_gen_model()
        self.criterion = nn.BCELoss()
        self.optimizer_cls = optim.Adam(self.cls.parameters(), lr=2e-4)
        self.optimizer_gen = optim.Adam(self.gen.parameters(), lr=2e-4)

        # 移动到设备
        self.cls.to(self.device)
        self.gen.to(self.device)

    def build_cls_model(self):
        """构建判别器模型"""
        return nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Conv2d(128, 128, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Flatten(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def build_gen_model(self):
        """构建生成器模型"""

        class Block(nn.Module):
            def __init__(self, dim_in, dim_out):
                super().__init__()

                def block(dim_in, dim_out, kernel_size=3, stride=1, padding=1):
                    return (
                        nn.ConvTranspose2d(dim_in, dim_out,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=padding),
                        nn.BatchNorm2d(dim_out),
                        nn.LeakyReLU(),
                    )

                self.s = nn.Sequential(
                    *block(dim_in, dim_in),
                    *block(dim_in, dim_in),
                    *block(dim_in, dim_in),
                    *block(dim_in, dim_out, kernel_size=3, stride=2, padding=0),
                    *block(dim_out, dim_out),
                    *block(dim_out, dim_out),
                    *block(dim_out, dim_out),
                )

                self.res = nn.ConvTranspose2d(dim_in, dim_out,
                                              kernel_size=3, stride=2, padding=0)

            def forward(self, x):
                return self.s(x) + self.res(x)

        return nn.Sequential(
            nn.Linear(128, 256 * 4 * 4),
            nn.InstanceNorm1d(256 * 4 * 4),
            nn.Unflatten(dim=1, unflattened_size=(256, 4, 4)),
            Block(256, 128),
            Block(128, 64),
            Block(64, 32),
            Block(32, 3),
            nn.UpsamplingNearest2d(size=64),
            nn.Conv2d(in_channels=3, out_channels=3,
                      kernel_size=1, stride=1, padding=0),
            nn.Tanh(),
        )

    def set_requires_grad(self, model, requires_grad):
        """设置模型参数是否需要梯度"""
        for param in model.parameters():
            param.requires_grad_(requires_grad)

    def train_cls(self, real_data):
        """训练判别器"""

        def update(data, label):
            pred = self.cls(data)
            label = torch.full((data.size(0), 1), label, device=self.device).float()
            loss = self.criterion(pred, label)

            loss.backward()
            self.optimizer_cls.step()
            self.optimizer_cls.zero_grad()

            return loss.item()

        self.set_requires_grad(self.cls, True)
        self.set_requires_grad(self.gen, False)

        # 生成假数据
        with torch.no_grad():
            fake_data = self.gen(torch.randn(real_data.size(0), 128, device=self.device))

        loss_fake = update(fake_data, 0)
        loss_real = update(real_data, 1)

        return loss_fake + loss_real

    def train_gen(self):
        """训练生成器"""
        self.set_requires_grad(self.cls, False)
        self.set_requires_grad(self.gen, True)

        fake_data = self.gen(torch.randn(64, 128, device=self.device))
        pred = self.cls(fake_data)

        loss = self.criterion(pred, torch.ones(pred.size(0), 1, device=self.device))
        loss.backward()
        self.optimizer_gen.step()
        self.optimizer_gen.zero_grad()

        return loss.item()

    def generate_images(self, num_images=10):
        """生成图像"""
        with torch.no_grad():
            noise = torch.randn(num_images, 128, device=self.device)
            generated_images = self.gen(noise)
        return generated_images

    def save_models(self, gen_path='gen.model', cls_path='cls.model'):
        """保存模型"""
        torch.save(self.gen.to('cpu'), gen_path)
        torch.save(self.cls.to('cpu'), cls_path)
        self.gen.to(self.device)
        self.cls.to(self.device)

    def load_models(self, gen_path='gen.model', cls_path='cls.model'):
        """加载模型"""
        self.gen = torch.load(gen_path).to(self.device)
        self.cls = torch.load(cls_path).to(self.device)


def load_dataset_from_hf():
    """从HuggingFace加载数据集"""
    dataset = load_dataset('lansinuote/gen.5.flower.book', split='train')
    dataset = dataset.remove_columns(['cls'])

    compose = torchvision.transforms.Compose([
        torchvision.transforms.Resize(64),
        torchvision.transforms.ToTensor(),
        lambda x: x * 2 - 1,
    ])

    def transform_func(data):
        image = compose(data['image'][0]).unsqueeze(dim=0)
        return {'image': image}

    dataset = dataset.with_transform(transform_func)

    dataset_tensor = torch.empty(len(dataset), 3, 64, 64)
    for i in range(len(dataset)):
        dataset_tensor[i] = dataset[i]['image']

    return dataset_tensor


def create_data_loader(dataset, batch_size=64, shuffle=True, drop_last=True):
    """创建数据加载器"""
    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last
    )


def show_images(images, num_images=50, figsize=(20, 10)):
    """显示图像"""
    images = images.to('cpu').detach()[:num_images]
    images = images.permute(0, 2, 3, 1)
    images = (images + 1) / 2

    plt.figure(figsize=figsize)

    rows = min(5, (num_images + 9) // 10)
    cols = min(10, num_images)

    for i in range(min(num_images, len(images))):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(images[i])
        plt.axis('off')

    plt.show()


def train_dcgan(dcgan, data_loader, num_epochs=100000, save_interval=5000):
    """训练DCGAN"""
    dcgan.cls.train()
    dcgan.gen.train()

    for epoch in range(num_epochs):
        real_data = next(iter(data_loader)).to(dcgan.device)
        loss_cls = dcgan.train_cls(real_data)
        loss_gen = dcgan.train_gen()

        if epoch % save_interval == 0:
            print(f"Epoch {epoch}: CLS Loss = {loss_cls:.4f}, GEN Loss = {loss_gen:.4f}")
            generated_images = dcgan.generate_images(10)
            show_images(generated_images, num_images=10)

    dcgan.save_models()


def load_pretrained_model():
    """加载预训练模型"""

    class Model(PreTrainedModel):
        config_class = PretrainedConfig

        def __init__(self, config):
            super().__init__(config)
            self.cls = None
            self.gen = None

    # 加载训练好的模型
    pretrained_model = Model.from_pretrained('lansinuote/gen.3.dcgan.book')
    return pretrained_model.gen


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='DCGAN 图像生成')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'generate', 'test_pretrained'],
                        help='运行模式: train, generate, test_pretrained')
    parser.add_argument('--num_images', type=int, default=50,
                        help='生成图像的数量')
    parser.add_argument('--epochs', type=int, default=100000,
                        help='训练轮数')
    parser.add_argument('--save_interval', type=int, default=5000,
                        help='保存间隔')

    args = parser.parse_args()

    if args.mode == 'train':
        # 训练模式
        print("加载数据集...")
        dataset = load_dataset_from_hf()
        data_loader = create_data_loader(dataset)

        print("初始化DCGAN...")
        dcgan = DCGAN()

        print("开始训练...")
        train_dcgan(dcgan, data_loader, args.epochs, args.save_interval)

    elif args.mode == 'generate':
        # 生成模式
        print("加载模型并生成图像...")
        dcgan = DCGAN()
        dcgan.load_models()

        generated_images = dcgan.generate_images(args.num_images)
        show_images(generated_images, args.num_images)

    elif args.mode == 'test_pretrained':
        # 测试预训练模型
        print("加载预训练模型并生成图像...")
        gen = load_pretrained_model()

        with torch.no_grad():
            noise = torch.randn(args.num_images, 128)
            generated_images = gen(noise)

        show_images(generated_images, args.num_images)


if __name__ == "__main__":
    main()