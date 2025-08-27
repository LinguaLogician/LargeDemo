# -*- coding: utf-8 -*-
# @project: LargeDemo
# @filename: wgangp_model.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/8/26 9:24
# https://chat.deepseek.com/a/chat/s/f68160c7-70e6-430b-824e-82cb0e4e3dcc

import torch
import torchvision
from datasets import load_dataset
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from transformers import PreTrainedModel, PretrainedConfig


class WGAN_GP:
    def __init__(self, device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.cls = None
        self.gen = None
        self.optimizer_cls = None
        self.optimizer_gen = None
        self.dataset = None
        self.loader = None

    def get_dataset(self):
        """加载数据集"""
        # 加载数据集
        dataset = load_dataset('lansinuote/gen.5.flower.book', split='train')

        # 删除多余的字段
        dataset = dataset.remove_columns(['cls'])

        # 图像数据预处理
        compose = torchvision.transforms.Compose([
            torchvision.transforms.Resize(64),
            torchvision.transforms.ToTensor(),
            lambda x: x * 2 - 1,
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

    def setup_data_loader(self, batch_size=64):
        """设置数据加载器"""
        if self.dataset is None:
            self.dataset = self.get_dataset()

        self.loader = DataLoader(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True
        )
        return self.loader

    def define_cls_model(self):
        """定义判别器模型"""
        self.cls = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=1),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=1),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=1),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=1),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Flatten(),
            torch.nn.Linear(512, 1),
        )
        return self.cls

    def define_gen_model(self):
        """定义生成器模型"""

        class Block(torch.nn.Module):
            def __init__(self, dim_in, dim_out):
                super().__init__()

                def block(dim_in, dim_out, kernel_size=3, stride=1, padding=1):
                    return (
                        torch.nn.ConvTranspose2d(dim_in,
                                                 dim_out,
                                                 kernel_size=kernel_size,
                                                 stride=stride,
                                                 padding=padding),
                        torch.nn.BatchNorm2d(dim_out),
                        torch.nn.LeakyReLU(),
                    )

                self.s = torch.nn.Sequential(
                    *block(dim_in, dim_in),
                    *block(dim_in, dim_in),
                    *block(dim_in, dim_in),
                    *block(dim_in, dim_out, kernel_size=3, stride=2, padding=0),
                    *block(dim_out, dim_out),
                    *block(dim_out, dim_out),
                    *block(dim_out, dim_out),
                )

                self.res = torch.nn.ConvTranspose2d(dim_in,
                                                    dim_out,
                                                    kernel_size=3,
                                                    stride=2,
                                                    padding=0)

            def forward(self, x):
                return self.s(x) + self.res(x)

        self.gen = torch.nn.Sequential(
            torch.nn.Linear(128, 256 * 4 * 4),
            torch.nn.InstanceNorm1d(256 * 4 * 4),
            torch.nn.Unflatten(dim=1, unflattened_size=(256, 4, 4)),
            Block(256, 128),
            Block(128, 64),
            Block(64, 32),
            Block(32, 3),
            torch.nn.UpsamplingNearest2d(size=64),
            torch.nn.Conv2d(in_channels=3,
                            out_channels=3,
                            kernel_size=1,
                            stride=1,
                            padding=0),
            torch.nn.Tanh(),
        )
        return self.gen

    def setup_models(self):
        """设置模型和优化器"""
        if self.cls is None:
            self.define_cls_model()
        if self.gen is None:
            self.define_gen_model()

        self.cls.to(self.device)
        self.gen.to(self.device)

        self.cls.train()
        self.gen.train()

        self.optimizer_cls = torch.optim.Adam(self.cls.parameters(), lr=2e-4)
        self.optimizer_gen = torch.optim.Adam(self.gen.parameters(), lr=2e-4)

        return self.cls, self.gen, self.optimizer_cls, self.optimizer_gen

    @staticmethod
    def set_requires_grad(model, requires_grad):
        """设置模型参数是否需要梯度"""
        for param in model.parameters():
            param.requires_grad_(requires_grad)

    @staticmethod
    def wasserstein(pred, label):
        """Wasserstein损失函数"""
        return -(pred * label).mean()

    def get_gradient_penalty(self, real, fake):
        """计算梯度惩罚"""
        # real -> [64, 3, 64, 64]
        # fake -> [64, 3, 64, 64]

        batch_size = real.shape[0]
        r = torch.rand((batch_size, 1, 1, 1), device=self.device)
        r.requires_grad = True

        # [64, 3, 64, 64]
        merge = r * real + (1 - r) * fake

        # [64, 3, 64, 64] -> [64, 1]
        pred = self.cls(merge)

        grad = torch.autograd.grad(
            inputs=merge,
            outputs=pred,
            grad_outputs=torch.ones(batch_size, 1, device=self.device),
            create_graph=True,
            retain_graph=True
        )

        # [64, 3, 64, 64] -> [64, 12288]
        grad = grad[0].reshape(batch_size, -1)

        # [64, 12288] -> [64]
        grad = grad.norm(p=2, dim=1)

        # [64] -> scala
        return (1 - grad).pow(2).mean()

    def train_cls(self):
        """训练判别器"""
        self.set_requires_grad(self.cls, True)
        self.set_requires_grad(self.gen, False)

        # 得到三份数据
        data_real = next(iter(self.loader)).to(self.device)
        with torch.no_grad():
            data_fake = self.gen(torch.randn(data_real.shape[0], 128, device=self.device))

        # 分别计算
        pred_real = self.cls(data_real)
        pred_fake = self.cls(data_fake)

        # 求loss,加权求和
        loss_real = self.wasserstein(pred_real, torch.ones(pred_real.shape[0], 1, device=self.device))
        loss_fake = self.wasserstein(pred_fake, -torch.ones(pred_fake.shape[0], 1, device=self.device))
        loss_grad = self.get_gradient_penalty(data_real, data_fake)

        loss = loss_fake + loss_real + loss_grad * 10

        loss.backward()
        self.optimizer_cls.step()
        self.optimizer_cls.zero_grad()

        return loss.item()

    def train_gen(self):
        """训练生成器"""
        self.set_requires_grad(self.cls, False)
        self.set_requires_grad(self.gen, True)

        batch_size = next(iter(self.loader)).shape[0]
        pred = self.cls(self.gen(torch.randn(batch_size, 128, device=self.device)))

        loss = self.wasserstein(pred, torch.ones(batch_size, 1, device=self.device))
        loss.backward()
        self.optimizer_gen.step()
        self.optimizer_gen.zero_grad()

        return loss.item()

    def show(self, images, n_images=50):
        """显示图像"""
        images = images.to('cpu').detach()[:n_images]
        images = images.permute(0, 2, 3, 1)
        images = (images + 1) / 2

        n_cols = min(10, n_images)
        n_rows = (n_images + n_cols - 1) // n_cols

        plt.figure(figsize=(n_cols * 2, n_rows * 2))

        for i in range(len(images)):
            plt.subplot(n_rows, n_cols, i + 1)
            plt.imshow(images[i])
            plt.axis('off')

        plt.show()

    def train(self, num_epochs=200000, cls_steps=5, save_interval=10000, save_path='save/'):
        """训练模型"""
        for epoch in range(num_epochs):
            for _ in range(cls_steps):
                loss_cls = self.train_cls()

            loss_gen = self.train_gen()

            if epoch % save_interval == 0:
                print(f"Epoch {epoch}: CLS Loss = {loss_cls:.4f}, GEN Loss = {loss_gen:.4f}")
                with torch.no_grad():
                    pred = self.gen(torch.randn(10, 128, device=self.device))
                self.show(pred, n_images=10)

        # 保存模型
        torch.save(self.gen.to('cpu'), f'{save_path}/gen.model')
        torch.save(self.cls.to('cpu'), f'{save_path}/cls.model')

        # 将模型移回设备
        self.gen.to(self.device)
        self.cls.to(self.device)

    def load_pretrained_model(self, model_path='lansinuote/gen.5.wgangp.book'):
        """加载预训练模型"""

        class Model(PreTrainedModel):
            config_class = PretrainedConfig

            def __init__(self, config):
                super().__init__(config)
                self.cls = None
                self.gen = None

        # 加载训练好的模型
        model = Model.from_pretrained(model_path)
        self.gen = model.gen.to(self.device)
        return self.gen

    def generate_samples(self, n_samples=50, use_pretrained=False):
        """生成样本"""
        if use_pretrained:
            if self.gen is None:
                self.load_pretrained_model()
        elif self.gen is None:
            raise ValueError("Generator model not defined. Please train or load a model first.")

        with torch.no_grad():
            pred = self.gen(torch.randn(n_samples, 128, device=self.device))

        self.show(pred, n_samples)
        return pred


def main():
    import argparse

    parser = argparse.ArgumentParser(description='WGAN-GP Implementation')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'generate', 'test_data'],
                        help='运行模式: train, generate 或 test_data')
    parser.add_argument('--use_pretrained', action='store_true',
                        help='是否使用预训练模型进行生成')
    parser.add_argument('--n_samples', type=int, default=50,
                        help='生成样本的数量')
    parser.add_argument('--epochs', type=int, default=200000,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='批次大小')

    args = parser.parse_args()

    wgan_gp = WGAN_GP()

    if args.mode == 'train':
        print("设置数据加载器...")
        wgan_gp.setup_data_loader(batch_size=args.batch_size)

        print("设置模型...")
        wgan_gp.setup_models()

        print("开始训练...")
        wgan_gp.train(num_epochs=args.epochs)

    elif args.mode == 'generate':
        print("生成样本...")
        wgan_gp.generate_samples(n_samples=args.n_samples, use_pretrained=args.use_pretrained)

    elif args.mode == 'test_data':
        print("测试数据加载和显示...")
        wgan_gp.setup_data_loader(batch_size=args.batch_size)
        batch = next(iter(wgan_gp.loader))
        wgan_gp.show(batch)


if __name__ == '__main__':
    main()
