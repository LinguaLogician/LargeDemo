# -*- coding: utf-8 -*-
# @project: LargeDemo
# @filename: ch8_ae_model.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/8/26 9:21
# https://chat.deepseek.com/a/chat/s/2b759f29-5afd-4c9c-8bd9-a44da2fca9f5

import torch
import torchvision
from datasets import load_dataset
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from transformers import PreTrainedModel, PretrainedConfig


class Block(torch.nn.Module):
    """基础卷积块，可用于编码器和解码器"""

    def __init__(self, dim_in, dim_out, is_encoder=True):
        super().__init__()

        cnn_type = torch.nn.Conv2d
        if not is_encoder:
            cnn_type = torch.nn.ConvTranspose2d

        def block(dim_in, dim_out, kernel_size=3, stride=1, padding=1):
            return (
                cnn_type(dim_in, dim_out,
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

        self.res = cnn_type(dim_in, dim_out,
                            kernel_size=3,
                            stride=2,
                            padding=0)

    def forward(self, x):
        return self.s(x) + self.res(x)


class Encoder(torch.nn.Module):
    """编码器网络"""

    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            Block(3, 32, True),
            Block(32, 64, True),
            Block(64, 128, True),
            Block(128, 256, True),
            torch.nn.Flatten(),
            torch.nn.Linear(2304, 128),
        )

    def forward(self, x):
        return self.model(x)


class Decoder(torch.nn.Module):
    """解码器网络"""

    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(128, 256 * 4 * 4),
            torch.nn.InstanceNorm1d(256 * 4 * 4),
            torch.nn.Unflatten(dim=1, unflattened_size=(256, 4, 4)),
            Block(256, 128, False),
            Block(128, 64, False),
            Block(64, 32, False),
            Block(32, 3, False),
            torch.nn.UpsamplingNearest2d(size=64),
            torch.nn.Conv2d(in_channels=3, out_channels=3,
                            kernel_size=1, stride=1, padding=0),
            torch.nn.Tanh(),
        )

    def forward(self, x):
        return self.model(x)


class AEModel(PreTrainedModel):
    """包装类，用于加载预训练模型"""

    config_class = PretrainedConfig

    def __init__(self, config):
        super().__init__(config)
        self.encoder = Encoder()
        self.decoder = Decoder()


def load_dataset_from_hf():
    """加载并预处理数据集"""
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


def create_dataloader(dataset, batch_size=64, shuffle=True, drop_last=True):
    """创建数据加载器"""
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last
    )


def show_images(images, n_images=50):
    """显示图像"""
    images = images.to('cpu').detach()[:n_images]
    images = images.permute(0, 2, 3, 1)
    images = (images + 1) / 2

    n_cols = 10
    n_rows = n_images // n_cols

    plt.figure(figsize=(20, 10))

    for i in range(len(images)):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(images[i])
        plt.axis('off')

    plt.show()


def train_model(encoder, decoder, dataloader, num_epochs=1000, lr=2e-4, device='cuda'):
    """训练自编码器模型"""
    encoder.to(device)
    decoder.to(device)

    encoder.train()
    decoder.train()

    optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1, end_factor=0,
        total_iters=num_epochs * len(dataloader)
    )
    criterion = torch.nn.MSELoss()

    for epoch in range(num_epochs):
        for _, data in enumerate(dataloader):
            data = data.to(device)

            pred = decoder(encoder(data))
            loss = criterion(pred, data)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

            with torch.no_grad():
                gen = decoder(torch.randn(10, 128, device=device))
            show_images(gen, n_images=10)

    # 保存模型
    torch.save(encoder.to('cpu'), 'save/encoder.model')
    torch.save(decoder.to('cpu'), 'save/decoder.model')

    return encoder, decoder


def test_model(decoder_path='save/decoder.model', n_images=50):
    """测试模型生成能力"""
    decoder = torch.load(decoder_path)

    with torch.no_grad():
        gen = decoder(torch.randn(n_images, 128))

    show_images(gen, n_images)


def test_pretrained_model(model_name='lansinuote/gen.1.ae.book', n_images=50):
    """测试预训练模型"""
    model = AEModel.from_pretrained(model_name)

    with torch.no_grad():
        gen = model.decoder(torch.randn(n_images, 128))

    show_images(gen, n_images)


def main():
    """主函数，用于控制程序流程"""
    import argparse

    parser = argparse.ArgumentParser(description='自编码器图像生成')
    parser.add_argument('--mode', type=str, default='test',
                        choices=['train', 'test', 'test_pretrained'],
                        help='运行模式: train, test 或 test_pretrained')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='训练时的批次大小')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='学习率')
    parser.add_argument('--n_images', type=int, default=50,
                        help='生成图像数量')

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    if args.mode == 'train':
        # 训练模式
        print("加载数据集...")
        dataset = load_dataset_from_hf()
        dataloader = create_dataloader(dataset, batch_size=args.batch_size)

        print("初始化模型...")
        encoder = Encoder()
        decoder = Decoder()

        print("开始训练...")
        train_model(encoder, decoder, dataloader,
                    num_epochs=args.epochs, lr=args.lr, device=device)

    elif args.mode == 'test':
        # 测试本地模型
        print("测试本地模型...")
        test_model(n_images=args.n_images)

    elif args.mode == 'test_pretrained':
        # 测试预训练模型
        print("测试预训练模型...")
        test_pretrained_model(n_images=args.n_images)


if __name__ == '__main__':
    main()