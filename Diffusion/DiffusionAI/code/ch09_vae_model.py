# -*- coding: utf-8 -*-
# @project: LargeDemo
# @filename: ch9_vae_model.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/8/26 9:22
# https://chat.deepseek.com/a/chat/s/5427d883-8991-4418-b005-419af522cb2e

import torch
import torchvision
from datasets import load_dataset
from matplotlib import pyplot as plt
from transformers import PreTrainedModel, PretrainedConfig


def load_and_preprocess_data():
    """加载并预处理数据集"""
    print("Loading and preprocessing dataset...")
    dataset = load_dataset('lansinuote/gen.5.flower.book', split='train')
    dataset = dataset.remove_columns(['cls'])

    compose = torchvision.transforms.Compose([
        torchvision.transforms.Resize(64),
        torchvision.transforms.ToTensor(),
        lambda x: x * 2 - 1,
    ])

    def transform_function(data):
        image = compose(data['image'][0]).unsqueeze(dim=0)
        return {'image': image}

    dataset = dataset.with_transform(transform_function)

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


def show_images(images, n_images=50):
    """显示图像"""
    images = images.to('cpu').detach()[:n_images]
    images = images.permute(0, 2, 3, 1)
    images = (images + 1) / 2

    plt.figure(figsize=(20, 10))

    n_cols = min(10, n_images)
    n_rows = (n_images + n_cols - 1) // n_cols

    for i in range(len(images)):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(images[i])
        plt.axis('off')

    plt.show()


class Block(torch.nn.Module):
    """编码器/解码器的基础块"""

    def __init__(self, dim_in, dim_out, is_encoder=True):
        super().__init__()

        cnn_type = torch.nn.Conv2d if is_encoder else torch.nn.ConvTranspose2d

        def create_block(dim_in, dim_out, kernel_size=3, stride=1, padding=1):
            return (
                cnn_type(dim_in, dim_out,
                         kernel_size=kernel_size,
                         stride=stride,
                         padding=padding),
                torch.nn.BatchNorm2d(dim_out),
                torch.nn.LeakyReLU(),
            )

        self.s = torch.nn.Sequential(
            *create_block(dim_in, dim_in),
            *create_block(dim_in, dim_in),
            *create_block(dim_in, dim_in),
            *create_block(dim_in, dim_out, kernel_size=3, stride=2, padding=0),
            *create_block(dim_out, dim_out),
            *create_block(dim_out, dim_out),
            *create_block(dim_out, dim_out),
        )

        self.res = cnn_type(dim_in, dim_out, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        return self.s(x) + self.res(x)


def create_encoder():
    """创建编码器模型"""
    return torch.nn.Sequential(
        Block(3, 32, True),
        Block(32, 64, True),
        Block(64, 128, True),
        Block(128, 256, True),
        torch.nn.Flatten(),
        torch.nn.Linear(2304, 128),
    )


def create_decoder():
    """创建解码器模型"""
    return torch.nn.Sequential(
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


class VAE(torch.nn.Module):
    """变分自编码器模型"""

    def __init__(self):
        super().__init__()

        self.encoder = create_encoder()
        self.decoder = create_decoder()

        self.fc_mu = torch.nn.Linear(128, 128)
        self.fc_log_var = torch.nn.Linear(128, 128)

    def forward(self, data):
        hidden = self.encoder(data)
        mu = self.fc_mu(hidden)
        log_var = self.fc_log_var(hidden)

        randn = torch.randn(mu.shape, device=hidden.device)
        hidden = mu + (log_var / 2).exp() * randn

        return self.decoder(hidden), mu, log_var


def setup_training(vae_model, learning_rate=2e-4):
    """设置训练所需的优化器、调度器和损失函数"""
    optimizer = torch.optim.Adam(vae_model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1, end_factor=0, total_iters=1000 * 31  # 31是loader长度
    )
    criterion = torch.nn.MSELoss(reduction='none')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vae_model.to(device)
    vae_model.train()

    return optimizer, scheduler, criterion, device


def train_model(vae_model, data_loader, num_epochs=1000, save_path='save/vae.model'):
    """训练VAE模型"""
    optimizer, scheduler, criterion, device = setup_training(vae_model)

    for epoch in range(num_epochs):
        for _, data in enumerate(data_loader):
            data = data.to(device)

            pred, mu, log_var = vae_model(data)

            loss_mse = criterion(pred, data) * 10000
            loss_mse = loss_mse.mean(dim=(1, 2, 3))

            loss_kl = 1 + log_var - mu ** 2 - log_var.exp()
            loss_kl = loss_kl.sum(dim=1) * -0.5

            loss = (loss_mse + loss_kl).mean()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

            with torch.no_grad():
                gen = vae_model.decoder(torch.randn(10, 128, device=device))
            show_images(gen, n_images=10)

    # 保存模型
    torch.save(vae_model.to('cpu'), save_path)
    print(f"Model saved to {save_path}")


def test_model(model_path='save/vae.model', n_images=50):
    """测试训练好的模型"""
    vae = torch.load(model_path)

    with torch.no_grad():
        gen = vae.decoder(torch.randn(n_images, 128))

    show_images(gen, n_images=n_images)


def load_pretrained_model():
    """加载预训练模型"""

    class Model(PreTrainedModel):
        config_class = PretrainedConfig

        def __init__(self, config):
            super().__init__(config)
            self.vae = VAE()

    # 加载训练好的模型
    model = Model.from_pretrained('lansinuote/gen.2.vae.book')
    decoder = model.vae.decoder

    with torch.no_grad():
        gen = decoder(torch.randn(50, 128))

    show_images(gen)


def main():
    """主函数，根据用户选择执行不同的功能"""
    import argparse

    parser = argparse.ArgumentParser(description='VAE Image Generation')
    parser.add_argument('--mode', type=str, default='test',
                        choices=['load_data', 'train', 'test', 'pretrained'],
                        help='运行模式: load_data, train, test, pretrained')
    parser.add_argument('--model_path', type=str, default='save/vae.model',
                        help='模型保存/加载路径')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='训练轮数')

    args = parser.parse_args()

    if args.mode == 'load_data':
        # 仅加载和显示数据
        dataset = load_and_preprocess_data()
        loader = create_data_loader(dataset)
        print(f"Dataset shape: {dataset.shape}")
        print(f"Loader length: {len(loader)}")

        # 显示一些样本
        sample_batch = next(iter(loader))
        show_images(sample_batch)

    elif args.mode == 'train':
        # 训练模型
        dataset = load_and_preprocess_data()
        loader = create_data_loader(dataset)
        vae = VAE()

        train_model(vae, loader, num_epochs=args.epochs, save_path=args.model_path)

    elif args.mode == 'test':
        # 测试训练好的模型
        test_model(model_path=args.model_path)

    elif args.mode == 'pretrained':
        # 加载预训练模型并测试
        load_pretrained_model()


if __name__ == "__main__":
    main()