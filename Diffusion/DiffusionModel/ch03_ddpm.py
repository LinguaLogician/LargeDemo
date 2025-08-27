# -*- coding: utf-8 -*-
# @project: LargeDemo
# @filename: chapter3_ddpm.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/8/25 14:57
# https://chat.deepseek.com/a/chat/s/bae0abd1-20d9-4cd2-9e03-cbf4a3ea1d74
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler, UNet2DModel
from matplotlib import pyplot as plt
import argparse


def setup_environment():
    """设置环境和安装必要的包"""
    # 检查是否安装了diffusers，如果没有则安装
    try:
        import diffusers
    except ImportError:
        print("正在安装diffusers...")
        import subprocess
        subprocess.check_call(["pip", "install", "-q", "diffusers"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'使用设备: {device}')
    return device


def load_and_prepare_data(data_path="mnist/", batch_size=8):
    """加载和准备MNIST数据集"""
    dataset = torchvision.datasets.MNIST(
        root=data_path,
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor()
    )
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataset, train_dataloader


def visualize_data_samples(train_dataloader):
    """可视化数据样本"""
    x, y = next(iter(train_dataloader))
    print('输入形状:', x.shape)
    print('标签:', y)
    plt.imshow(torchvision.utils.make_grid(x)[0], cmap='Greys')
    plt.title('输入数据样本')
    plt.show()
    return x, y


def corrupt_data(x, amount):
    """使用指定数量的噪声破坏输入数据"""
    noise = torch.rand_like(x)
    amount = amount.view(-1, 1, 1, 1)  # 调整形状以便广播
    return x * (1 - amount) + noise * amount


def visualize_corruption(x):
    """可视化数据损坏过程"""
    fig, axs = plt.subplots(2, 1, figsize=(12, 5))
    axs[0].set_title('输入数据')
    axs[0].imshow(torchvision.utils.make_grid(x)[0], cmap='Greys')

    # 添加噪声
    amount = torch.linspace(0, 1, x.shape[0])  # 从左到右 -> 更多的损坏
    noised_x = corrupt_data(x, amount)

    # 绘制损坏后的版本
    axs[1].set_title('损坏的数据 (-- 损坏程度增加 -->)')
    axs[1].imshow(torchvision.utils.make_grid(noised_x)[0], cmap='Greys')
    plt.tight_layout()
    plt.show()

    return noised_x


class BasicUNet(nn.Module):
    """一个简化的UNet实现"""

    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.down_layers = torch.nn.ModuleList([
            nn.Conv2d(in_channels, 32, kernel_size=5, padding=2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
        ])
        self.up_layers = torch.nn.ModuleList([
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.Conv2d(64, 32, kernel_size=5, padding=2),
            nn.Conv2d(32, out_channels, kernel_size=5, padding=2),
        ])
        self.act = nn.SiLU()  # 激活函数
        self.downscale = nn.MaxPool2d(2)
        self.upscale = nn.Upsample(scale_factor=2)

    def forward(self, x):
        h = []
        for i, l in enumerate(self.down_layers):
            x = self.act(l(x))  # 通过层和激活函数
            if i < 2:  # 对于除了第三个（最后一个）下采样层之外的所有层：
                h.append(x)  # 存储输出用于跳跃连接
                x = self.downscale(x)  # 下采样准备下一层

        for i, l in enumerate(self.up_layers):
            if i > 0:  # 对于除了第一个上采样层之外的所有层
                x = self.upscale(x)  # 上采样
                x += h.pop()  # 获取存储的输出（跳跃连接）
            x = self.act(l(x))  # 通过层和激活函数

        return x


def test_basic_unet():
    """测试BasicUNet模型"""
    net = BasicUNet()
    x = torch.rand(8, 1, 28, 28)
    output_shape = net(x).shape
    print(f"BasicUNet输出形状: {output_shape}")

    num_params = sum([p.numel() for p in net.parameters()])
    print(f"BasicUNet参数数量: {num_params}")

    return net, num_params


def train_basic_unet(dataset, device, batch_size=128, n_epochs=3, lr=1e-3):
    """训练BasicUNet模型"""
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 创建网络
    net = BasicUNet()
    net.to(device)

    # 损失函数
    loss_fn = nn.MSELoss()

    # 优化器
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    # 记录损失
    losses = []

    # 训练循环
    for epoch in range(n_epochs):
        for x, y in train_dataloader:
            # 获取数据并准备损坏版本
            x = x.to(device)  # 数据在GPU上
            noise_amount = torch.rand(x.shape[0]).to(device)  # 选择随机噪声量
            noisy_x = corrupt_data(x, noise_amount)  # 创建噪声x

            # 获取模型预测
            pred = net(noisy_x)

            # 计算损失
            loss = loss_fn(pred, x)  # 输出与真实'干净'x的接近程度

            # 反向传播和更新参数
            opt.zero_grad()
            loss.backward()
            opt.step()

            # 存储损失
            losses.append(loss.item())

        # 打印该epoch的平均损失
        avg_loss = sum(losses[-len(train_dataloader):]) / len(train_dataloader)
        print(f'完成epoch {epoch}. 该epoch的平均损失: {avg_loss:05f}')

    # 绘制损失曲线
    plt.plot(losses)
    plt.ylim(0, 0.1)
    plt.title('训练损失曲线')
    plt.xlabel('迭代次数')
    plt.ylabel('损失')
    plt.show()

    return net, losses


def visualize_predictions(net, train_dataloader, device):
    """可视化模型在噪声输入上的预测"""
    # 获取一些数据
    x, y = next(iter(train_dataloader))
    x = x[:8]  # 只使用前8个以便绘图

    # 用一系列数量损坏
    amount = torch.linspace(0, 1, x.shape[0])  # 从左到右 -> 更多的损坏
    noised_x = corrupt_data(x, amount)

    # 获取模型预测
    with torch.no_grad():
        preds = net(noised_x.to(device)).detach().cpu()

    # 绘图
    fig, axs = plt.subplots(3, 1, figsize=(12, 7))
    axs[0].set_title('输入数据')
    axs[0].imshow(torchvision.utils.make_grid(x)[0].clip(0, 1), cmap='Greys')
    axs[1].set_title('损坏的数据')
    axs[1].imshow(torchvision.utils.make_grid(noised_x)[0].clip(0, 1), cmap='Greys')
    axs[2].set_title('网络预测')
    axs[2].imshow(torchvision.utils.make_grid(preds)[0].clip(0, 1), cmap='Greys')
    plt.tight_layout()
    plt.show()

    return preds


def sampling_process(net, device, n_steps=5, batch_size=8):
    """采样过程：将过程分解为多个步骤"""
    x = torch.rand(batch_size, 1, 28, 28).to(device)  # 从随机开始
    step_history = [x.detach().cpu()]
    pred_output_history = []

    for i in range(n_steps):
        with torch.no_grad():  # 推理期间不需要跟踪梯度
            pred = net(x)  # 预测去噪的x0
        pred_output_history.append(pred.detach().cpu())  # 存储模型输出用于绘图
        mix_factor = 1 / (n_steps - i)  # 我们向预测移动多少
        x = x * (1 - mix_factor) + pred * mix_factor  # 部分移动到那里
        step_history.append(x.detach().cpu())  # 存储步骤用于绘图

    fig, axs = plt.subplots(n_steps, 2, figsize=(9, 4), sharex=True)
    axs[0, 0].set_title('x (模型输入)')
    axs[0, 1].set_title('模型预测')
    for i in range(n_steps):
        axs[i, 0].imshow(torchvision.utils.make_grid(step_history[i])[0].clip(0, 1), cmap='Greys')
        axs[i, 1].imshow(torchvision.utils.make_grid(pred_output_history[i])[0].clip(0, 1), cmap='Greys')
    plt.tight_layout()
    plt.show()

    return x, step_history, pred_output_history


def generate_samples(net, device, n_steps=40, n_samples=64):
    """使用更多采样步骤生成样本"""
    x = torch.rand(n_samples, 1, 28, 28).to(device)
    for i in range(n_steps):
        noise_amount = torch.ones((x.shape[0],)).to(device) * (1 - (i / n_steps))  # 从高开始变低
        with torch.no_grad():
            pred = net(x)
        mix_factor = 1 / (n_steps - i)
        x = x * (1 - mix_factor) + pred * mix_factor

    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.imshow(torchvision.utils.make_grid(x.detach().cpu(), nrow=8)[0].clip(0, 1), cmap='Greys')
    ax.set_title('生成的样本')
    plt.show()

    return x


def create_diffusers_unet():
    """创建Diffusers的UNet2DModel"""
    model = UNet2DModel(
        sample_size=28,  # 目标图像分辨率
        in_channels=1,  # 输入通道数，RGB图像为3
        out_channels=1,  # 输出通道数
        layers_per_block=2,  # 每个UNet块使用多少ResNet层
        block_out_channels=(32, 64, 64),  # 大致匹配我们的基本unet示例
        down_block_types=(
            "DownBlock2D",  # 常规ResNet下采样块
            "AttnDownBlock2D",  # 带有空间自注意力的ResNet下采样块
            "AttnDownBlock2D",
        ),
        up_block_types=(
            "AttnUpBlock2D",
            "AttnUpBlock2D",  # 带有空间自注意力的ResNet上采样块
            "UpBlock2D",  # 常规ResNet上采样块
        ),
    )
    print(model)

    num_params = sum([p.numel() for p in model.parameters()])
    print(f"Diffusers UNet参数数量: {num_params} (BasicUNet约309k参数)")

    return model, num_params


def train_diffusers_unet(dataset, device, batch_size=128, n_epochs=3, lr=1e-3):
    """训练Diffusers的UNet2DModel"""
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 创建网络
    net = UNet2DModel(
        sample_size=28,  # 目标图像分辨率
        in_channels=1,  # 输入通道数，RGB图像为3
        out_channels=1,  # 输出通道数
        layers_per_block=2,  # 每个UNet块使用多少ResNet层
        block_out_channels=(32, 64, 64),  # 大致匹配我们的基本unet示例
        down_block_types=(
            "DownBlock2D",  # 常规ResNet下采样块
            "AttnDownBlock2D",  # 带有空间自注意力的ResNet下采样块
            "AttnDownBlock2D",
        ),
        up_block_types=(
            "AttnUpBlock2D",
            "AttnUpBlock2D",  # 带有空间自注意力的ResNet上采样块
            "UpBlock2D",  # 常规ResNet上采样块
        ),
    )
    net.to(device)

    # 损失函数
    loss_fn = nn.MSELoss()

    # 优化器
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    # 记录损失
    losses = []

    # 训练循环
    for epoch in range(n_epochs):
        for x, y in train_dataloader:
            # 获取数据并准备损坏版本
            x = x.to(device)  # 数据在GPU上
            noise_amount = torch.rand(x.shape[0]).to(device)  # 选择随机噪声量
            noisy_x = corrupt_data(x, noise_amount)  # 创建噪声x

            # 获取模型预测
            pred = net(noisy_x, 0).sample  # 使用时间步0，添加.sample

            # 计算损失
            loss = loss_fn(pred, x)  # 输出与真实'干净'x的接近程度

            # 反向传播和更新参数
            opt.zero_grad()
            loss.backward()
            opt.step()

            # 存储损失
            losses.append(loss.item())

        # 打印该epoch的平均损失
        avg_loss = sum(losses[-len(train_dataloader):]) / len(train_dataloader)
        print(f'完成epoch {epoch}. 该epoch的平均损失: {avg_loss:05f}')

    # 绘制损失和样本
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # 损失
    axs[0].plot(losses)
    axs[0].set_ylim(0, 0.1)
    axs[0].set_title('随时间变化的损失')
    axs[0].set_xlabel('迭代次数')
    axs[0].set_ylabel('损失')

    # 样本
    n_steps = 40
    x = torch.rand(64, 1, 28, 28).to(device)
    for i in range(n_steps):
        noise_amount = torch.ones((x.shape[0],)).to(device) * (1 - (i / n_steps))  # 从高开始变低
        with torch.no_grad():
            pred = net(x, 0).sample
        mix_factor = 1 / (n_steps - i)
        x = x * (1 - mix_factor) + pred * mix_factor

    axs[1].imshow(torchvision.utils.make_grid(x.detach().cpu(), nrow=8)[0].clip(0, 1), cmap='Greys')
    axs[1].set_title('生成的样本')
    plt.tight_layout()
    plt.show()

    return net, losses


def visualize_ddpm_noise(noise_scheduler, train_dataloader, device):
    """可视化DDPM不同时间步的噪声过程"""
    # 噪声一批图像以查看效果
    fig, axs = plt.subplots(3, 1, figsize=(16, 10))
    xb, yb = next(iter(train_dataloader))
    xb = xb.to(device)[:8]
    xb = xb * 2. - 1.  # 映射到(-1, 1)
    print('X形状', xb.shape)

    # 显示干净输入
    axs[0].imshow(torchvision.utils.make_grid(xb[:8])[0].detach().cpu(), cmap='Greys')
    axs[0].set_title('干净X')

    # 使用调度器添加噪声
    timesteps = torch.linspace(0, 999, 8).long().to(device)
    noise = torch.randn_like(xb)  # 注意：randn不是rand
    noisy_xb = noise_scheduler.add_noise(xb, noise, timesteps)
    print('噪声X形状', noisy_xb.shape)

    # 显示噪声版本（有和没有裁剪）
    axs[1].imshow(torchvision.utils.make_grid(noisy_xb[:8])[0].detach().cpu().clip(-1, 1), cmap='Greys')
    axs[1].set_title('噪声X（裁剪到(-1, 1)')
    axs[2].imshow(torchvision.utils.make_grid(noisy_xb[:8])[0].detach().cpu(), cmap='Greys')
    axs[2].set_title('噪声X')
    plt.tight_layout()
    plt.show()

    return noisy_xb


def main():
    """主函数，根据命令行参数运行不同的代码模块"""
    parser = argparse.ArgumentParser(description='扩散模型实验')
    parser.add_argument('--setup', action='store_true', help='设置环境')
    parser.add_argument('--load_data', action='store_true', help='加载数据')
    parser.add_argument('--visualize_data', action='store_true', help='可视化数据')
    parser.add_argument('--test_unet', action='store_true', help='测试BasicUNet')
    parser.add_argument('--train_basic_unet', action='store_true', help='训练BasicUNet')
    parser.add_argument('--visualize_predictions', action='store_true', help='可视化预测')
    parser.add_argument('--sampling', action='store_true', help='采样过程')
    parser.add_argument('--generate', action='store_true', help='生成样本')
    parser.add_argument('--create_diffusers', action='store_true', help='创建Diffusers UNet')
    parser.add_argument('--train_diffusers', action='store_true', help='训练Diffusers UNet')
    parser.add_argument('--ddpm_noise', action='store_true', help='可视化DDPM噪声')
    parser.add_argument('--all', action='store_true', help='运行所有步骤')

    args = parser.parse_args()

    device = setup_environment()

    if args.setup or args.all:
        print("环境设置完成")

    if args.load_data or args.all:
        dataset, train_dataloader = load_and_prepare_data()
        print("数据加载完成")

    if args.visualize_data or args.all:
        if 'dataset' not in locals():
            dataset, train_dataloader = load_and_prepare_data()
        x, y = visualize_data_samples(train_dataloader)
        noised_x = visualize_corruption(x)

    if args.test_unet or args.all:
        net, num_params = test_basic_unet()

    if args.train_basic_unet or args.all:
        if 'dataset' not in locals():
            dataset, train_dataloader = load_and_prepare_data()
        net, losses = train_basic_unet(dataset, device)

    if args.visualize_predictions or args.all:
        if 'dataset' not in locals():
            dataset, train_dataloader = load_and_prepare_data()
        if 'net' not in locals():
            net, losses = train_basic_unet(dataset, device)
        preds = visualize_predictions(net, train_dataloader, device)

    if args.sampling or args.all:
        if 'net' not in locals():
            if 'dataset' not in locals():
                dataset, train_dataloader = load_and_prepare_data()
            net, losses = train_basic_unet(dataset, device)
        x, step_history, pred_output_history = sampling_process(net, device)

    if args.generate or args.all:
        if 'net' not in locals():
            if 'dataset' not in locals():
                dataset, train_dataloader = load_and_prepare_data()
            net, losses = train_basic_unet(dataset, device)
        generated_samples = generate_samples(net, device)

    if args.create_diffusers or args.all:
        model, num_params = create_diffusers_unet()

    if args.train_diffusers or args.all:
        if 'dataset' not in locals():
            dataset, train_dataloader = load_and_prepare_data()
        net, losses = train_diffusers_unet(dataset, device)

    if args.ddpm_noise or args.all:
        if 'dataset' not in locals():
            dataset, train_dataloader = load_and_prepare_data()
        noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
        plt.plot(noise_scheduler.alphas_cumprod.cpu() ** 0.5, label=r"${\sqrt{\bar{\alpha}_t}}$")
        plt.plot((1 - noise_scheduler.alphas_cumprod.cpu()) ** 0.5, label=r"$\sqrt{(1 - \bar{\alpha}_t)}$")
        plt.legend(fontsize="x-large")
        plt.title('DDPM调度器参数')
        plt.show()

        noisy_xb = visualize_ddpm_noise(noise_scheduler, train_dataloader, device)


if __name__ == "__main__":
    main()