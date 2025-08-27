# -*- coding: utf-8 -*-
# @project: LargeDemo
# @filename: ch11_autoencoder.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/8/27 9:57

# https://chat.deepseek.com/a/chat/s/85725f19-d5bd-4c5c-a858-9e5c22be6669

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
from torchvision.utils import make_grid
from torchsummary import summary
import numpy as np
import matplotlib.pyplot as plt
import cv2
import requests
from PIL import Image
from sklearn.manifold import TSNE
import os
from tqdm import trange


# ==================== 通用工具函数 ====================
def setup_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def get_mnist_dataloaders(batch_size=256, device='cpu'):
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
        transforms.Lambda(lambda x: x.to(device))
    ])

    trn_ds = datasets.MNIST('/content/', transform=img_transform, train=True, download=True)
    val_ds = datasets.MNIST('/content/', transform=img_transform, train=False, download=True)

    trn_dl = DataLoader(trn_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return trn_dl, val_dl


# ==================== 11.1 简单自编码器 ====================
class AutoEncoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128), nn.ReLU(True),
            nn.Linear(128, 64), nn.ReLU(True),
            nn.Linear(64, latent_dim))
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(True),
            nn.Linear(64, 128), nn.ReLU(True),
            nn.Linear(128, 28 * 28), nn.Tanh())

    def forward(self, x):
        x = x.view(len(x), -1)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(len(x), 1, 28, 28)
        return x


def train_batch_ae(input, model, criterion, optimizer):
    model.train()
    optimizer.zero_grad()
    output = model(input)
    loss = criterion(output, input)
    loss.backward()
    optimizer.step()
    return loss


@torch.no_grad()
def validate_batch_ae(input, model, criterion):
    model.eval()
    output = model(input)
    loss = criterion(output, input)
    return loss


def train_autoencoder(latent_dim, num_epochs=5, batch_size=256):
    device = setup_device()
    trn_dl, val_dl = get_mnist_dataloaders(batch_size, device)

    model = AutoEncoder(latent_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)

    for epoch in range(num_epochs):
        # 训练循环
        model.train()
        for data, _ in trn_dl:
            loss = train_batch_ae(data, model, criterion, optimizer)

        # 验证循环
        model.eval()
        val_loss = 0
        for data, _ in val_dl:
            val_loss += validate_batch_ae(data, model, criterion).item()
        val_loss /= len(val_dl)
        print(f'Epoch {epoch + 1}, Val Loss: {val_loss:.4f}')

    return model


def run_autoencoder_experiment():
    print("训练不同潜在维度的自编码器...")
    latent_dims = [50, 2, 3, 5, 10]
    models = []

    for dim in latent_dims:
        print(f"训练潜在维度: {dim}")
        model = train_autoencoder(dim)
        models.append(model)

    # 可视化结果
    _, val_ds = get_mnist_dataloaders(256, setup_device())
    for _ in range(3):
        idx = np.random.randint(len(val_ds))
        im, _ = val_ds[idx]

        fig, axes = plt.subplots(1, len(models) + 1, figsize=(15, 3))
        axes[0].imshow(im[0].cpu(), cmap='gray')
        axes[0].set_title('Original')
        axes[0].axis('off')

        for i, model in enumerate(models):
            with torch.no_grad():
                reconstructed = model(im.unsqueeze(0).to(setup_device()))[0]
            axes[i + 1].imshow(reconstructed[0].cpu(), cmap='gray')
            axes[i + 1].set_title(f'Latent: {model.latent_dim}')
            axes[i + 1].axis('off')

        plt.tight_layout()
        plt.show()


# ==================== 11.2 卷积自编码器 ====================
class ConvAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2), nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 5, stride=3, padding=1), nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, 2, stride=2, padding=1), nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def train_conv_autoencoder(num_epochs=5):
    device = setup_device()
    trn_dl, val_dl = get_mnist_dataloaders(128, device)

    model = ConvAutoEncoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)

    for epoch in range(num_epochs):
        # 训练
        model.train()
        train_loss = 0
        for data, _ in trn_dl:
            loss = train_batch_ae(data, model, criterion, optimizer)
            train_loss += loss.item()

        # 验证
        model.eval()
        val_loss = 0
        for data, _ in val_dl:
            val_loss += validate_batch_ae(data, model, criterion).item()

        print(f'Epoch {epoch + 1}, Train Loss: {train_loss / len(trn_dl):.4f}, Val Loss: {val_loss / len(val_dl):.4f}')

    return model


def run_conv_autoencoder():
    print("训练卷积自编码器...")
    model = train_conv_autoencoder()

    # 可视化结果
    _, val_ds = get_mnist_dataloaders(128, setup_device())
    for _ in range(3):
        idx = np.random.randint(len(val_ds))
        im, _ = val_ds[idx]

        with torch.no_grad():
            reconstructed = model(im.unsqueeze(0).to(setup_device()))[0]

        fig, axes = plt.subplots(1, 2, figsize=(6, 3))
        axes[0].imshow(im[0].cpu(), cmap='gray')
        axes[0].set_title('Original')
        axes[0].axis('off')

        axes[1].imshow(reconstructed[0].cpu(), cmap='gray')
        axes[1].set_title('Reconstructed')
        axes[1].axis('off')

        plt.tight_layout()
        plt.show()

    # 潜在空间可视化
    device = setup_device()
    _, val_dl = get_mnist_dataloaders(128, device)

    latent_vectors = []
    classes = []

    for im, clss in val_dl:
        latent_vectors.append(model.encoder(im).view(len(im), -1))
        classes.extend(clss.cpu().numpy())

    latent_vectors = torch.cat(latent_vectors).cpu().detach().numpy()

    # t-SNE降维
    tsne = TSNE(2)
    clustered = tsne.fit_transform(latent_vectors)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(clustered[:, 0], clustered[:, 1], c=classes, cmap='tab10', alpha=0.6)
    plt.colorbar(scatter)
    plt.title('t-SNE Visualization of Latent Space')
    plt.show()


# ==================== 11.3 变分自编码器 ====================
class VAE(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim):
        super(VAE, self).__init__()
        self.d1 = nn.Linear(x_dim, h_dim1)
        self.d2 = nn.Linear(h_dim1, h_dim2)
        self.d31 = nn.Linear(h_dim2, z_dim)  # 均值
        self.d32 = nn.Linear(h_dim2, z_dim)  # 对数方差
        self.d4 = nn.Linear(z_dim, h_dim2)
        self.d5 = nn.Linear(h_dim2, h_dim1)
        self.d6 = nn.Linear(h_dim1, x_dim)

    def encoder(self, x):
        h = F.relu(self.d1(x))
        h = F.relu(self.d2(h))
        return self.d31(h), self.d32(h)

    def sampling(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    def decoder(self, z):
        h = F.relu(self.d4(z))
        h = F.relu(self.d5(h))
        return torch.sigmoid(self.d6(h))

    def forward(self, x):
        mean, log_var = self.encoder(x.view(-1, 784))
        z = self.sampling(mean, log_var)
        return self.decoder(z), mean, log_var


def vae_loss_function(recon_x, x, mean, log_var):
    RECON = F.mse_loss(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return RECON + KLD, RECON, KLD


def train_vae(num_epochs=10):
    device = setup_device()

    # 数据加载
    train_dataset = datasets.MNIST(root='MNIST/', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(root='MNIST/', train=False, transform=transforms.ToTensor(), download=True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

    # 模型初始化
    vae = VAE(x_dim=784, h_dim1=512, h_dim2=256, z_dim=50).to(device)
    optimizer = optim.AdamW(vae.parameters(), lr=1e-3)

    # 训练循环
    for epoch in range(num_epochs):
        # 训练
        vae.train()
        train_loss = 0
        for data, _ in train_loader:
            data = data.to(device)
            optimizer.zero_grad()

            recon_batch, mean, log_var = vae(data)
            loss, recon, kld = vae_loss_function(recon_batch, data, mean, log_var)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 测试
        vae.eval()
        test_loss = 0
        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(device)
                recon, mean, log_var = vae(data)
                loss, recon, kld = vae_loss_function(recon, data, mean, log_var)
                test_loss += loss.item()

        print(f'Epoch {epoch + 1}, Train Loss: {train_loss / len(train_loader.dataset):.4f}, '
              f'Test Loss: {test_loss / len(test_loader.dataset):.4f}')

        # 生成样本
        with torch.no_grad():
            z = torch.randn(64, 50).to(device)
            sample = vae.decoder(z).cpu()
            sample = sample.view(64, 1, 28, 28)
            grid = make_grid(sample, nrow=8, normalize=True)
            plt.figure(figsize=(10, 10))
            plt.imshow(grid.permute(1, 2, 0))
            plt.title(f'Generated Samples - Epoch {epoch + 1}')
            plt.axis('off')
            plt.show()

    return vae


def run_vae():
    print("训练变分自编码器...")
    vae = train_vae()


# ==================== 11.4 对抗攻击 ====================
def setup_pretrained_model():
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model = model.eval()
    return model


def get_imagenet_classes():
    url = 'https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt'
    response = requests.get(url)
    image_net_classes = response.text
    image_net_ids = eval(image_net_classes)
    return {i: j for j, i in image_net_ids.items()}


def load_and_preprocess_image(url):
    response = requests.get(url, stream=True)
    original_image = Image.open(response.raw).convert('RGB')
    original_image = np.array(original_image)
    return torch.Tensor(original_image)


def image2tensor(input, device='cpu'):
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    x = normalize(input.clone().permute(2, 0, 1) / 255.)[None]
    return x.to(device)


def tensor2image(input):
    denormalize = transforms.Normalize([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                                       [1 / 0.229, 1 / 0.224, 1 / 0.225])
    x = (denormalize(input[0].clone()).permute(1, 2, 0) * 255.).type(torch.uint8)
    return x


def predict_on_image(model, input, image_net_ids):
    model.eval()
    input_tensor = image2tensor(input, next(model.parameters()).device)
    pred = model(input_tensor)
    pred = F.softmax(pred, dim=-1)[0]
    prob, clss = torch.max(pred, 0)
    clss = image_net_ids[clss.item()]
    print(f'PREDICTION: `{clss}` @ {prob.item():.4f}')
    return clss, prob.item()


def adversarial_attack(image, model, target, epsilon=1e-6):
    input = image2tensor(image, next(model.parameters()).device)
    input.requires_grad = True

    pred = model(input)
    loss = nn.CrossEntropyLoss()(pred, target)
    loss.backward()

    output = input - epsilon * input.grad.sign()
    output = tensor2image(output)

    return output.detach()


def run_adversarial_attack():
    print("执行对抗攻击...")
    device = setup_device()
    model = setup_pretrained_model().to(device)
    image_net_classes = get_imagenet_classes()

    # 加载图像
    url = 'https://lionsvalley.co.za/wp-content/uploads/2015/11/african-elephant-square.jpg'
    original_image = load_and_preprocess_image(url)

    # 原始预测
    print("原始图像预测:")
    predict_on_image(model, original_image, image_net_classes)

    # 对抗攻击
    desired_targets = ['lemon', 'comic book', 'sax, saxophone']
    modified_images = []

    for target_name in desired_targets:
        target = torch.tensor([image_net_classes[target_name]]).to(device)
        image_to_attack = original_image.clone()

        for _ in trange(10, desc=f"Attacking for {target_name}"):
            image_to_attack = adversarial_attack(image_to_attack, model, target)

        modified_images.append(image_to_attack)
        print(f"\n攻击目标: {target_name}")
        predict_on_image(model, image_to_attack, image_net_classes)

    # 可视化结果
    fig, axes = plt.subplots(1, len(modified_images) + 1, figsize=(15, 5))

    axes[0].imshow(original_image.numpy().astype(np.uint8))
    axes[0].set_title('Original')
    axes[0].axis('off')

    for i, img in enumerate(modified_images):
        axes[i + 1].imshow(img.cpu().numpy())
        axes[i + 1].set_title(f'Target: {desired_targets[i]}')
        axes[i + 1].axis('off')

    plt.tight_layout()
    plt.show()


# ==================== 11.5 神经风格迁移 ====================
class GramMatrix(nn.Module):
    def forward(self, input):
        b, c, h, w = input.size()
        feat = input.view(b, c, h * w)
        G = feat @ feat.transpose(1, 2)
        G.div_(h * w)
        return G


class GramMSELoss(nn.Module):
    def forward(self, input, target):
        out = F.mse_loss(GramMatrix()(input), target)
        return out


class VGG19Modified(nn.Module):
    def __init__(self):
        super().__init__()
        features = list(models.vgg19(pretrained=True).features)
        self.features = nn.ModuleList(features).eval()

    def forward(self, x, layers=[]):
        order = np.argsort(layers)
        _results, results = [], []

        for ix, model in enumerate(self.features):
            x = model(x)
            if ix in layers:
                _results.append(x)

        for o in order:
            results.append(_results[o])

        return results if layers else x


def download_style_transfer_images():
    style_url = "https://www.dropbox.com/s/1svdliljyo0a98v/style_image.png"
    content_url = "https://www.dropbox.com/s/z1y0fy2r6z6m6py/60.jpg"

    os.makedirs("style_transfer", exist_ok=True)

    for url, filename in zip([style_url, content_url], ["style_image.png", "content_image.jpg"]):
        if not os.path.exists(f"style_transfer/{filename}"):
            response = requests.get(url)
            with open(f"style_transfer/{filename}", "wb") as f:
                f.write(response.content)

    return "style_transfer/style_image.png", "style_transfer/content_image.jpg"


def run_style_transfer():
    print("执行神经风格迁移...")
    device = setup_device()

    # 下载图像
    style_path, content_path = download_style_transfer_images()

    # 预处理
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Lambda(lambda x: x.mul_(255))
    ])

    postprocess = transforms.Compose([
        transforms.Lambda(lambda x: x.mul_(1. / 255)),
        transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                             std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
    ])

    # 加载图像
    style_image = Image.open(style_path).resize((512, 512)).convert('RGB')
    content_image = Image.open(content_path).resize((512, 512)).convert('RGB')

    style_tensor = preprocess(style_image).to(device)[None]
    content_tensor = preprocess(content_image).to(device)[None]

    # 初始化模型
    vgg = VGG19Modified().to(device)

    # 定义目标层和权重
    style_layers = [0, 5, 10, 19, 28]
    content_layers = [21]
    loss_layers = style_layers + content_layers

    style_weights = [1000 / n ** 2 for n in [64, 128, 256, 512, 512]]
    content_weights = [1]
    weights = style_weights + content_weights

    loss_fns = [GramMSELoss()] * len(style_layers) + [nn.MSELoss()] * len(content_layers)
    loss_fns = [loss_fn.to(device) for loss_fn in loss_fns]

    # 计算目标特征
    style_targets = [GramMatrix()(A).detach() for A in vgg(style_tensor, style_layers)]
    content_targets = [A.detach() for A in vgg(content_tensor, content_layers)]
    targets = style_targets + content_targets

    # 初始化优化图像
    opt_img = content_tensor.data.clone()
    opt_img.requires_grad = True

    # 优化
    optimizer = optim.LBFGS([opt_img])
    max_iters = 100  # 减少迭代次数以加快演示

    losses = []
    iters = 0

    while iters < max_iters:
        def closure():
            nonlocal iters
            iters += 1
            optimizer.zero_grad()

            out = vgg(opt_img, loss_layers)
            layer_losses = [weights[i] * loss_fns[i](A, targets[i]) for i, A in enumerate(out)]
            loss = sum(layer_losses)

            loss.backward()
            losses.append(loss.item())

            if iters % 10 == 0:
                print(f'Iteration {iters}, Loss: {loss.item():.4f}')

            return loss

        optimizer.step(closure)

    # 后处理并显示结果
    with torch.no_grad():
        out_img = postprocess(opt_img[0]).permute(1, 2, 0).cpu()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(style_image)
    axes[0].set_title('Style Image')
    axes[0].axis('off')

    axes[1].imshow(content_image)
    axes[1].set_title('Content Image')
    axes[1].axis('off')

    axes[2].imshow(out_img)
    axes[2].set_title('Style Transfer Result')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('Style Transfer Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.show()


# ==================== 11.6 深度伪造生成 ====================
# 注意：此部分代码需要额外的文件和数据集，这里只提供框架
def run_deep_fakes():
    print("深度伪造生成需要额外的数据集和文件，此处只提供框架")
    print("请参考原始代码获取完整实现")


# ==================== 主函数 ====================
def main():
    import argparse

    parser = argparse.ArgumentParser(description='运行不同的计算机视觉模块')
    parser.add_argument('--module', type=int, choices=[1, 2, 3, 4, 5, 6],
                        help='选择要运行的模块: 1-自编码器, 2-卷积自编码器, 3-VAE, 4-对抗攻击, 5-风格迁移, 6-深度伪造')

    args = parser.parse_args()

    if args.module == 1:
        run_autoencoder_experiment()
    elif args.module == 2:
        run_conv_autoencoder()
    elif args.module == 3:
        run_vae()
    elif args.module == 4:
        run_adversarial_attack()
    elif args.module == 5:
        run_style_transfer()
    elif args.module == 6:
        run_deep_fakes()
    else:
        print("请使用 --module 参数选择要运行的模块 (1-6)")


if __name__ == "__main__":
    main()