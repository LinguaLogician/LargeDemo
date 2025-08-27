# -*- coding: utf-8 -*-
# @project: LargeDemo
# @filename: ch12_gan_base.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/8/27 9:58
# https://chat.deepseek.com/a/chat/s/25105a3b-b40e-42bd-bd21-24e6cf0d1f23
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.datasets import MNIST
import torchvision.utils as vutils
import cv2
import numpy as np
from PIL import Image
import os
import glob
from torch_snippets import *
import warnings

warnings.filterwarnings('ignore')

# 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"


def weights_init(m):
    """初始化网络权重"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def download_and_extract_data(url, extract_to='.'):
    """下载并解压数据"""
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)

    zip_name = os.path.join(extract_to, os.path.basename(url))

    # 下载文件
    if not os.path.exists(zip_name):
        os.system(f"wget {url} -O {zip_name}")

    # 解压文件
    os.system(f"unzip -q {zip_name} -d {extract_to}")


def detect_and_crop_faces(input_folder, output_folder, cascade_file='haarcascade_frontalface_default.xml'):
    """检测并裁剪图像中的人脸"""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascade_file)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    images = glob.glob(os.path.join(input_folder, '*.jpg'))

    for i, img_path in enumerate(images):
        img = read(img_path, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            img_cropped = img[y:(y + h), x:(x + w), :]

        output_path = os.path.join(output_folder, f'{i}.jpg')
        cv2.imwrite(output_path, cv2.cvtColor(img_cropped, cv2.COLOR_RGB2BGR))


class FacesDataset(Dataset):
    """人脸数据集类"""

    def __init__(self, folders, transform=None, conditional=False):
        super().__init__()
        self.conditional = conditional
        self.transform = transform

        if isinstance(folders, list):
            self.images = []
            for i, folder in enumerate(folders):
                folder_images = sorted(glob.glob(os.path.join(folder, '*.jpg')))
                self.images.extend([(img, i) for img in folder_images] if conditional else folder_images)
        else:
            self.images = sorted(glob.glob(os.path.join(folders, '*.jpg')))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.conditional:
            image_path, label = self.images[idx]
            image = Image.open(image_path)
        else:
            image_path = self.images[idx]
            image = Image.open(image_path)
            label = 0  # 非条件GAN不需要标签

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label).long()


class ConditionalDiscriminator(nn.Module):
    """条件判别器"""

    def __init__(self, emb_size=32):
        super(ConditionalDiscriminator, self).__init__()
        self.emb_size = emb_size
        self.label_embeddings = nn.Embedding(2, self.emb_size)

        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64 * 8, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten()
        )

        self.model2 = nn.Sequential(
            nn.Linear(288, 100),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(100, 1),
            nn.Sigmoid()
        )

        self.apply(weights_init)

    def forward(self, input, labels):
        x = self.model(input)
        y = self.label_embeddings(labels)
        input_combined = torch.cat([x, y], 1)
        return self.model2(input_combined)


class ConditionalGenerator(nn.Module):
    """条件生成器"""

    def __init__(self, emb_size=32):
        super(ConditionalGenerator, self).__init__()
        self.emb_size = emb_size
        self.label_embeddings = nn.Embedding(2, self.emb_size)

        self.model = nn.Sequential(
            nn.ConvTranspose2d(100 + self.emb_size, 64 * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(64 * 2, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

        self.apply(weights_init)

    def forward(self, input_noise, labels):
        label_embeddings = self.label_embeddings(labels).view(len(labels), self.emb_size, 1, 1)
        input_combined = torch.cat([input_noise, label_embeddings], 1)
        return self.model(input_combined)


class DCGANDiscriminator(nn.Module):
    """DCGAN判别器"""

    def __init__(self):
        super(DCGANDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64 * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        self.apply(weights_init)

    def forward(self, input):
        return self.model(input)


class DCGANGenerator(nn.Module):
    """DCGAN生成器"""

    def __init__(self):
        super(DCGANGenerator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(100, 64 * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(64 * 2, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        self.apply(weights_init)

    def forward(self, input):
        return self.model(input)


class MNISTDiscriminator(nn.Module):
    """MNIST判别器"""

    def __init__(self):
        super(MNISTDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


class MNISTGenerator(nn.Module):
    """MNIST生成器"""

    def __init__(self):
        super(MNISTGenerator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 784),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)


def discriminator_train_step(discriminator, d_optimizer, loss_fn, real_data, real_labels=None, fake_data=None,
                             fake_labels=None, conditional=False):
    """判别器训练步骤"""
    d_optimizer.zero_grad()

    if conditional:
        prediction_real = discriminator(real_data, real_labels)
        prediction_fake = discriminator(fake_data, fake_labels)
    else:
        prediction_real = discriminator(real_data)
        prediction_fake = discriminator(fake_data)

    error_real = loss_fn(prediction_real.squeeze(), torch.ones(len(real_data)).to(device))
    error_fake = loss_fn(prediction_fake.squeeze(), torch.zeros(len(fake_data)).to(device))

    error_real.backward()
    error_fake.backward()
    d_optimizer.step()

    return error_real + error_fake


def generator_train_step(generator, discriminator, g_optimizer, loss_fn, fake_data, fake_labels=None,
                         conditional=False):
    """生成器训练步骤"""
    g_optimizer.zero_grad()

    if conditional:
        prediction = discriminator(fake_data, fake_labels)
    else:
        prediction = discriminator(fake_data)

    error = loss_fn(prediction.squeeze(), torch.ones(len(fake_data)).to(device))
    error.backward()
    g_optimizer.step()

    return error


def train_gan(generator, discriminator, dataloader, n_epochs, gan_type='conditional', lr=0.0002, betas=(0.5, 0.999)):
    """训练GAN模型"""
    # 初始化优化器和损失函数
    d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=betas)
    g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=betas)
    loss_fn = nn.BCELoss()

    # 固定噪声用于生成样本
    if gan_type == 'conditional':
        fixed_noise = torch.randn(64, 100, 1, 1, device=device)
        fixed_fake_labels = torch.LongTensor([0] * 32 + [1] * 32).to(device)
    elif gan_type == 'dcgan':
        fixed_noise = torch.randn(64, 100, 1, 1, device=device)
    else:  # mnist
        fixed_noise = torch.randn(64, 100, device=device)

    log = Report(n_epochs)
    img_list = []

    for epoch in range(n_epochs):
        N = len(dataloader)
        for i, data in enumerate(dataloader):
            if gan_type == 'conditional':
                real_data, real_labels = data
                real_data, real_labels = real_data.to(device), real_labels.to(device)

                # 训练判别器
                fake_labels = torch.LongTensor(np.random.randint(0, 2, len(real_data))).to(device)
                fake_data = generator(torch.randn(len(real_data), 100, 1, 1, device=device), fake_labels)
                fake_data = fake_data.detach()
                d_loss = discriminator_train_step(discriminator, d_optimizer, loss_fn,
                                                  real_data, real_labels, fake_data, fake_labels, conditional=True)

                # 训练生成器
                fake_labels = torch.LongTensor(np.random.randint(0, 2, len(real_data))).to(device)
                fake_data = generator(torch.randn(len(real_data), 100, 1, 1, device=device), fake_labels)
                g_loss = generator_train_step(generator, discriminator, g_optimizer, loss_fn,
                                              fake_data, fake_labels, conditional=True)

            elif gan_type == 'dcgan':
                real_data = data[0].to(device)

                # 训练判别器
                fake_data = generator(torch.randn(len(real_data), 100, 1, 1, device=device))
                fake_data = fake_data.detach()
                d_loss = discriminator_train_step(discriminator, d_optimizer, loss_fn, real_data, fake_data=fake_data)

                # 训练生成器
                fake_data = generator(torch.randn(len(real_data), 100, 1, 1, device=device))
                g_loss = generator_train_step(generator, discriminator, g_optimizer, loss_fn, fake_data)

            else:  # mnist
                real_data = data[0].view(len(data[0]), -1).to(device)

                # 训练判别器
                fake_data = generator(torch.randn(len(real_data), 100, device=device))
                fake_data = fake_data.detach()
                d_loss = discriminator_train_step(discriminator, d_optimizer, loss_fn, real_data, fake_data=fake_data)

                # 训练生成器
                fake_data = generator(torch.randn(len(real_data), 100, device=device))
                g_loss = generator_train_step(generator, discriminator, g_optimizer, loss_fn, fake_data)

            # 记录损失
            pos = epoch + (1 + i) / N
            log.record(pos, d_loss=d_loss.item(), g_loss=g_loss.item(), end='\r')

        log.report_avgs(epoch + 1)

        # 生成样本图像
        with torch.no_grad():
            if gan_type == 'conditional':
                fake = generator(fixed_noise, fixed_fake_labels).detach().cpu()
                imgs = vutils.make_grid(fake, padding=2, normalize=True).permute(1, 2, 0)
            elif gan_type == 'dcgan':
                fake = generator(fixed_noise).detach().cpu()
                imgs = vutils.make_grid(fake, padding=2, normalize=True).permute(1, 2, 0)
            else:  # mnist
                fake = generator(fixed_noise).data.cpu().view(64, 1, 28, 28)
                imgs = make_grid(fake, nrow=8, normalize=True).permute(1, 2, 0)

            img_list.append(imgs)
            show(imgs, sz=10 if gan_type != 'mnist' else 5, title=f'Epoch {epoch + 1}')

    log.plot_epochs(['d_loss', 'g_loss'])
    return img_list


def run_conditional_face_gan():
    """运行条件GAN生成人脸"""
    print("准备条件GAN人脸生成...")

    # 下载数据
    url = "https://www.dropbox.com/s/rbajpdlh7efkdo1/male_female_face_images.zip"
    download_and_extract_data(url)

    # 检测并裁剪人脸
    detect_and_crop_faces('/content/females', 'cropped_faces_female')
    detect_and_crop_faces('/content/males', 'cropped_faces_male')

    # 定义数据转换
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # 创建数据集和数据加载器
    dataset = FacesDataset(['cropped_faces_female', 'cropped_faces_male'], transform=transform, conditional=True)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=8)

    # 创建模型
    discriminator = ConditionalDiscriminator().to(device)
    generator = ConditionalGenerator().to(device)

    # 训练模型
    img_list = train_gan(generator, discriminator, dataloader, n_epochs=25, gan_type='conditional')

    print("条件GAN人脸生成完成!")


def run_dcgan_face_generation():
    """运行DCGAN生成人脸"""
    print("准备DCGAN人脸生成...")

    # 下载数据
    url = "https://www.dropbox.com/s/rbajpdlh7efkdo1/male_female_face_images.zip"
    download_and_extract_data(url)

    # 检测并裁剪人脸
    detect_and_crop_faces('/content/females', 'cropped_faces')
    detect_and_crop_faces('/content/males', 'cropped_faces')

    # 定义数据转换
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 创建数据集和数据加载器
    dataset = FacesDataset('cropped_faces', transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=8)

    # 创建模型
    discriminator = DCGANDiscriminator().to(device)
    generator = DCGANGenerator().to(device)

    # 训练模型
    img_list = train_gan(generator, discriminator, dataloader, n_epochs=25, gan_type='dcgan')

    print("DCGAN人脸生成完成!")


def run_mnist_gan():
    """运行MNIST GAN生成手写数字"""
    print("准备MNIST GAN手写数字生成...")

    # 定义数据转换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])

    # 创建数据集和数据加载器
    dataset = MNIST('~/data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, drop_last=True)

    # 创建模型
    discriminator = MNISTDiscriminator().to(device)
    generator = MNISTGenerator().to(device)

    # 训练模型
    img_list = train_gan(generator, discriminator, dataloader, n_epochs=200, gan_type='mnist')

    print("MNIST GAN手写数字生成完成!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='运行不同类型的GAN模型')
    parser.add_argument('--model', type=str, choices=['conditional', 'dcgan', 'mnist', 'all'],
                        default='all', help='选择要运行的模型类型')

    args = parser.parse_args()

    if args.model == 'conditional' or args.model == 'all':
        run_conditional_face_gan()

    if args.model == 'dcgan' or args.model == 'all':
        run_dcgan_face_generation()

    if args.model == 'mnist' or args.model == 'all':
        run_mnist_gan()