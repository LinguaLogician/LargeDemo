# -*- coding: utf-8 -*-
# @project: LargeDemo
# @filename: ch13_gan_advanced.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/8/27 9:58
# https://chat.deepseek.com/a/chat/s/f831c9aa-828f-40a1-815f-da2f9a885f56
import os
import glob
import itertools
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split

# 设置设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ==================== StyleGAN2 相关代码 ====================
def setup_stylegan_environment():
    """设置StyleGAN2环境"""
    if not os.path.exists('pytorch_stylegan_encoder'):
        os.system('git clone https://github.com/jacobhallberg/pytorch_stylegan_encoder.git')
        os.chdir('pytorch_stylegan_encoder')
        os.system('git submodule update --init --recursive')
        os.system(
            'wget -q https://github.com/jacobhallberg/pytorch_stylegan_encoder/releases/download/v1.0/trained_models.zip')
        os.system('unzip -q trained_models.zip')
        os.system('rm trained_models.zip')
        os.system('pip install -qU torch_snippets')
        os.system('mv trained_models/stylegan_ffhq.pth InterFaceGAN/models/pretrain')
    else:
        os.chdir('pytorch_stylegan_encoder')


def load_stylegan_models():
    """加载StyleGAN2模型"""
    from InterFaceGAN.models.stylegan_generator import StyleGANGenerator
    from models.latent_optimizer import PostSynthesisProcessing

    synthesizer = StyleGANGenerator("stylegan_ffhq").model.synthesis
    mapper = StyleGANGenerator("stylegan_ffhq").model.mapping
    trunc = StyleGANGenerator("stylegan_ffhq").model.truncation

    post_processing = PostSynthesisProcessing()
    post_process = lambda image: post_processing(image).detach().cpu().numpy().astype(np.uint8)[0]

    def latent2image(latent):
        img = post_process(synthesizer(latent))
        img = img.transpose(1, 2, 0)
        return img

    return synthesizer, mapper, trunc, latent2image


def encode_image_stylegan(image_path):
    """使用StyleGAN编码图像"""
    # 下载并准备图像
    os.system(f'wget -q {image_path} -O MyImage.jpg')
    os.system('git clone https://github.com/sizhky/stylegan-encoder.git')
    os.system('mkdir -p stylegan-encoder/raw_images')
    os.system('mkdir -p stylegan-encoder/aligned_images')
    os.system('mv MyImage.jpg stylegan-encoder/raw_images')

    # 对齐图像
    os.system('python stylegan-encoder/align_images.py stylegan-encoder/raw_images/ stylegan-encoder/aligned_images/')
    os.system('mv stylegan-encoder/aligned_images/* ./MyImage.jpg')

    # 编码图像
    os.system(
        'python encode_image.py ./MyImage.jpg pred_dlatents_myImage.npy --use_latent_finder true --image_to_latent_path ./trained_models/image_to_latent.pt')

    # 加载预测的潜在向量
    pred_dlatents = np.load('pred_dlatents_myImage.npy')
    pred_dlatent = torch.from_numpy(pred_dlatents).float().cuda()

    return pred_dlatent


def stylegan_latent_transfer(my_latents, generated_latents, idxs_to_swap):
    """执行潜在向量交换"""
    x = my_latents.clone()
    x[:, idxs_to_swap] = generated_latents[:, idxs_to_swap]
    return latent2image(x.float().cuda())


def stylegan_edit_image(latent_path, boundary_path, output_dir):
    """使用InterfaceGAN编辑图像"""
    os.system(
        f'python InterFaceGAN/edit.py -m stylegan_ffhq -o {output_dir} -b {boundary_path} -i {latent_path} -s WP --steps 20')
    return glob.glob(f'{output_dir}/*.jpg')


# ==================== CycleGAN 相关代码 ====================
def setup_cyclegan_data():
    """设置CycleGAN数据"""
    os.system('wget -q https://www.dropbox.com/s/2xltmolfbfharri/apples_oranges.zip')
    os.system('unzip -q apples_oranges.zip')


def weights_init_normal(m):
    """权重初始化"""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class ResidualBlock(nn.Module):
    """残差块"""

    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.block(x)


class GeneratorResNet(nn.Module):
    """CycleGAN生成器"""

    def __init__(self, num_residual_blocks=9):
        super(GeneratorResNet, self).__init__()
        out_features = 64
        channels = 3
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(channels, out_features, 7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        in_features = out_features

        # 下采样
        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # 残差块
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_features)]

        # 上采样
        for _ in range(2):
            out_features //= 2
            model += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # 输出层
        model += [nn.ReflectionPad2d(channels), nn.Conv2d(out_features, channels, 7), nn.Tanh()]
        self.model = nn.Sequential(*model)
        self.apply(weights_init_normal)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    """CycleGAN判别器"""

    def __init__(self):
        super(Discriminator, self).__init__()
        channels, height, width = 3, 256, 256

        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )
        self.apply(weights_init_normal)

    def forward(self, img):
        return self.model(img)


class CycleGANDataset(Dataset):
    """CycleGAN数据集"""

    def __init__(self, apples, oranges, image_size=256):
        self.apples = glob.glob(f'{apples}/*')
        self.oranges = glob.glob(f'{oranges}/*')
        self.transform = transforms.Compose([
            transforms.Resize(int(image_size * 1.33)),
            transforms.RandomCrop((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __getitem__(self, ix):
        apple = self.apples[ix % len(self.apples)]
        orange = np.random.choice(self.oranges)
        apple = Image.open(apple).convert('RGB')
        orange = Image.open(orange).convert('RGB')
        return apple, orange

    def __len__(self):
        return max(len(self.apples), len(self.oranges))

    def collate_fn(self, batch):
        srcs, trgs = list(zip(*batch))
        srcs = torch.cat([self.transform(img)[None] for img in srcs], 0).to(device).float()
        trgs = torch.cat([self.transform(img)[None] for img in trgs], 0).to(device).float()
        return srcs.to(device), trgs.to(device)


def train_cyclegan(epochs=10):
    """训练CycleGAN"""
    # 设置数据
    setup_cyclegan_data()
    trn_ds = CycleGANDataset('apples_train', 'oranges_train')
    val_ds = CycleGANDataset('apples_test', 'oranges_test')

    trn_dl = DataLoader(trn_ds, batch_size=1, shuffle=True, collate_fn=trn_ds.collate_fn)
    val_dl = DataLoader(val_ds, batch_size=5, shuffle=True, collate_fn=val_ds.collate_fn)

    # 初始化模型
    G_AB = GeneratorResNet().to(device)
    G_BA = GeneratorResNet().to(device)
    D_A = Discriminator().to(device)
    D_B = Discriminator().to(device)

    # 损失函数和优化器
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()

    optimizer_G = torch.optim.Adam(
        itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=0.0002, betas=(0.5, 0.999)
    )
    optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=0.0002, betas=(0.5, 0.999))

    lambda_cyc, lambda_id = 10.0, 5.0

    # 训练循环
    for epoch in range(epochs):
        for bx, batch in enumerate(trn_dl):
            real_A, real_B = batch

            # 训练生成器
            optimizer_G.zero_grad()

            # 身份损失
            loss_id_A = criterion_identity(G_BA(real_A), real_A)
            loss_id_B = criterion_identity(G_AB(real_B), real_B)
            loss_identity = (loss_id_A + loss_id_B) / 2

            # GAN损失
            fake_B = G_AB(real_A)
            loss_GAN_AB = criterion_GAN(D_B(fake_B), torch.ones((len(real_A), 1, 16, 16)).to(device))
            fake_A = G_BA(real_B)
            loss_GAN_BA = criterion_GAN(D_A(fake_A), torch.ones((len(real_A), 1, 16, 16)).to(device))
            loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

            # 循环一致性损失
            recov_A = G_BA(fake_B)
            loss_cycle_A = criterion_cycle(recov_A, real_A)
            recov_B = G_AB(fake_A)
            loss_cycle_B = criterion_cycle(recov_B, real_B)
            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

            # 总损失
            loss_G = loss_GAN + lambda_cyc * loss_cycle + lambda_id * loss_identity
            loss_G.backward()
            optimizer_G.step()

            # 训练判别器A
            optimizer_D_A.zero_grad()
            loss_real_A = criterion_GAN(D_A(real_A), torch.ones((len(real_A), 1, 16, 16)).to(device))
            loss_fake_A = criterion_GAN(D_A(fake_A.detach()), torch.zeros((len(real_A), 1, 16, 16)).to(device))
            loss_D_A = (loss_real_A + loss_fake_A) / 2
            loss_D_A.backward()
            optimizer_D_A.step()

            # 训练判别器B
            optimizer_D_B.zero_grad()
            loss_real_B = criterion_GAN(D_B(real_B), torch.ones((len(real_B), 1, 16, 16)).to(device))
            loss_fake_B = criterion_GAN(D_B(fake_B.detach()), torch.zeros((len(real_B), 1, 16, 16)).to(device))
            loss_D_B = (loss_real_B + loss_fake_B) / 2
            loss_D_B.backward()
            optimizer_D_B.step()

            if bx % 100 == 0:
                print(
                    f'Epoch: {epoch}, Batch: {bx}, Loss_G: {loss_G.item()}, Loss_D: {(loss_D_A + loss_D_B).item() / 2}')

    return G_AB, G_BA


# ==================== SRGAN 相关代码 ====================
def setup_srgan_model():
    """设置SRGAN模型"""
    if not os.path.exists('srgan.pth.tar'):
        os.system('pip install -q torch_snippets')
        os.system(
            'wget -q https://raw.githubusercontent.com/sizhky/a-PyTorch-Tutorial-to-Super-Resolution/master/models.py -O models.py')

        # 这里需要Google Drive认证代码，但为了简化，我们假设模型已下载
        # 实际使用时需要根据环境调整

    model = torch.load('srgan.pth.tar', map_location='cpu')['generator'].to(device)
    model.eval()
    return model


def super_resolve_image(model, image_path):
    """使用SRGAN进行超分辨率"""
    # 下载图像
    os.system(f'wget https://www.dropbox.com/s/nmzwu68nrl9j0lf/{image_path}')

    # 预处理
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.Lambda(lambda x: x.to(device))
    ])

    # 后处理
    postprocess = transforms.Compose([
        transforms.Lambda(lambda x: (x.cpu().detach() + 1) / 2),
        transforms.ToPILImage()
    ])

    # 读取和处理图像
    image = Image.open(image_path).convert('RGB')
    image = image.resize((130, 90))  # 缩小图像
    im = preprocess(image)

    # 生成超分辨率图像
    sr = model(im[None])[0]
    sr = postprocess(sr)

    return image, sr


# ==================== Pix2Pix 相关代码 ====================
def setup_pix2pix_data():
    """设置Pix2Pix数据"""
    if not os.path.exists('ShoeV2_photo'):
        os.system('wget https://www.dropbox.com/s/g6b6gtvmdu0h77x/ShoeV2_photo.zip')
        os.system('unzip -q ShoeV2_photo.zip')


def detect_edges(img):
    """检测图像边缘"""
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_gray = cv2.bilateralFilter(img_gray, 5, 50, 50)
    img_gray_edges = cv2.Canny(img_gray, 45, 100)
    img_gray_edges = cv2.bitwise_not(img_gray_edges)  # 反转黑白
    img_edges = cv2.cvtColor(img_gray_edges, cv2.COLOR_GRAY2RGB)
    return img_edges


class ShoesData(Dataset):
    """Pix2Pix鞋子数据集"""

    def __init__(self, items, image_size=256):
        self.items = items
        self.image_size = image_size
        self.preprocess = transforms.Lambda(
            lambda x: torch.Tensor(x.copy()).permute(2, 0, 1).to(device)
        )
        self.normalize = lambda x: (x - 127.5) / 127.5

    def __len__(self):
        return len(self.items)

    def __getitem__(self, ix):
        f = self.items[ix]
        try:
            im = cv2.imread(f)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        except:
            blank = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
            return self.preprocess(blank), self.preprocess(blank)

        edges = detect_edges(im)
        im = cv2.resize(im, (self.image_size, self.image_size))
        edges = cv2.resize(edges, (self.image_size, self.image_size))
        im, edges = self.normalize(im), self.normalize(edges)

        # 在源图像上绘制彩色圆圈
        non_white_mask = np.sum(im, axis=-1) < 2.75
        non_white_y, non_white_x = np.nonzero(non_white_mask)

        if len(non_white_y) > 0:
            n_color_points = min(len(non_white_y), 300)
            idxs = np.random.choice(len(non_white_y), n_color_points, replace=False)

            for i in idxs:
                center_y, center_x = non_white_y[i], non_white_x[i]
                radius = 2
                y0 = max(0, center_y - radius + 1)
                y1 = min(self.image_size, center_y + radius)
                x0 = max(0, center_x - radius + 1)
                x1 = min(self.image_size, center_x + radius)

                if y0 < y1 and x0 < x1:
                    color = np.mean(im[y0:y1, x0:x1], axis=(0, 1))
                    edges[y0:y1, x0:x1] = color

        im, edges = self.preprocess(im), self.preprocess(edges)
        return edges, im


class UNetDown(nn.Module):
    """U-Net下采样块"""

    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    """U-Net上采样块"""

    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x


class GeneratorUNet(nn.Module):
    """Pix2Pix生成器"""

    def __init__(self, in_channels=3, out_channels=3):
        super(GeneratorUNet, self).__init__()
        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)
        return self.final(u7)


class Discriminator(nn.Module):
    """Pix2Pix判别器"""

    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels * 2, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)


def train_pix2pix(epochs=100):
    """训练Pix2Pix模型"""
    # 设置数据
    setup_pix2pix_data()
    train_items, val_items = train_test_split(glob.glob('ShoeV2_photo/*.png'), test_size=0.2, random_state=2)
    trn_ds = ShoesData(train_items)
    val_ds = ShoesData(val_items)

    trn_dl = DataLoader(trn_ds, batch_size=32, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=32, shuffle=True)

    # 初始化模型
    generator = GeneratorUNet().to(device)
    discriminator = Discriminator().to(device)

    # 权重初始化
    def weights_init_normal(m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm2d") != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    # 损失函数和优化器
    criterion_GAN = torch.nn.MSELoss()
    criterion_pixelwise = torch.nn.L1Loss()
    lambda_pixel = 100

    g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # 训练循环
    for epoch in range(epochs):
        for bx, batch in enumerate(trn_dl):
            real_src, real_trg = batch

            # 训练判别器
            d_optimizer.zero_grad()

            # 真实图像
            prediction_real = discriminator(real_trg, real_src)
            error_real = criterion_GAN(prediction_real, torch.ones(len(real_src), 1, 16, 16).to(device))

            # 生成图像
            fake_trg = generator(real_src)
            prediction_fake = discriminator(fake_trg.detach(), real_src)
            error_fake = criterion_GAN(prediction_fake, torch.zeros(len(real_src), 1, 16, 16).to(device))

            # 判别器损失
            errD = (error_real + error_fake) / 2
            errD.backward()
            d_optimizer.step()

            # 训练生成器
            g_optimizer.zero_grad()
            prediction = discriminator(fake_trg, real_src)
            loss_GAN = criterion_GAN(prediction, torch.ones(len(real_src), 1, 16, 16).to(device))
            loss_pixel = criterion_pixelwise(fake_trg, real_trg)
            errG = loss_GAN + lambda_pixel * loss_pixel
            errG.backward()
            g_optimizer.step()

            if bx % 100 == 0:
                print(f'Epoch: {epoch}, Batch: {bx}, Loss_D: {errD.item()}, Loss_G: {errG.item()}')

    return generator


# ==================== 主函数 ====================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='GAN Models')
    parser.add_argument('--model', type=str, required=True,
                        choices=['stylegan', 'cyclegan', 'srgan', 'pix2pix'],
                        help='Which GAN model to run')
    parser.add_argument('--image_path', type=str, default='MyImage.jpg',
                        help='Path to input image (for StyleGAN and SRGAN)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs (for CycleGAN and Pix2Pix)')

    args = parser.parse_args()

    if args.model == 'stylegan':
        print("Running StyleGAN2...")
        setup_stylegan_environment()
        synthesizer, mapper, trunc, latent2image = load_stylegan_models()

        # 生成随机潜在向量
        rand_latents = torch.randn(1, 512).to(device)
        generated_image = latent2image(trunc(mapper(rand_latents)))

        # 编码图像
        pred_dlatent = encode_image_stylegan(args.image_path)
        pred_image = latent2image(pred_dlatent)

        print("StyleGAN2 completed successfully!")

    elif args.model == 'cyclegan':
        print("Running CycleGAN...")
        G_AB, G_BA = train_cyclegan(epochs=args.epochs)
        print("CycleGAN training completed!")

    elif args.model == 'srgan':
        print("Running SRGAN...")
        model = setup_srgan_model()
        original, super_resolved = super_resolve_image(model, args.image_path)
        print("SRGAN completed successfully!")

    elif args.model == 'pix2pix':
        print("Running Pix2Pix...")
        generator = train_pix2pix(epochs=args.epochs)
        print("Pix2Pix training completed!")