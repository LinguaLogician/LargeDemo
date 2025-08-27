# -*- coding: utf-8 -*-
# @project: LargeDemo
# @filename: ch10_od_is_application.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/8/26 20:04
# https://chat.deepseek.com/a/chat/s/520c8d4c-8f5a-43d1-8b8f-5ba635861b30
"""
多任务计算机视觉处理系统
集成了人群计数、人体姿态检测、图像着色和多目标分割功能
"""

import os
import json
import zipfile
import datetime
import requests
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models, datasets
import torchvision

# 检测相关库
try:
    import detectron2
    from detectron2 import model_zoo
    from detectron2.engine import DefaultPredictor, DefaultTrainer
    from detectron2.config import get_cfg
    from detectron2.utils.visualizer import Visualizer, ColorMode
    from detectron2.data import MetadataCatalog, DatasetCatalog
    from detectron2.data.datasets import register_coco_instances

    DETECTRON_AVAILABLE = True
except ImportError:
    DETECTRON_AVAILABLE = False

try:
    import h5py
    from scipy import io

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from pycococreatortools import pycococreatortools

    PYCOCO_AVAILABLE = True
except ImportError:
    PYCOCO_AVAILABLE = False

# 设置设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ==================== 通用工具函数 ====================

def download_file(url, filename):
    """下载文件"""
    print(f"Downloading {filename}...")
    response = requests.get(url, stream=True)
    with open(filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)


def setup_kaggle():
    """设置Kaggle API"""
    if not os.path.exists('/root/.kaggle/kaggle.json'):
        from google.colab import files
        files.upload()  # upload kaggle.json
        os.makedirs('/root/.kaggle', exist_ok=True)
        os.system('mv kaggle.json ~/.kaggle/')
        os.system('chmod 600 /root/.kaggle/kaggle.json')


# ==================== 人群计数模块 ====================

class CrowdsDataset(Dataset):
    """人群计数数据集"""

    def __init__(self, stems, image_folder, heatmap_folder, gt_folder):
        self.stems = stems
        self.image_folder = image_folder
        self.heatmap_folder = heatmap_folder
        self.gt_folder = gt_folder
        self.tfm = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, ix):
        _stem = self.stems[ix]
        image_path = f'{self.image_folder}/{_stem}.jpg'
        heatmap_path = f'{self.heatmap_folder}/{_stem}.h5'
        gt_path = f'{self.gt_folder}/GT_{_stem}.mat'

        pts = io.loadmat(gt_path)
        pts = len(pts['image_info'][0, 0][0, 0][0])

        image = Image.open(image_path).convert('RGB')
        with h5py.File(heatmap_path, 'r') as hf:
            gt = hf['density'][:]

        # 调整热图大小
        gt = np.array(Image.fromarray(gt).resize((gt.shape[1] // 8, gt.shape[0] // 8))) * 64
        return image, gt, pts

    def collate_fn(self, batch):
        ims, gts, pts = list(zip(*batch))
        ims = torch.stack([self.tfm(im) for im in ims]).to(device)
        gts = torch.stack([self.tfm(gt) for gt in gts]).to(device)
        return ims, gts, torch.tensor(pts).to(device)


def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
    """创建卷积层"""
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class CSRNet(nn.Module):
    """CSRNet人群计数模型"""

    def __init__(self, load_weights=False):
        super(CSRNet, self).__init__()
        self.seen = 0
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat = [512, 512, 512, 256, 128, 64]
        self.frontend = make_layers(self.frontend_feat)
        self.backend = make_layers(self.backend_feat, in_channels=512, dilation=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        if not load_weights:
            self._initialize_weights()
            mod = models.vgg16(pretrained=True)
            items = list(self.frontend.state_dict().items())
            _items = list(mod.state_dict().items())
            for i in range(len(items)):
                items[i][1].data[:] = _items[i][1].data[:]

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def train_crowd_counting_batch(model, data, optimizer, criterion):
    """训练人群计数批次"""
    model.train()
    optimizer.zero_grad()
    ims, gts, pts = data
    _gts = model(ims)
    loss = criterion(_gts, gts)
    loss.backward()
    optimizer.step()
    pts_loss = nn.L1Loss()(_gts.sum(), gts.sum())
    return loss.item(), pts_loss.item()


@torch.no_grad()
def validate_crowd_counting_batch(model, data, criterion):
    """验证人群计数批次"""
    model.eval()
    ims, gts, pts = data
    _gts = model(ims)
    loss = criterion(_gts, gts)
    pts_loss = nn.L1Loss()(_gts.sum(), gts.sum())
    return loss.item(), pts_loss.item()


def setup_crowd_counting_data():
    """设置人群计数数据"""
    if not os.path.exists('CSRNet-pytorch/'):
        os.system('git clone https://github.com/sizhky/CSRNet-pytorch.git')

    if not os.path.exists('shanghaitech_with_people_density_map.zip'):
        setup_kaggle()
        os.system('kaggle datasets download -d tthien/shanghaitech-with-people-density-map/')
        os.system('unzip -qq shanghaitech_with_people_density_map.zip')

    os.chdir('CSRNet-pytorch')
    if not os.path.exists('shanghaitech_with_people_density_map'):
        os.symlink('../shanghaitech_with_people_density_map', 'shanghaitech_with_people_density_map')


def run_crowd_counting():
    """运行人群计数"""
    if not SCIPY_AVAILABLE:
        print("请先安装scipy和h5py库")
        return

    setup_crowd_counting_data()

    # 数据路径
    image_folder = 'shanghaitech_with_people_density_map/ShanghaiTech/part_A/train_data/images/'
    heatmap_folder = 'shanghaitech_with_people_density_map/ShanghaiTech/part_A/train_data/ground-truth-h5/'
    gt_folder = 'shanghaitech_with_people_density_map/ShanghaiTech/part_A/train_data/ground-truth/'

    # 获取所有图像stem
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
    stems = [os.path.splitext(f)[0] for f in image_files]

    # 划分训练验证集
    trn_stems, val_stems = train_test_split(stems, random_state=10)

    # 创建数据集和数据加载器
    trn_ds = CrowdsDataset(trn_stems, image_folder, heatmap_folder, gt_folder)
    val_ds = CrowdsDataset(val_stems, image_folder, heatmap_folder, gt_folder)

    trn_dl = DataLoader(trn_ds, batch_size=1, shuffle=True, collate_fn=trn_ds.collate_fn)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=val_ds.collate_fn)

    # 初始化模型
    model = CSRNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-6)
    n_epochs = 20

    # 训练循环
    for epoch in range(n_epochs):
        # 训练阶段
        model.train()
        trn_losses, trn_pts_losses = [], []
        for bx, data in enumerate(trn_dl):
            loss, pts_loss = train_crowd_counting_batch(model, data, optimizer, criterion)
            trn_losses.append(loss)
            trn_pts_losses.append(pts_loss)

        # 验证阶段
        model.eval()
        val_losses, val_pts_losses = [], []
        for bx, data in enumerate(val_dl):
            loss, pts_loss = validate_crowd_counting_batch(model, data, criterion)
            val_losses.append(loss)
            val_pts_losses.append(pts_loss)

        print(f'Epoch {epoch + 1}/{n_epochs}, '
              f'Trn Loss: {np.mean(trn_losses):.4f}, Trn Pts Loss: {np.mean(trn_pts_losses):.4f}, '
              f'Val Loss: {np.mean(val_losses):.4f}, Val Pts Loss: {np.mean(val_pts_losses):.4f}')

    # 测试模型
    test_folder = 'shanghaitech_with_people_density_map/ShanghaiTech/part_A/test_data/'
    test_images = [f for f in os.listdir(f'{test_folder}/images') if f.endswith('.jpg')]
    test_image_path = os.path.join(test_folder, 'images', test_images[0])

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img = transform(Image.open(test_image_path).convert('RGB')).to(device)
    output = model(img.unsqueeze(0))
    print("Predicted Count:", int(output.detach().cpu().sum().numpy()))

    # 可视化结果
    temp = output.detach().cpu().squeeze().numpy()
    plt.imshow(temp, cmap='jet')
    plt.show()


# ==================== 人体姿态检测模块 ====================

def run_human_pose_detection():
    """运行人体姿态检测"""
    if not DETECTRON_AVAILABLE:
        print("请先安装detectron2库")
        return

    # 下载示例图像
    if not os.path.exists('image.png'):
        download_file('https://i.imgur.com/ldzGSHk.jpg', 'image.png')

    # 配置模型
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)

    # 读取和预处理图像
    im = Image.open('image.png').convert('RGB')
    im = im.resize((im.width // 2, im.height // 2))  # 调整大小为一半

    # 预测
    outputs = predictor(np.array(im))

    # 可视化结果
    v = Visualizer(np.array(im)[:, :, ::-1],
                   MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
                   scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    plt.figure(figsize=(12, 8))
    plt.imshow(out.get_image())
    plt.axis('off')
    plt.show()


# ==================== 图像着色模块 ====================

class Identity(nn.Module):
    """恒等层"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class DownConv(nn.Module):
    """下采样卷积块"""

    def __init__(self, ni, no, maxpool=True):
        super().__init__()
        self.model = nn.Sequential(
            nn.MaxPool2d(2) if maxpool else Identity(),
            nn.Conv2d(ni, no, 3, padding=1),
            nn.BatchNorm2d(no),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(no, no, 3, padding=1),
            nn.BatchNorm2d(no),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.model(x)


class UpConv(nn.Module):
    """上采样卷积块"""

    def __init__(self, ni, no, maxpool=True):
        super().__init__()
        self.convtranspose = nn.ConvTranspose2d(ni, no, 2, stride=2)
        self.convlayers = nn.Sequential(
            nn.Conv2d(no + no, no, 3, padding=1),
            nn.BatchNorm2d(no),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(no, no, 3, padding=1),
            nn.BatchNorm2d(no),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x, y):
        x = self.convtranspose(x)
        x = torch.cat([x, y], axis=1)
        x = self.convlayers(x)
        return x


class UNet(nn.Module):
    """U-Net图像着色模型"""

    def __init__(self):
        super().__init__()
        self.d1 = DownConv(3, 64, maxpool=False)
        self.d2 = DownConv(64, 128)
        self.d3 = DownConv(128, 256)
        self.d4 = DownConv(256, 512)
        self.d5 = DownConv(512, 1024)
        self.u5 = UpConv(1024, 512)
        self.u4 = UpConv(512, 256)
        self.u3 = UpConv(256, 128)
        self.u2 = UpConv(128, 64)
        self.u1 = nn.Conv2d(64, 3, kernel_size=1, stride=1)

    def forward(self, x):
        x0 = self.d1(x)  # 32
        x1 = self.d2(x0)  # 16
        x2 = self.d3(x1)  # 8
        x3 = self.d4(x2)  # 4
        x4 = self.d5(x3)  # 2
        X4 = self.u5(x4, x3)  # 4
        X3 = self.u4(X4, x2)  # 8
        X2 = self.u3(X3, x1)  # 16
        X1 = self.u2(X2, x0)  # 32
        X0 = self.u1(X1)  # 3
        return X0


class ColorizeDataset(Dataset):
    """图像着色数据集"""

    def __init__(self, root, train=True):
        self.dataset = datasets.CIFAR10(root, train=train, download=True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, ix):
        im, _ = self.dataset[ix]
        bw = im.convert('L').convert('RGB')
        bw, im = np.array(bw) / 255., np.array(im) / 255.
        bw, im = [torch.tensor(i).permute(2, 0, 1).to(device).float() for i in [bw, im]]
        return bw, im


def train_colorization_batch(model, data, optimizer, criterion):
    """训练图像着色批次"""
    model.train()
    x, y = data
    _y = model(x)
    optimizer.zero_grad()
    loss = criterion(_y, y)
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def validate_colorization_batch(model, data, criterion):
    """验证图像着色批次"""
    model.eval()
    x, y = data
    _y = model(x)
    loss = criterion(_y, y)
    return loss.item()


def run_image_colorization():
    """运行图像着色"""
    # 设置数据路径
    data_folder = '~/cifar10/cifar/'

    # 创建数据集
    trn_ds = ColorizeDataset(data_folder, train=True)
    val_ds = ColorizeDataset(data_folder, train=False)

    # 创建数据加载器
    trn_dl = DataLoader(trn_ds, batch_size=256, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=256, shuffle=False)

    # 初始化模型
    model = UNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    n_epochs = 100

    # 训练循环
    for epoch in range(n_epochs):
        # 训练阶段
        model.train()
        trn_losses = []
        for bx, data in enumerate(trn_dl):
            loss = train_colorization_batch(model, data, optimizer, criterion)
            trn_losses.append(loss)

        # 验证阶段
        model.eval()
        val_losses = []
        for bx, data in enumerate(val_dl):
            loss = validate_colorization_batch(model, data, criterion)
            val_losses.append(loss)

        # 更新学习率
        exp_lr_scheduler.step()

        if (epoch + 1) % 5 == 0:
            print(f'Epoch {epoch + 1}/{n_epochs}, '
                  f'Trn Loss: {np.mean(trn_losses):.4f}, '
                  f'Val Loss: {np.mean(val_losses):.4f}')

            # 可视化一些结果
            model.eval()
            a, b = next(iter(DataLoader(val_ds, batch_size=1, shuffle=True)))
            _b = model(a)

            # 将张量转换为图像格式
            a = a[0].cpu().permute(1, 2, 0).numpy()
            b = b[0].cpu().permute(1, 2, 0).numpy()
            _b = _b[0].detach().cpu().permute(1, 2, 0).numpy()

            # 显示图像
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(a)
            axes[0].set_title('Input (B&W)')
            axes[0].axis('off')

            axes[1].imshow(b)
            axes[1].set_title('Ground Truth')
            axes[1].axis('off')

            axes[2].imshow(_b)
            axes[2].set_title('Predicted')
            axes[2].axis('off')

            plt.show()


# ==================== 多目标分割模块 ====================

def setup_multi_object_segmentation_data():
    """设置多目标分割数据"""
    required_classes = 'person,dog,bird,car,elephant,football,jug,laptop,Mushroom,Pizza,Rocket,Shirt,Traffic sign,Watermelon,Zebra'
    required_classes = [c.lower() for c in required_classes.lower().split(',')]

    # 下载类别文件
    if not os.path.exists('classes.csv'):
        download_file('https://raw.githubusercontent.com/openimages/dataset/master/dict.csv', 'classes.csv')

    # 下载标注文件
    if not os.path.exists('train-annotations-object-segmentation.csv'):
        download_file('https://storage.googleapis.com/openimages/v5/train-annotations-object-segmentation.csv',
                      'train-annotations-object-segmentation.csv')

    # 读取类别数据
    classes = pd.read_csv('classes.csv', header=None)
    classes.columns = ['class', 'class_name']
    classes = classes[classes['class_name'].map(lambda x: x in required_classes)]

    # 读取标注数据
    df = pd.read_csv('train-annotations-object-segmentation.csv')
    data = pd.merge(df, classes, left_on='LabelName', right_on='class')

    # 选择子集
    subset_data = data.groupby('class_name').agg({'ImageID': lambda x: list(x)[:500]})
    subset_data = [item for sublist in subset_data.ImageID.tolist() for item in sublist]
    subset_data = data[data['ImageID'].map(lambda x: x in subset_data)]
    subset_masks = subset_data['MaskPath'].tolist()

    # 创建目录
    os.makedirs('masks', exist_ok=True)

    # 下载和解压掩码
    for c in '0123456789abcdef':
        zip_file = f'train-masks-{c}.zip'
        if not os.path.exists(zip_file):
            download_file(f'https://storage.googleapis.com/openimages/v5/train-masks/train-masks-{c}.zip', zip_file)

        # 解压文件
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall('tmp_masks')

        # 移动需要的文件
        tmp_masks = [f for f in os.listdir('tmp_masks') if os.path.isfile(os.path.join('tmp_masks', f))]
        for mask_file in tmp_masks:
            if mask_file in subset_masks:
                os.rename(os.path.join('tmp_masks', mask_file), os.path.join('masks', mask_file))

        # 清理
        os.system('rm -rf tmp_masks')
        os.remove(zip_file)

    # 过滤数据
    masks = [f for f in os.listdir('masks') if os.path.isfile(os.path.join('masks', f))]
    subset_data = subset_data[subset_data['MaskPath'].map(lambda x: x in masks)]
    subset_imageIds = subset_data['ImageID'].tolist()

    # 下载图像
    os.makedirs('images', exist_ok=True)
    # 这里需要实现图像下载逻辑，但由于OpenImages下载较复杂，简化处理

    # 创建COCO格式标注
    INFO = {
        "description": "MyData2020",
        "url": "None",
        "version": "1.0",
        "year": 2020,
        "contributor": "sizhky",
        "date_created": datetime.datetime.utcnow().isoformat(' ')
    }

    LICENSES = [{"id": 1, "name": "MIT"}]

    CATEGORIES = [{'id': id + 1, 'name': name.replace('/', ''), 'supercategory': 'none'}
                  for id, (_, row) in enumerate(classes.iterrows())]

    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    # 这里需要实现COCO格式转换逻辑

    # 保存COCO标注
    with open('images.json', 'w') as f:
        json.dump(coco_output, f)

    return classes


def run_multi_object_segmentation():
    """运行多目标分割"""
    if not all([DETECTRON_AVAILABLE, PYCOCO_AVAILABLE]):
        print("请先安装detectron2和pycocotools库")
        return

    # 设置数据
    classes = setup_multi_object_segmentation_data()

    # 注册COCO数据集
    register_coco_instances("dataset_train", {}, "images.json", "images")

    # 配置模型
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("dataset_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 5000
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(classes)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # 训练模型
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    # 保存模型
    torch.save(trainer.model.state_dict(), os.path.join(cfg.OUTPUT_DIR, "trained_model.pth"))

    # 测试模型
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "trained_model.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.25
    predictor = DefaultPredictor(cfg)

    # 可视化一些结果
    image_files = [f for f in os.listdir('images') if f.endswith(('.jpg', '.png'))]
    for i in range(min(5, len(image_files))):
        im = Image.open(os.path.join('images', image_files[i])).convert('RGB')
        outputs = predictor(np.array(im))

        v = Visualizer(np.array(im)[:, :, ::-1],
                       metadata=MetadataCatalog.get("dataset_train"),
                       scale=0.5,
                       instance_mode=ColorMode.IMAGE_BW)

        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        plt.figure(figsize=(12, 8))
        plt.imshow(out.get_image())
        plt.axis('off')
        plt.show()


# ==================== 主程序 ====================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='多任务计算机视觉处理系统')
    parser.add_argument('--task', type=str, required=True,
                        choices=['crowd_counting', 'pose_detection', 'colorization', 'segmentation'],
                        help='选择要运行的任务')

    args = parser.parse_args()

    if args.task == 'crowd_counting':
        run_crowd_counting()
    elif args.task == 'pose_detection':
        run_human_pose_detection()
    elif args.task == 'colorization':
        run_image_colorization()
    elif args.task == 'segmentation':
        run_multi_object_segmentation()
    else:
        print("请选择有效的任务: crowd_counting, pose_detection, colorization, segmentation")