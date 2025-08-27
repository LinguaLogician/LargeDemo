# -*- coding: utf-8 -*-
# @project: LargeDemo
# @filename: ch09_image_segmentation.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/8/26 20:03
# https://chat.deepseek.com/a/chat/s/1c05223a-c98f-4adb-8ce6-833def30354d

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision import transforms
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

# 根据环境选择导入方式
try:
    from torch_snippets import *
    from torch_snippets.inspector import inspect
except:
    print("torch_snippets not available, some functions may not work")

# 尝试导入engine等工具模块
try:
    from engine import train_one_epoch, evaluate
    import utils
    import transforms as T
except:
    print("Detection utils not available, some functions may not work")

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def download_and_setup_instance_segmentation_data():
    """下载并设置实例分割数据"""
    if not os.path.exists('images/training'):
        os.system('wget --quiet http://sceneparsing.csail.mit.edu/data/ChallengeData2017/images.tar')
        os.system('wget --quiet http://sceneparsing.csail.mit.edu/data/ChallengeData2017/annotations_instance.tar')
        os.system('tar -xf images.tar')
        os.system('tar -xf annotations_instance.tar')
        os.system('rm images.tar annotations_instance.tar')

    # 安装必要依赖
    os.system('pip install -qU torch_snippets')
    os.system(
        'wget --quiet https://raw.githubusercontent.com/pytorch/vision/release/0.12/references/detection/engine.py')
    os.system(
        'wget --quiet https://raw.githubusercontent.com/pytorch/vision/release/0.12/references/detection/utils.py')
    os.system(
        'wget --quiet https://raw.githubusercontent.com/pytorch/vision/release/0.12/references/detection/transforms.py')
    os.system(
        'wget --quiet https://raw.githubusercontent.com/pytorch/vision/release/0.12/references/detection/coco_eval.py')
    os.system(
        'wget --quiet https://raw.githubusercontent.com/pytorch/vision/release/0.12/references/detection/coco_utils.py')
    os.system('pip install -q -U "git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI"')


def download_and_setup_semantic_segmentation_data():
    """下载并设置语义分割数据"""
    if not os.path.exists('dataset1'):
        os.system('wget -q https://www.dropbox.com/s/0pigmmmynbf9xwq/dataset1.zip')
        os.system('unzip -q dataset1.zip')
        os.system('rm dataset1.zip')
        os.system('pip install -q torch_snippets pytorch_model_summary')


class InstanceSegmentationDataset(Dataset):
    """实例分割数据集类（单类别）"""

    def __init__(self, items, transforms, N, class_id=4):
        self.items = items
        self.transforms = transforms
        self.N = N
        self.class_id = class_id

    def get_mask(self, path):
        an = read(path, 1).transpose(2, 0, 1)
        r, g, b = an
        nzs = np.nonzero(r == self.class_id)
        instances = np.unique(g[nzs])
        masks = np.zeros((len(instances), *r.shape))
        for ix, _id in enumerate(instances):
            masks[ix] = g == _id
        return masks

    def __getitem__(self, ix):
        _id = self.items[ix]
        img_path = f'images/training/{_id}.jpg'
        mask_path = f'annotations_instance/training/{_id}.png'
        masks = self.get_mask(mask_path)
        obj_ids = np.arange(1, len(masks) + 1)
        img = Image.open(img_path).convert("RGB")
        num_objs = len(obj_ids)
        boxes = []

        for i in range(num_objs):
            obj_pixels = np.where(masks[i])
            xmin = np.min(obj_pixels[1])
            xmax = np.max(obj_pixels[1])
            ymin = np.min(obj_pixels[0])
            ymax = np.max(obj_pixels[0])
            if (((xmax - xmin) <= 10) | (ymax - ymin) <= 10):
                xmax = xmin + 10
                ymax = ymin + 10
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        image_id = torch.tensor([ix])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        if (img.dtype == torch.float32) or (img.dtype == torch.uint8):
            img = img / 255.

        return img, target

    def __len__(self):
        return self.N

    def choose(self):
        return self[randint(len(self))]


class MultiClassInstanceSegmentationDataset(Dataset):
    """多类别实例分割数据集类"""

    def __init__(self, items, transforms, N, classes_list=[4, 6]):
        self.items = items
        self.transforms = transforms
        self.N = N
        self.classes_list = classes_list

    def get_mask(self, path):
        an = read(path, 1).transpose(2, 0, 1)
        r, g, b = an
        cls = list(set(np.unique(r)).intersection(set(self.classes_list)))
        masks = []
        labels = []

        for _cls in cls:
            nzs = np.nonzero(r == _cls)
            instances = np.unique(g[nzs])
            for ix, _id in enumerate(instances):
                masks.append(g == _id)
                labels.append(self.classes_list.index(_cls) + 1)

        return np.array(masks), np.array(labels)

    def __getitem__(self, ix):
        _id = self.items[ix]
        img_path = f'images/training/{_id}.jpg'
        mask_path = f'annotations_instance/training/{_id}.png'
        masks, labels = self.get_mask(mask_path)
        obj_ids = np.arange(1, len(masks) + 1)
        img = Image.open(img_path).convert("RGB")
        num_objs = len(obj_ids)
        boxes = []

        for i in range(num_objs):
            obj_pixels = np.where(masks[i])
            xmin = np.min(obj_pixels[1])
            xmax = np.max(obj_pixels[1])
            ymin = np.min(obj_pixels[0])
            ymax = np.max(obj_pixels[0])
            if (((xmax - xmin) <= 10) | (ymax - ymin) <= 10):
                xmax = xmin + 10
                ymax = ymin + 10
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        image_id = torch.tensor([ix])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        if (img.dtype == torch.float32) or (img.dtype == torch.uint8):
            img = img / 255.

        return img, target

    def __len__(self):
        return self.N

    def choose(self):
        return self[randint(len(self))]


class SemanticSegmentationDataset(Dataset):
    """语义分割数据集类"""

    def __init__(self, split):
        self.items = stems(f'dataset1/images_prepped_{split}')
        self.split = split
        self.tfms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, ix):
        image = read(f'dataset1/images_prepped_{self.split}/{self.items[ix]}.png', 1)
        image = cv2.resize(image, (224, 224))
        mask = read(f'dataset1/annotations_prepped_{self.split}/{self.items[ix]}.png')
        mask = cv2.resize(mask, (224, 224))
        return image, mask

    def choose(self):
        return self[randint(len(self))]

    def collate_fn(self, batch):
        ims, masks = list(zip(*batch))
        ims = torch.cat([self.tfms(im.copy() / 255.)[None] for im in ims]).float().to(device)
        ce_masks = torch.cat([torch.Tensor(mask[None]) for mask in masks]).long().to(device)
        return ims, ce_masks


def get_transform(train):
    """获取数据变换"""
    transforms_list = []
    transforms_list.append(T.PILToTensor())
    if train:
        transforms_list.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms_list)


def get_model_instance_segmentation(num_classes):
    """获取实例分割模型"""
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # 替换分类头
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # 替换掩码预测头
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer, num_classes)
    return model


def run_instance_segmentation():
    """运行实例分割（单类别）"""
    print("Running instance segmentation (single class)...")
    download_and_setup_instance_segmentation_data()

    all_images = Glob('images/training')
    all_annots = Glob('annotations_instance/training')

    # 筛选包含特定类别的标注
    annots = []
    for ann in Tqdm(all_annots[:5000]):
        _ann = read(ann, 1).transpose(2, 0, 1)
        r, g, b = _ann
        if 4 not in np.unique(r):
            continue
        annots.append(ann)

    _annots = stems(annots)
    trn_items, val_items = train_test_split(_annots, random_state=2)

    # 创建数据集和数据加载器
    dataset = InstanceSegmentationDataset(trn_items, get_transform(train=True), N=len(trn_items))
    dataset_test = InstanceSegmentationDataset(val_items, get_transform(train=False), N=len(val_items))

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=0,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=utils.collate_fn)

    # 初始化模型
    num_classes = 2
    model = get_model_instance_segmentation(num_classes).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # 训练
    num_epochs = 1
    trn_history = []

    for epoch in range(num_epochs):
        res = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        trn_history.append(res)
        lr_scheduler.step()
        res = evaluate(model, data_loader_test, device=device)

    # 绘制训练损失
    plt.figure(figsize=(10, 5))
    plt.title('Training Loss')
    losses = [np.mean(list(trn_history[i].meters['loss'].deque)) for i in range(len(trn_history))]
    plt.plot(losses)
    plt.show()

    # 测试模型
    model.eval()
    im = dataset_test[10][0]
    show(im)

    with torch.no_grad():
        prediction = model([im.to(device)])
        for i in range(len(prediction[0]['masks'])):
            plt.imshow(Image.fromarray(prediction[0]['masks'][i, 0].mul(255).byte().cpu().numpy()))
            plt.title('Class: ' + str(prediction[0]['labels'][i].cpu().numpy()) + ' Score:' + str(
                prediction[0]['scores'][i].cpu().numpy()))
            plt.show()


def run_multi_class_instance_segmentation():
    """运行多类别实例分割"""
    print("Running multi-class instance segmentation...")
    download_and_setup_instance_segmentation_data()

    all_images = Glob('images/training')
    all_annots = Glob('annotations_instance/training')

    classes_list = [4, 6]
    annots = []

    for ann in Tqdm(all_annots):
        _ann = read(ann, 1).transpose(2, 0, 1)
        r, g, b = _ann
        if np.array([num in np.unique(r) for num in classes_list]).sum() == 0:
            continue
        annots.append(ann)

    _annots = stems(annots)
    trn_items, val_items = train_test_split(_annots, random_state=2)

    # 创建数据集和数据加载器
    dataset = MultiClassInstanceSegmentationDataset(trn_items, get_transform(train=True), N=3000)
    dataset_test = MultiClassInstanceSegmentationDataset(val_items, get_transform(train=False), N=800)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=0,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=utils.collate_fn)

    # 初始化模型
    num_classes = 3
    model = get_model_instance_segmentation(num_classes).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # 训练
    num_epochs = 5
    trn_history = []

    for epoch in range(num_epochs):
        res = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        trn_history.append(res)
        lr_scheduler.step()
        res = evaluate(model, data_loader_test, device=device)

    # 绘制训练损失
    plt.figure(figsize=(10, 5))
    plt.title('Training Loss')
    losses = [np.mean(list(trn_history[i].meters['loss'].deque)) for i in range(len(trn_history))]
    plt.plot(losses)
    plt.show()

    # 测试模型
    model.eval()
    k = 423
    im = dataset_test[k][0]
    show(im, sz=5)

    with torch.no_grad():
        prediction = model([im.to(device)])
        for i in range(len(prediction[0]['masks'])):
            plt.imshow(Image.fromarray(prediction[0]['masks'][i, 0].mul(255).byte().cpu().numpy()))
            plt.title('Class: ' + str(prediction[0]['labels'][i].cpu().numpy()) + ' Score:' + str(
                prediction[0]['scores'][i].cpu().numpy()))
            plt.show()


# U-Net模型定义
def conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


def up_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
        nn.ReLU(inplace=True)
    )


class UNet(nn.Module):
    """U-Net语义分割模型"""

    def __init__(self, pretrained=True, out_channels=12):
        super().__init__()

        self.encoder = torchvision.models.vgg16_bn(pretrained=pretrained).features
        self.block1 = nn.Sequential(*self.encoder[:6])
        self.block2 = nn.Sequential(*self.encoder[6:13])
        self.block3 = nn.Sequential(*self.encoder[13:20])
        self.block4 = nn.Sequential(*self.encoder[20:27])
        self.block5 = nn.Sequential(*self.encoder[27:34])

        self.bottleneck = nn.Sequential(*self.encoder[34:])
        self.conv_bottleneck = conv(512, 1024)

        self.up_conv6 = up_conv(1024, 512)
        self.conv6 = conv(512 + 512, 512)
        self.up_conv7 = up_conv(512, 256)
        self.conv7 = conv(256 + 512, 256)
        self.up_conv8 = up_conv(256, 128)
        self.conv8 = conv(128 + 256, 128)
        self.up_conv9 = up_conv(128, 64)
        self.conv9 = conv(64 + 128, 64)
        self.up_conv10 = up_conv(64, 32)
        self.conv10 = conv(32 + 64, 32)
        self.conv11 = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)

        bottleneck = self.bottleneck(block5)
        x = self.conv_bottleneck(bottleneck)

        x = self.up_conv6(x)
        x = torch.cat([x, block5], dim=1)
        x = self.conv6(x)

        x = self.up_conv7(x)
        x = torch.cat([x, block4], dim=1)
        x = self.conv7(x)

        x = self.up_conv8(x)
        x = torch.cat([x, block3], dim=1)
        x = self.conv8(x)

        x = self.up_conv9(x)
        x = torch.cat([x, block2], dim=1)
        x = self.conv9(x)

        x = self.up_conv10(x)
        x = torch.cat([x, block1], dim=1)
        x = self.conv10(x)

        x = self.conv11(x)

        return x


def UnetLoss(preds, targets):
    """U-Net损失函数"""
    ce = nn.CrossEntropyLoss()
    ce_loss = ce(preds, targets)
    acc = (torch.max(preds, 1)[1] == targets).float().mean()
    return ce_loss, acc


def train_batch(model, data, optimizer, criterion):
    """训练批次"""
    model.train()
    ims, ce_masks = data
    _masks = model(ims)
    optimizer.zero_grad()
    loss, acc = criterion(_masks, ce_masks)
    loss.backward()
    optimizer.step()
    return loss.item(), acc.item()


@torch.no_grad()
def validate_batch(model, data, criterion):
    """验证批次"""
    model.eval()
    ims, masks = data
    _masks = model(ims)
    loss, acc = criterion(_masks, masks)
    return loss.item(), acc.item()


def run_semantic_segmentation():
    """运行语义分割"""
    print("Running semantic segmentation with U-Net...")
    download_and_setup_semantic_segmentation_data()

    # 创建数据集和数据加载器
    trn_ds = SemanticSegmentationDataset('train')
    val_ds = SemanticSegmentationDataset('test')
    trn_dl = DataLoader(trn_ds, batch_size=4, shuffle=True, collate_fn=trn_ds.collate_fn)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=True, collate_fn=val_ds.collate_fn)

    # 初始化模型
    model = UNet().to(device)
    criterion = UnetLoss
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    n_epochs = 20

    # 训练循环
    log = Report(n_epochs)
    for ex in range(n_epochs):
        N = len(trn_dl)
        for bx, data in enumerate(trn_dl):
            loss, acc = train_batch(model, data, optimizer, criterion)
            log.record(ex + (bx + 1) / N, trn_loss=loss, trn_acc=acc, end='\r')

        N = len(val_dl)
        for bx, data in enumerate(val_dl):
            loss, acc = validate_batch(model, data, criterion)
            log.record(ex + (bx + 1) / N, val_loss=loss, val_acc=acc, end='\r')

        log.report_avgs(ex + 1)

    # 绘制损失曲线
    log.plot_epochs(['trn_loss', 'val_loss'])

    # 测试模型
    im, mask = next(iter(val_dl))
    _mask = model(im)
    _, _mask = torch.max(_mask, dim=1)

    subplots([im[0].permute(1, 2, 0).detach().cpu()[:, :, 0],
              mask.permute(1, 2, 0).detach().cpu()[:, :, 0],
              _mask.permute(1, 2, 0).detach().cpu()[:, :, 0]],
             nc=3, titles=['Original image', 'Original mask', 'Predicted mask'])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run different segmentation tasks')
    parser.add_argument('--task', type=str, choices=['instance', 'multi_class', 'semantic', 'all'],
                        default='all', help='Which task to run')

    args = parser.parse_args()

    if args.task == 'instance' or args.task == 'all':
        run_instance_segmentation()

    if args.task == 'multi_class' or args.task == 'all':
        run_multi_class_instance_segmentation()

    if args.task == 'semantic' or args.task == 'all':
        run_semantic_segmentation()
