# -*- coding: utf-8 -*-
# @project: LargeDemo
# @filename: ch07_object_detect_base.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/8/26 20:00

# https://chat.deepseek.com/a/chat/s/4a86ae48-92d8-4caa-8287-c19b8b9a805c

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from torchvision.ops import RoIPool, nms
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import selectivesearch
from skimage.segmentation import felzenszwalb
import os
from typing import List, Tuple, Dict, Any, Optional

# ==================== 全局配置与工具函数 ====================

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_iou(boxA: List[float], boxB: List[float], epsilon: float = 1e-5) -> float:
    x1 = max(boxA[0], boxB[0])
    y1 = max(boxA[1], boxB[1])
    x2 = min(boxA[2], boxB[2])
    y2 = min(boxA[3], boxB[3])
    width = (x2 - x1)
    height = (y2 - y1)
    if (width < 0) or (height < 0):
        return 0.0
    area_overlap = width * height
    area_a = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    area_b = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    area_combined = area_a + area_b - area_overlap
    iou = area_overlap / (area_combined + epsilon)
    return iou


def extract_candidates(img: np.ndarray) -> List[List[int]]:
    img_lbl, regions = selectivesearch.selective_search(img, scale=200, min_size=100)
    img_area = np.prod(img.shape[:2])
    candidates = []
    for r in regions:
        if r['rect'] in candidates:
            continue
        if r['size'] < (0.05 * img_area):
            continue
        if r['size'] > (1 * img_area):
            continue
        x, y, w, h = r['rect']
        candidates.append([x, y, w, h])
    return candidates


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def preprocess_image(img: np.ndarray) -> torch.Tensor:
    img = torch.tensor(img).permute(2, 0, 1)
    img = normalize(img)
    return img.to(device).float()


def decode(_y: torch.Tensor) -> torch.Tensor:
    _, preds = _y.max(-1)
    return preds


# ==================== 数据加载与预处理 ====================

class OpenImages(Dataset):
    def __init__(self, df: pd.DataFrame, image_folder: str):
        self.root = image_folder
        self.df = df
        self.unique_images = df['ImageID'].unique()

    def __len__(self):
        return len(self.unique_images)

    def __getitem__(self, ix: int) -> Tuple[np.ndarray, List, List, str]:
        image_id = self.unique_images[ix]
        image_path = f'{self.root}/{image_id}.jpg'
        image = cv2.imread(image_path, 1)[..., ::-1]  # BGR to RGB
        h, w, _ = image.shape
        df = self.df.copy()
        df = df[df['ImageID'] == image_id]
        boxes = df['XMin,YMin,XMax,YMax'.split(',')].values
        boxes = (boxes * np.array([w, h, w, h])).astype(np.uint16).tolist()
        classes = df['LabelName'].values.tolist()
        return image, boxes, classes, image_path


def prepare_data(df_path: str, image_root: str, num_samples: int = 500) -> Tuple:
    DF_RAW = pd.read_csv(df_path)
    ds = OpenImages(df=DF_RAW, image_folder=image_root)

    FPATHS, GTBBS, CLSS, DELTAS, ROIS, IOUS = [], [], [], [], [], []

    for ix, (im, bbs, labels, fpath) in enumerate(ds):
        if ix == num_samples:
            break
        H, W, _ = im.shape
        candidates = extract_candidates(im)
        candidates = np.array([(x, y, x + w, y + h) for x, y, w, h in candidates])
        ious, rois, clss, deltas = [], [], [], []
        ious = np.array([[get_iou(candidate, _bb_) for candidate in candidates] for _bb_ in bbs]).T

        for jx, candidate in enumerate(candidates):
            cx, cy, cX, cY = candidate
            candidate_ious = ious[jx]
            best_iou_at = np.argmax(candidate_ious)
            best_iou = candidate_ious[best_iou_at]
            _x, _y, _X, _Y = bbs[best_iou_at]
            if best_iou > 0.3:
                clss.append(labels[best_iou_at])
            else:
                clss.append('background')
            delta = np.array([_x - cx, _y - cy, _X - cX, _Y - cY]) / np.array([W, H, W, H])
            deltas.append(delta)
            rois.append(candidate / np.array([W, H, W, H]))

        FPATHS.append(fpath)
        IOUS.append(ious)
        ROIS.append(rois)
        CLSS.append(clss)
        DELTAS.append(deltas)
        GTBBS.append(bbs)

    FPATHS = [f'{image_root}/{os.path.basename(f).split(".")[0]}.jpg' for f in FPATHS]
    return FPATHS, GTBBS, CLSS, DELTAS, ROIS


# ==================== RCNN 模型与训练 ====================

class RCNNDataset(Dataset):
    def __init__(self, fpaths, rois, labels, deltas, gtbbs):
        self.fpaths = fpaths
        self.gtbbs = gtbbs
        self.rois = rois
        self.labels = labels
        self.deltas = deltas

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, ix):
        fpath = str(self.fpaths[ix])
        image = cv2.imread(fpath, 1)[..., ::-1]
        H, W, _ = image.shape
        sh = np.array([W, H, W, H])
        gtbbs = self.gtbbs[ix]
        rois = self.rois[ix]
        bbs = (np.array(rois) * sh).astype(np.uint16)
        labels = self.labels[ix]
        deltas = self.deltas[ix]
        crops = [image[y:Y, x:X] for (x, y, X, Y) in bbs]
        return image, crops, bbs, labels, deltas, gtbbs, fpath

    def collate_fn(self, batch):
        input, labels, deltas = [], [], []
        for ix in range(len(batch)):
            image, crops, image_bbs, image_labels, image_deltas, image_gt_bbs, image_fpath = batch[ix]
            crops = [cv2.resize(crop, (224, 224)) for crop in crops]
            crops = [preprocess_image(crop / 255.)[None] for crop in crops]
            input.extend(crops)
            labels.extend([label2target[c] for c in image_labels])
            deltas.extend(image_deltas)
        input = torch.cat(input).to(device)
        labels = torch.Tensor(labels).long().to(device)
        deltas = torch.Tensor(deltas).float().to(device)
        return input, labels, deltas


class RCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.vgg16(pretrained=True)
        self.backbone.classifier = nn.Sequential()
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval().to(device)
        feature_dim = 25088
        self.cls_score = nn.Linear(feature_dim, num_classes)
        self.bbox = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 4),
            nn.Tanh(),
        )
        self.cel = nn.CrossEntropyLoss()
        self.sl1 = nn.L1Loss()

    def forward(self, x):
        feat = self.backbone(x)
        cls_score = self.cls_score(feat)
        bbox = self.bbox(feat)
        return cls_score, bbox

    def calc_loss(self, probs, _deltas, labels, deltas):
        detection_loss = self.cel(probs, labels)
        ixs, = torch.where(labels != background_class)
        _deltas = _deltas[ixs]
        deltas = deltas[ixs]
        lmb = 10.0
        if len(ixs) > 0:
            regression_loss = self.sl1(_deltas, deltas)
            return detection_loss + lmb * regression_loss, detection_loss.detach(), regression_loss.detach()
        else:
            return detection_loss, detection_loss.detach(), torch.tensor(0.0)


def train_rcnn(data_path, image_root, num_epochs=5):
    FPATHS, GTBBS, CLSS, DELTAS, ROIS = prepare_data(data_path, image_root)
    targets = pd.DataFrame(np.concatenate(CLSS), columns=['label'])
    global label2target, target2label, background_class
    label2target = {l: t for t, l in enumerate(targets['label'].unique())}
    target2label = {t: l for l, t in label2target.items()}
    background_class = label2target['background']

    n_train = 9 * len(FPATHS) // 10
    train_ds = RCNNDataset(FPATHS[:n_train], ROIS[:n_train], CLSS[:n_train], DELTAS[:n_train], GTBBS[:n_train])
    test_ds = RCNNDataset(FPATHS[n_train:], ROIS[n_train:], CLSS[n_train:], DELTAS[n_train:], GTBBS[n_train:])

    train_loader = DataLoader(train_ds, batch_size=2, collate_fn=train_ds.collate_fn, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=2, collate_fn=test_ds.collate_fn, drop_last=True)

    model = RCNN(len(label2target)).to(device)
    criterion = model.calc_loss
    optimizer = optim.SGD(model.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        model.train()
        for inputs in train_loader:
            input, labels, deltas = inputs
            optimizer.zero_grad()
            _clss, _deltas = model(input)
            loss, loc_loss, regr_loss = criterion(_clss, _deltas, labels, deltas)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            for inputs in test_loader:
                input, labels, deltas = inputs
                _clss, _deltas = model(input)
                loss, loc_loss, regr_loss = criterion(_clss, _deltas, labels, deltas)

    return model


# ==================== Fast RCNN 模型与训练 ====================

class FRCNNDataset(Dataset):
    def __init__(self, fpaths, rois, labels, deltas, gtbbs):
        self.fpaths = fpaths
        self.gtbbs = gtbbs
        self.rois = rois
        self.labels = labels
        self.deltas = deltas

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, ix):
        fpath = str(self.fpaths[ix])
        image = cv2.imread(fpath, 1)[..., ::-1]
        gtbbs = self.gtbbs[ix]
        rois = self.rois[ix]
        labels = self.labels[ix]
        deltas = self.deltas[ix]
        return image, rois, labels, deltas, gtbbs, fpath

    def collate_fn(self, batch):
        input, rois, rixs, labels, deltas = [], [], [], [], []
        for ix in range(len(batch)):
            image, image_rois, image_labels, image_deltas, image_gt_bbs, image_fpath = batch[ix]
            image = cv2.resize(image, (224, 224))
            input.append(preprocess_image(image / 255.)[None])
            rois.extend(image_rois)
            rixs.extend([ix] * len(image_rois))
            labels.extend([label2target[c] for c in image_labels])
            deltas.extend(image_deltas)
        input = torch.cat(input).to(device)
        rois = torch.Tensor(rois).float().to(device)
        rixs = torch.Tensor(rixs).float().to(device)
        labels = torch.Tensor(labels).long().to(device)
        deltas = torch.Tensor(deltas).float().to(device)
        return input, rois, rixs, labels, deltas


class FRCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        rawnet = models.vgg16_bn(pretrained=True)
        for param in rawnet.features.parameters():
            param.requires_grad = True
        self.seq = nn.Sequential(*list(rawnet.features.children())[:-1])
        self.roipool = RoIPool(7, spatial_scale=14 / 224)
        feature_dim = 512 * 7 * 7
        self.cls_score = nn.Linear(feature_dim, num_classes)
        self.bbox = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 4),
            nn.Tanh(),
        )
        self.cel = nn.CrossEntropyLoss()
        self.sl1 = nn.L1Loss()

    def forward(self, input, rois, ridx):
        res = input
        res = self.seq(res)
        rois = torch.cat([ridx.unsqueeze(-1), rois * 224], dim=-1)
        res = self.roipool(res, rois)
        feat = res.view(len(res), -1)
        cls_score = self.cls_score(feat)
        bbox = self.bbox(feat)
        return cls_score, bbox

    def calc_loss(self, probs, _deltas, labels, deltas):
        detection_loss = self.cel(probs, labels)
        ixs, = torch.where(labels != background_class)
        _deltas = _deltas[ixs]
        deltas = deltas[ixs]
        lmb = 10.0
        if len(ixs) > 0:
            regression_loss = self.sl1(_deltas, deltas)
            return detection_loss + lmb * regression_loss, detection_loss.detach(), regression_loss.detach()
        else:
            return detection_loss, detection_loss.detach(), torch.tensor(0.0)


def train_fast_rcnn(data_path, image_root, num_epochs=5):
    FPATHS, GTBBS, CLSS, DELTAS, ROIS = prepare_data(data_path, image_root)
    targets = pd.DataFrame(np.concatenate(CLSS), columns=['label'])
    global label2target, target2label, background_class
    label2target = {l: t for t, l in enumerate(targets['label'].unique())}
    target2label = {t: l for l, t in label2target.items()}
    background_class = label2target['background']

    n_train = 9 * len(FPATHS) // 10
    train_ds = FRCNNDataset(FPATHS[:n_train], ROIS[:n_train], CLSS[:n_train], DELTAS[:n_train], GTBBS[:n_train])
    test_ds = FRCNNDataset(FPATHS[n_train:], ROIS[n_train:], CLSS[n_train:], DELTAS[n_train:], GTBBS[n_train:])

    train_loader = DataLoader(train_ds, batch_size=2, collate_fn=train_ds.collate_fn, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=2, collate_fn=test_ds.collate_fn, drop_last=True)

    model = FRCNN(len(label2target)).to(device)
    criterion = model.calc_loss
    optimizer = optim.SGD(model.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        model.train()
        for inputs in train_loader:
            input, rois, rixs, labels, deltas = inputs
            optimizer.zero_grad()
            _clss, _deltas = model(input, rois, rixs)
            loss, loc_loss, regr_loss = criterion(_clss, _deltas, labels, deltas)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            for inputs in test_loader:
                input, rois, rixs, labels, deltas = inputs
                _clss, _deltas = model(input, rois, rixs)
                loss, loc_loss, regr_loss = criterion(_clss, _deltas, labels, deltas)

    return model


# ==================== 测试与可视化 ====================

def test_predictions_rcnn(model, filename):
    img = np.array(Image.open(filename))
    candidates = extract_candidates(img)
    candidates = [(x, y, x + w, y + h) for x, y, w, h in candidates]
    input = []
    for candidate in candidates:
        x, y, X, Y = candidate
        crop = cv2.resize(img[y:Y, x:X], (224, 224))
        input.append(preprocess_image(crop / 255.)[None])
    input = torch.cat(input).to(device)
    with torch.no_grad():
        model.eval()
        probs, deltas = model(input)
        probs = torch.softmax(probs, -1)
        confs, clss = torch.max(probs, -1)
    candidates = np.array(candidates)
    confs, clss, probs, deltas = [tensor.cpu().numpy() for tensor in [confs, clss, probs, deltas]]

    ixs = clss != background_class
    confs, clss, probs, deltas, candidates = [tensor[ixs] for tensor in [confs, clss, probs, deltas, candidates]]
    bbs = (candidates + deltas).astype(np.uint16)
    ixs = nms(torch.tensor(bbs.astype(np.float32)), torch.tensor(confs), 0.05)
    confs, clss, probs, deltas, candidates, bbs = [tensor[ixs] for tensor in
                                                   [confs, clss, probs, deltas, candidates, bbs]]

    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    ax[0].imshow(img)
    ax[0].set_title('Original Image')
    ax[0].axis('off')
    ax[1].imshow(img)
    for bb in bbs:
        x, y, x2, y2 = bb
        rect = plt.Rectangle((x, y), x2 - x, y2 - y, fill=False, edgecolor='red', linewidth=2)
        ax[1].add_patch(rect)
    ax[1].set_title('Detected Objects')
    ax[1].axis('off')
    plt.show()


# ==================== 主程序入口 ====================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['rcnn', 'fast_rcnn', 'test'], default='test')
    parser.add_argument('--data_path', type=str, default='df.csv')
    parser.add_argument('--image_root', type=str, default='images/images')
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--test_image', type=str, default=None)
    args = parser.parse_args()

    if args.mode == 'rcnn':
        model = train_rcnn(args.data_path, args.image_root)
        torch.save(model.state_dict(), 'rcnn_model.pth')
    elif args.mode == 'fast_rcnn':
        model = train_fast_rcnn(args.data_path, args.image_root)
        torch.save(model.state_dict(), 'fast_rcnn_model.pth')
    elif args.mode == 'test':
        if args.model_path and args.test_image:
            model = RCNN(num_classes=3)  # 根据实际情况调整
            model.load_state_dict(torch.load(args.model_path))
            test_predictions_rcnn(model, args.test_image)
        else:
            print("Please provide --model_path and --test_image for testing")
