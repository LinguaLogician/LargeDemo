# -*- coding: utf-8 -*-
# @project: LargeDemo
# @filename: ch08_object_detect_advanced.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/8/26 20:00
# https://chat.deepseek.com/a/chat/s/d7ae89ac-7c7a-4a77-b0d9-52105a3e0b9e
import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import nms
import subprocess
import sys
from typing import Dict, List, Tuple, Optional

# Common utility functions and variables
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def setup_environment():
    """Install necessary packages and download datasets if not present"""
    if not os.path.exists('images') and not os.path.exists('open-images-bus-trucks'):
        print("Setting up environment...")
        # Install torch_snippets if not available
        try:
            import torch_snippets
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "torch_snippets"])

        # For Google Colab environment
        try:
            from google.colab import files
            files.upload()  # upload kaggle.json
            os.makedirs('/root/.kaggle', exist_ok=True)
            os.rename('kaggle.json', '/root/.kaggle/kaggle.json')
            os.chmod('/root/.kaggle/kaggle.json', 0o600)
            subprocess.check_call(['kaggle', 'datasets', 'download', '-d', 'sixhky/open-images-bus-trucks/'])
            subprocess.check_call(['unzip', '-qq', 'open-images-bus-trucks.zip'])
            os.remove('open-images-bus-trucks.zip')
        except:
            # Alternative download method
            subprocess.check_call(
                ['wget', '--quiet', 'https://www.dropbox.com/s/agmzwk95v96ihic/open-images-bus-trucks.tar.xz'])
            subprocess.check_call(['tar', '-xf', 'open-images-bus-trucks.tar.xz'])
            os.remove('open-images-bus-trucks.tar.xz')


def preprocess_image_frcnn(img: np.ndarray) -> torch.Tensor:
    """Preprocess image for Faster R-CNN"""
    img = torch.tensor(img).permute(2, 0, 1)
    return img.to(device).float()


def preprocess_image_ssd(img: np.ndarray) -> torch.Tensor:
    """Preprocess image for SSD"""
    from torchvision import transforms
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    img = torch.tensor(img).permute(2, 0, 1)
    img = normalize(img)
    return img.to(device).float()


# Common Dataset class for Faster R-CNN and SSD
class OpenDataset(torch.utils.data.Dataset):
    def __init__(self, df, image_dir, img_size=(224, 224), model_type='frcnn'):
        self.image_dir = image_dir
        self.files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        self.df = df
        self.image_infos = df['ImageID'].unique()
        self.w, self.h = img_size
        self.model_type = model_type

    def __getitem__(self, ix):
        image_id = self.image_infos[ix]
        img_path = os.path.join(self.image_dir, image_id)
        img = Image.open(img_path).convert("RGB")
        img = np.array(img.resize((self.w, self.h), resample=Image.BILINEAR)) / 255.

        data = self.df[self.df['ImageID'] == image_id]
        labels = data['LabelName'].values.tolist()
        data = data[['XMin', 'YMin', 'XMax', 'YMax']].values
        data[:, [0, 2]] *= self.w
        data[:, [1, 3]] *= self.h
        boxes = data.astype(np.uint32).tolist()

        if self.model_type == 'frcnn':
            target = {
                "boxes": torch.Tensor(boxes).float(),
                "labels": torch.Tensor([label2target[i] for i in labels]).long()
            }
            img = preprocess_image_frcnn(img)
            return img, target
        else:  # SSD
            return img, boxes, labels

    def collate_fn(self, batch):
        if self.model_type == 'frcnn':
            return tuple(zip(*batch))
        else:
            images, boxes, labels = [], [], []
            for item in batch:
                img, image_boxes, image_labels = item
                img = preprocess_image_ssd(img)[None]
                images.append(img)
                boxes.append(torch.tensor(image_boxes).float().to(device) / 300.)
                labels.append(torch.tensor([label2target[c] for c in image_labels]).long().to(device))
            images = torch.cat(images).to(device)
            return images, boxes, labels

    def __len__(self):
        return len(self.image_infos)


# Faster R-CNN functions
def train_faster_rcnn():
    """Train Faster R-CNN model"""
    print("Training Faster R-CNN...")
    global label2target, target2label, num_classes

    IMAGE_ROOT = 'images/images' if os.path.exists('images') else 'open-images-bus-trucks/images'
    DF_RAW = pd.read_csv('df.csv' if os.path.exists('df.csv') else 'open-images-bus-trucks/df.csv')

    label2target = {l: t + 1 for t, l in enumerate(DF_RAW['LabelName'].unique())}
    label2target['background'] = 0
    target2label = {t: l for l, t in label2target.items()}
    num_classes = len(label2target)

    trn_ids, val_ids = train_test_split(DF_RAW['ImageID'].unique(), test_size=0.1, random_state=99)
    trn_df, val_df = DF_RAW[DF_RAW['ImageID'].isin(trn_ids)], DF_RAW[DF_RAW['ImageID'].isin(val_ids)]

    train_ds = OpenDataset(trn_df, IMAGE_ROOT, model_type='frcnn')
    test_ds = OpenDataset(val_df, IMAGE_ROOT, model_type='frcnn')

    train_loader = DataLoader(train_ds, batch_size=4, collate_fn=train_ds.collate_fn, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=4, collate_fn=test_ds.collate_fn, drop_last=True)

    def get_model():
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model

    def train_batch(inputs, model, optimizer):
        model.train()
        input, targets = inputs
        input = [image.to(device) for image in input]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        optimizer.zero_grad()
        losses = model(input, targets)
        loss = sum(loss for loss in losses.values())
        loss.backward()
        optimizer.step()
        return loss, losses

    @torch.no_grad()
    def validate_batch(inputs, model):
        model.train()
        input, targets = inputs
        input = [image.to(device) for image in input]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        losses = model(input, targets)
        loss = sum(loss for loss in losses.values())
        return loss, losses

    model = get_model().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    n_epochs = 5

    for epoch in range(n_epochs):
        # Training loop
        for ix, inputs in enumerate(train_loader):
            loss, losses = train_batch(inputs, model, optimizer)
            if ix % 10 == 0:
                print(f'Epoch {epoch + 1}/{n_epochs}, Batch {ix}, Loss: {loss.item():.4f}')

        # Validation loop
        val_losses = []
        for ix, inputs in enumerate(test_loader):
            loss, _ = validate_batch(inputs, model)
            val_losses.append(loss.item())
        print(f'Epoch {epoch + 1}/{n_epochs}, Val Loss: {np.mean(val_losses):.4f}')

    # Test inference
    model.eval()

    def decode_output(output):
        bbs = output['boxes'].cpu().detach().numpy().astype(np.uint16)
        labels = np.array([target2label[i] for i in output['labels'].cpu().detach().numpy()])
        confs = output['scores'].cpu().detach().numpy()
        ixs = nms(torch.tensor(bbs.astype(np.float32)), torch.tensor(confs), 0.05)
        bbs, confs, labels = [tensor[ixs] for tensor in [bbs, confs, labels]]
        if len(ixs) == 1:
            bbs, confs, labels = [np.array([tensor]) for tensor in [bbs, confs, labels]]
        return bbs.tolist(), confs.tolist(), labels.tolist()

    for ix, (images, targets) in enumerate(test_loader):
        if ix == 3: break
        images = [im for im in images]
        outputs = model(images)
        for ix, output in enumerate(outputs):
            bbs, confs, labels = decode_output(output)
            print(f"Detected: {labels} with confidences {confs}")


# SSD functions
def train_ssd():
    """Train SSD model"""
    print("Training SSD...")
    # SSD-specific imports
    sys.path.append('ssd-utils')
    from model import SSD300, MultiBoxLoss
    from detect import detect

    DATA_ROOT = 'open-images-bus-trucks'
    IMAGE_ROOT = f'{DATA_ROOT}/images'
    DF_RAW = pd.read_csv(f'{DATA_ROOT}/df.csv')

    global label2target, target2label, num_classes
    label2target = {l: t + 1 for t, l in enumerate(DF_RAW['LabelName'].unique())}
    label2target['background'] = 0
    target2label = {t: l for l, t in label2target.items()}
    num_classes = len(label2target)

    trn_ids, val_ids = train_test_split(DF_RAW['ImageID'].unique(), test_size=0.1, random_state=99)
    trn_df, val_df = DF_RAW[DF_RAW['ImageID'].isin(trn_ids)], DF_RAW[DF_RAW['ImageID'].isin(val_ids)]

    train_ds = OpenDataset(trn_df, IMAGE_ROOT, img_size=(300, 300), model_type='ssd')
    test_ds = OpenDataset(val_df, IMAGE_ROOT, img_size=(300, 300), model_type='ssd')

    train_loader = DataLoader(train_ds, batch_size=4, collate_fn=train_ds.collate_fn, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=4, collate_fn=test_ds.collate_fn, drop_last=True)

    def train_batch(inputs, model, criterion, optimizer):
        model.train()
        images, boxes, labels = inputs
        _regr, _clss = model(images)
        loss = criterion(_regr, _clss, boxes, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss

    @torch.no_grad()
    def validate_batch(inputs, model, criterion):
        model.eval()
        images, boxes, labels = inputs
        _regr, _clss = model(images)
        loss = criterion(_regr, _clss, boxes, labels)
        return loss

    model = SSD300(num_classes, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy, device=device)
    n_epochs = 3

    for epoch in range(n_epochs):
        # Training loop
        for ix, inputs in enumerate(train_loader):
            loss = train_batch(inputs, model, criterion, optimizer)
            if ix % 10 == 0:
                print(f'Epoch {epoch + 1}/{n_epochs}, Batch {ix}, Loss: {loss.item():.4f}')

        # Validation loop
        val_losses = []
        for ix, inputs in enumerate(test_loader):
            loss = validate_batch(inputs, model, criterion)
            val_losses.append(loss.item())
        print(f'Epoch {epoch + 1}/{n_epochs}, Val Loss: {np.mean(val_losses):.4f}')

    # Test inference
    from torch_snippets import Glob, choose
    image_paths = Glob(f'{DATA_ROOT}/images/*')
    for _ in range(3):
        image_id = choose(test_ds.image_infos)
        img_path = os.path.join(IMAGE_ROOT, image_id)
        original_image = Image.open(img_path, mode='r')
        bbs, labels, scores = detect(original_image, model, min_score=0.9, max_overlap=0.5, top_k=200, device=device)
        labels = [target2label[c.item()] for c in labels]
        print(f"Detected: {labels} with scores {scores}")


# YOLO functions
def train_yolo():
    """Train YOLO model using darknet"""
    print("Training YOLO...")

    # Clone darknet if not exists
    if not os.path.exists('darknet'):
        subprocess.check_call(['git', 'clone', 'https://github.com/AlexeyAB/darknet'])
        os.chdir('darknet')

        # Configure Makefile
        with open('Makefile', 'r') as f:
            content = f.read()

        # Enable OPENCV
        content = content.replace('OPENCV=0', 'OPENCV=1')

        # Enable GPU if available
        if torch.cuda.is_available():
            content = content.replace('GPU=0', 'GPU=1')
            content = content.replace('CUDNN=0', 'CUDNN=1')
            content = content.replace('CUDNN_HALF=0', 'CUDNN_HALF=1')

        with open('Makefile', 'w') as f:
            f.write(content)

        subprocess.check_call(['make'])

        # Download dataset
        subprocess.check_call(
            ['wget', '--quiet', 'https://www.dropbox.com/s/agmzwk95v96ihic/open-images-bus-trucks.tar.xz'])
        subprocess.check_call(['tar', '-xf', 'open-images-bus-trucks.tar.xz'])
        os.remove('open-images-bus-trucks.tar.xz')

        # Download pretrained weights
        subprocess.check_call(['wget', '--quiet',
                               'https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights'])

    os.chdir('darknet')

    # Create configuration files
    with open('data/obj.names', 'w') as f:
        f.write('bus\ntruck\n')

    with open('data/obj.data', 'w') as f:
        f.write('classes = 2\ntrain = data/train.txt\nvalid = data/val.txt\nnames = data/obj.names\nbackup = backup/\n')

    # Prepare data
    os.makedirs('data/obj', exist_ok=True)
    subprocess.check_call(['cp', '-r', '../open-images-bus-trucks/images/*', 'data/obj/'])
    subprocess.check_call(['cp', '-r', '../open-images-bus-trucks/yolo_labels/all/train.txt', 'data/'])
    subprocess.check_call(['cp', '-r', '../open-images-bus-trucks/yolo_labels/all/val.txt', 'data/'])
    subprocess.check_call(['cp', '-r', '../open-images-bus-trucks/yolo_labels/all/labels/*.txt', 'data/obj/'])

    # Configure YOLO
    subprocess.check_call(['cp', 'cfg/yolov4-tiny-custom.cfg', 'cfg/yolov4-tiny-bus-trucks.cfg'])

    # Update configuration
    config_updates = {
        'max_batches = 500200': 'max_batches=4000',
        'subdivisions=1': 'subdivisions=16',
        'steps=400000,450000': 'steps=3200,3600',
        'classes=80': 'classes=2',
        'filters=255': 'filters=21',
        'filters=57': 'filters=33'
    }

    with open('cfg/yolov4-tiny-bus-trucks.cfg', 'r') as f:
        content = f.read()

    for old, new in config_updates.items():
        content = content.replace(old, new)

    with open('cfg/yolov4-tiny-bus-trucks.cfg', 'w') as f:
        f.write(content)

    # Download pretrained weights
    subprocess.check_call(['wget', '--quiet',
                           'https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.conv.29'])
    subprocess.check_call(['cp', 'yolov4-tiny.conv.29', 'build/darknet/x64/'])

    # Train YOLO
    subprocess.check_call([
        './darknet', 'detector', 'train', 'data/obj.data',
        'cfg/yolov4-tiny-bus-trucks.cfg', 'yolov4-tiny.conv.29',
        '-dont_show', '-map'
    ])

    # Test with sample images
    from torch_snippets import Glob, stem
    image_paths = Glob('../images-of-trucks-and-busses/*')
    for f in image_paths:
        subprocess.check_call([
            './darknet', 'detector', 'test', 'data/obj.data',
            'cfg/yolov4-tiny-bus-trucks.cfg',
            'backup/yolov4-tiny-bus-trucks_4000.weights', f
        ])
        subprocess.check_call(['mv', 'predictions.jpg', f'{stem(f)}_pred.jpg'])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train object detection models')
    parser.add_argument('--model', choices=['frcnn', 'ssd', 'yolo', 'all'],
                        default='frcnn', help='Model to train')
    parser.add_argument('--setup', action='store_true', help='Setup environment only')

    args = parser.parse_args()

    if args.setup:
        setup_environment()
        print("Environment setup completed.")
        sys.exit(0)

    setup_environment()

    if args.model in ['frcnn', 'all']:
        train_faster_rcnn()

    if args.model in ['ssd', 'all']:
        train_ssd()

    if args.model in ['yolo', 'yolo', 'all']:
        train_yolo()