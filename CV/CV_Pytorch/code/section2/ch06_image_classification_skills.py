# -*- coding: utf-8 -*-
# @project: LargeDemo
# @filename: ch06_image_classification_skills.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/8/26 19:52
# https://chat.deepseek.com/a/chat/s/600fd407-31fc-4821-b8f1-e25ed7684380
import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from sklearn.model_selection import train_test_split
from torch_snippets import *

# ==================== CONFIGURATION ====================
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ==================== COMMON COMPONENTS ====================
def convBlock(ni, no, use_bn=True, dropout=0.2):
    layers = [
        nn.Dropout(dropout),
        nn.Conv2d(ni, no, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    ]
    if use_bn:
        layers.append(nn.BatchNorm2d(no))
    layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class CustomClassifier(nn.Module):
    def __init__(self, backbone, num_classes, loss_fn=nn.CrossEntropyLoss()):
        super().__init__()
        self.model = backbone
        self.loss_fn = loss_fn

    def forward(self, x):
        return self.model(x)

    def compute_metrics(self, preds, targets):
        loss = self.loss_fn(preds, targets)
        acc = (torch.max(preds, 1)[1] == targets).float().mean()
        return loss, acc


def train_batch(model, data, optimizer, criterion):
    model.train()
    if len(data) == 3:
        ims, labels, _ = data
    else:
        ims, labels = data
    _preds = model(ims)
    optimizer.zero_grad()
    loss, acc = criterion(_preds, labels)
    loss.backward()
    optimizer.step()
    return loss.item(), acc.item()


@torch.no_grad()
def validate_batch(model, data, criterion):
    model.eval()
    if len(data) == 3:
        ims, labels, _ = data
    else:
        ims, labels = data
    _preds = model(ims)
    loss, acc = criterion(_preds, labels)
    return loss.item(), acc.item()


def train_model(model, trn_dl, val_dl, optimizer, criterion, n_epochs, lr_schedule=None):
    log = Report(n_epochs)
    for ex in range(n_epochs):
        if lr_schedule and ex in lr_schedule:
            new_lr = lr_schedule[ex]
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
            print(f'Learning rate changed to {new_lr} at epoch {ex + 1}')

        N = len(trn_dl)
        for bx, data in enumerate(trn_dl):
            loss, acc = train_batch(model, data, optimizer, criterion)
            log.record(ex + (bx + 1) / N, trn_loss=loss, trn_acc=acc, end='\r')

        N = len(val_dl)
        for bx, data in enumerate(val_dl):
            loss, acc = validate_batch(model, data, criterion)
            log.record(ex + (bx + 1) / N, val_loss=loss, val_acc=acc, end='\r')

        log.report_avgs(ex + 1)
    return log


# ==================== MALARIA CLASSIFICATION ====================
def setup_malaria_data():
    if not os.path.exists('cell_images'):
        os.system('pip install -U -q torch_snippets')
        os.system('wget -q ftp://lhcftp.nlm.nih.gov/Open-Access-Datasets/Malaria/cell_images.zip')
        os.system('unzip -qq cell_images.zip')
        os.system('rm cell_images.zip')

    id2int = {'Parasitized': 0, 'Uninfected': 1}

    trn_tfms = T.Compose([
        T.ToPILImage(),
        T.Resize(128),
        T.CenterCrop(128),
        T.ColorJitter(brightness=(0.95, 1.05), contrast=(0.95, 1.05),
                      saturation=(0.95, 1.05), hue=0.05),
        T.RandomAffine(5, translate=(0.01, 0.1)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    val_tfms = T.Compose([
        T.ToPILImage(),
        T.Resize(128),
        T.CenterCrop(128),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    class MalariaImages(Dataset):
        def __init__(self, files, transform=None):
            self.files = files
            self.transform = transform
            logger.info(len(self))

        def __len__(self):
            return len(self.files)

        def __getitem__(self, ix):
            fpath = self.files[ix]
            clss = fname(parent(fpath))
            img = read(fpath, 1)
            return img, clss

        def choose(self):
            return self[randint(len(self))]

        def collate_fn(self, batch):
            _imgs, classes = list(zip(*batch))
            if self.transform:
                imgs = [self.transform(img)[None] for img in _imgs]
            classes = [torch.tensor([id2int[clss]]) for clss in classes]
            imgs, classes = [torch.cat(i).to(device) for i in [imgs, classes]]
            return imgs, classes, _imgs

    all_files = Glob('cell_images/*/*.png')
    np.random.seed(10)
    np.random.shuffle(all_files)
    trn_files, val_files = train_test_split(all_files, random_state=1)
    trn_ds = MalariaImages(trn_files, transform=trn_tfms)
    val_ds = MalariaImages(val_files, transform=val_tfms)
    trn_dl = DataLoader(trn_ds, 32, shuffle=True, collate_fn=trn_ds.collate_fn)
    val_dl = DataLoader(val_ds, 32, shuffle=False, collate_fn=val_ds.collate_fn)

    return trn_dl, val_dl, id2int


def create_malaria_model(num_classes):
    backbone = nn.Sequential(
        convBlock(3, 64),
        convBlock(64, 64),
        convBlock(64, 128),
        convBlock(128, 256),
        convBlock(256, 512),
        convBlock(512, 64),
        nn.Flatten(),
        nn.Linear(256, 256),
        nn.Dropout(0.2),
        nn.ReLU(inplace=True),
        nn.Linear(256, num_classes)
    )
    return CustomClassifier(backbone, num_classes)


def run_malaria_classification():
    print("Setting up Malaria data...")
    trn_dl, val_dl, id2int = setup_malaria_data()
    model = create_malaria_model(len(id2int)).to(device)
    criterion = model.compute_metrics
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print("Training Malaria classifier...")
    log = train_model(model, trn_dl, val_dl, optimizer, criterion, n_epochs=2)
    log.plot_epochs()

    return model, val_dl


def run_malaria_gradcam(model, val_ds):
    im2fmap = nn.Sequential(*(list(model.model[:5].children()) + list(model.model[5][:2].children())))

    def im2gradCAM(x):
        model.eval()
        logits = model(x)
        heatmaps = []
        activations = im2fmap(x)
        pred = logits.max(-1)[-1]
        model.zero_grad()
        logits[0, pred].backward(retain_graph=True)
        pooled_grads = model.model[-6][1].weight.grad.data.mean((1, 2, 3))
        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_grads[i]
        heatmap = torch.mean(activations, dim=1)[0].cpu().detach()
        return heatmap, 'Uninfected' if pred.item() else 'Parasitized'

    SZ = 128

    def upsampleHeatmap(map, img):
        m, M = map.min(), map.max()
        map = 255 * ((map - m) / (M - m))
        map = np.uint8(map)
        map = cv2.resize(map, (SZ, SZ))
        map = cv2.applyColorMap(255 - map, cv2.COLORMAP_JET)
        map = np.uint8(map)
        map = np.uint8(map * 0.7 + img * 0.3)
        return map

    N = 20
    _val_dl = DataLoader(val_ds, batch_size=N, shuffle=True, collate_fn=val_ds.collate_fn)
    x, y, z = next(iter(_val_dl))

    for i in range(N):
        image = resize(z[i], SZ)
        heatmap, pred = im2gradCAM(x[i:i + 1])
        if pred == 'Uninfected':
            continue
        heatmap = upsampleHeatmap(heatmap, image)
        subplots([image, heatmap], nc=2, figsize=(5, 3), suptitle=pred)


# ==================== ROAD SIGN DETECTION ====================
def setup_gtsrb_data(augment=False, use_bn=True):
    if not os.path.exists('GTSRB'):
        os.system('pip install -U -q torch_snippets')
        os.system(
            'wget -qq https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip')
        os.system(
            'wget -qq https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip')
        os.system('unzip -qq GTSRB_Final_Training_Images.zip')
        os.system('unzip -qq GTSRB_Final_Test_Images.zip')
        os.system(
            'wget https://raw.githubusercontent.com/georgesung/traffic_sign_classification_german/master/signnames.csv')
        os.system('rm GTSRB_Final_Training_Images.zip GTSRB_Final_Test_Images.zip')

    classIds = pd.read_csv('signnames.csv')
    classIds.set_index('ClassId', inplace=True)
    classIds = classIds.to_dict()['SignName']
    classIds = {f'{k:05d}': v for k, v in classIds.items()}
    id2int = {v: ix for ix, (k, v) in enumerate(classIds.items())}

    trn_tfms_list = [
        T.ToPILImage(),
        T.Resize(32),
        T.CenterCrop(32),
    ]

    if augment:
        trn_tfms_list.extend([
            T.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2),
                          saturation=(0.8, 1.2), hue=0.25),
            T.RandomAffine(5, translate=(0.01, 0.1)),
        ])

    trn_tfms_list.extend([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    trn_tfms = T.Compose(trn_tfms_list)

    val_tfms = T.Compose([
        T.ToPILImage(),
        T.Resize(32),
        T.CenterCrop(32),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    class GTSRB(Dataset):
        def __init__(self, files, transform=None):
            self.files = files
            self.transform = transform
            logger.info(len(self))

        def __len__(self):
            return len(self.files)

        def __getitem__(self, ix):
            fpath = self.files[ix]
            clss = fname(parent(fpath))
            img = read(fpath, 1)
            return img, classIds[clss]

        def choose(self):
            return self[randint(len(self))]

        def collate_fn(self, batch):
            imgs, classes = list(zip(*batch))
            if self.transform:
                imgs = [self.transform(img)[None] for img in imgs]
            classes = [torch.tensor([id2int[clss]]) for clss in classes]
            imgs, classes = [torch.cat(i).to(device) for i in [imgs, classes]]
            return imgs, classes

    all_files = Glob('GTSRB/Final_Training/Images/*/*.ppm')
    np.random.seed(10)
    np.random.shuffle(all_files)
    trn_files, val_files = train_test_split(all_files, random_state=1)
    trn_ds = GTSRB(trn_files, transform=trn_tfms)
    val_ds = GTSRB(val_files, transform=val_tfms)
    trn_dl = DataLoader(trn_ds, 32, shuffle=True, collate_fn=trn_ds.collate_fn)
    val_dl = DataLoader(val_ds, 32, shuffle=False, collate_fn=val_ds.collate_fn)

    return trn_dl, val_dl, id2int


def create_gtsrb_model(num_classes, use_bn=True):
    backbone = nn.Sequential(
        convBlock(3, 64, use_bn=use_bn),
        convBlock(64, 64, use_bn=use_bn),
        convBlock(64, 128, use_bn=use_bn),
        convBlock(128, 64, use_bn=use_bn),
        nn.Flatten(),
        nn.Linear(256, 256),
        nn.Dropout(0.2),
        nn.ReLU(inplace=True),
        nn.Linear(256, num_classes)
    )
    return CustomClassifier(backbone, num_classes)


def run_road_sign_detection(augment=False, use_bn=True):
    print("Setting up GTSRB data...")
    trn_dl, val_dl, id2int = setup_gtsrb_data(augment=augment, use_bn=use_bn)
    model = create_gtsrb_model(len(id2int), use_bn=use_bn).to(device)
    criterion = model.compute_metrics
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print("Training Road Sign classifier...")
    lr_schedule = {10: 1e-4}  # Change learning rate at epoch 10
    log = train_model(model, trn_dl, val_dl, optimizer, criterion, n_epochs=40, lr_schedule=lr_schedule)

    log_name = f"{'aug' if augment else 'no-aug'}_{'bn' if use_bn else 'no-bn'}.log"
    dumpdill(log, log_name)
    log.plot_epochs()

    return log


# ==================== MAIN EXECUTION ====================
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run computer vision tasks')
    parser.add_argument('--task', type=str, required=True,
                        choices=['malaria', 'malaria_gradcam', 'road_sign'],
                        help='Task to run: malaria classification, gradcam, or road sign detection')
    parser.add_argument('--augment', action='store_true', help='Use data augmentation for road signs')
    parser.add_argument('--use_bn', action='store_true', help='Use batch normalization for road signs')
    args = parser.parse_args()

    if args.task == 'malaria':
        model, val_dl = run_malaria_classification()
    elif args.task == 'malaria_gradcam':
        # Need to have a trained model first
        trn_dl, val_dl, id2int = setup_malaria_data()
        model = create_malaria_model(len(id2int)).to(device)
        # In practice, you would load a pre-trained model here
        # model.load_state_dict(torch.load('malaria_model.pth'))
        run_malaria_gradcam(model, val_dl.dataset)
    elif args.task == 'road_sign':
        run_road_sign_detection(augment=args.augment, use_bn=args.use_bn)