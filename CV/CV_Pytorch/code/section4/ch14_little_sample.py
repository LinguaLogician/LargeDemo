# -*- coding: utf-8 -*-
# @project: LargeDemo
# @filename: ch14_little_sample.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/8/27 9:59
# https://chat.deepseek.com/a/chat/s/47bb135d-9fbb-443d-bec0-fa97e56429c3
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision
import numpy as np
from sklearn.preprocessing import normalize
import gzip
import _pickle as cPickle
import matplotlib.pyplot as plt
import os
import sys
from torch_snippets import *

# ==================== Siamese Network ====================

def setup_siamese_data(folder_path, transform=None):
    class SiameseNetworkDataset(Dataset):
        def __init__(self, folder, transform=None):
            self.folder = folder
            self.items = Glob(f'{self.folder}/*/*')
            self.transform = transform

        def __getitem__(self, ix):
            itemA = self.items[ix]
            person = fname(parent(itemA))
            same_person = randint(2)
            if same_person:
                itemB = choose(Glob(f'{self.folder}/{person}/*', silent=True))
            else:
                while True:
                    itemB = choose(self.items)
                    if person != fname(parent(itemB)):
                        break
            imgA = read(itemA)
            imgB = read(itemB)
            if self.transform:
                imgA = self.transform(imgA)
                imgB = self.transform(imgB)
            return imgA, imgB, np.array([1 - same_person])

        def __len__(self):
            return len(self.items)

    return SiameseNetworkDataset(folder_path, transform)

def get_siamese_transforms():
    trn_tfms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(5, (0.01, 0.2), scale=(0.9, 1.1)),
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])
    val_tfms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])
    return trn_tfms, val_tfms

def build_siamese_model():
    def convBlock(ni, no):
        return nn.Sequential(
            nn.Dropout(0.2),
            nn.Conv2d(ni, no, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(no),
        )

    class SiameseNetwork(nn.Module):
        def __init__(self):
            super(SiameseNetwork, self).__init__()
            self.features = nn.Sequential(
                convBlock(1, 4),
                convBlock(4, 8),
                convBlock(8, 8),
                nn.Flatten(),
                nn.Linear(8 * 100 * 100, 500), nn.ReLU(inplace=True),
                nn.Linear(500, 500), nn.ReLU(inplace=True),
                nn.Linear(500, 5)
            )

        def forward(self, input1, input2):
            output1 = self.features(input1)
            output2 = self.features(input2)
            return output1, output2

    return SiameseNetwork()

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        acc = ((euclidean_distance > 0.6) == label).float().mean()
        return loss_contrastive, acc

def train_siamese_batch(model, data, optimizer, criterion):
    imgsA, imgsB, labels = [t.to(device) for t in data]
    optimizer.zero_grad()
    codesA, codesB = model(imgsA, imgsB)
    loss, acc = criterion(codesA, codesB, labels)
    loss.backward()
    optimizer.step()
    return loss.item(), acc.item()

@torch.no_grad()
def validate_siamese_batch(model, data, criterion):
    imgsA, imgsB, labels = [t.to(device) for t in data]
    codesA, codesB = model(imgsA, imgsB)
    loss, acc = criterion(codesA, codesB, labels)
    return loss.item(), acc.item()

def train_siamese_network(trn_dl, val_dl, n_epochs=200):
    model = build_siamese_model().to(device)
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    log = Report(n_epochs)

    for epoch in range(n_epochs):
        N = len(trn_dl)
        for i, data in enumerate(trn_dl):
            loss, acc = train_siamese_batch(model, data, optimizer, criterion)
            log.record(epoch + (1 + i) / N, trn_loss=loss, trn_acc=acc, end='\r')
        N = len(val_dl)
        for i, data in enumerate(val_dl):
            loss, acc = validate_siamese_batch(model, data, criterion)
            log.record(epoch + (1 + i) / N, val_loss=loss, val_acc=acc, end='\r')
        if (epoch + 1) % 20 == 0:
            log.report_avgs(epoch + 1)
        if epoch == 10:
            optimizer = optim.Adam(model.parameters(), lr=0.0005)

    log.plot_epochs(['trn_loss', 'val_loss'], log=True, title='Variation in training and validation loss')
    log.plot_epochs(['trn_acc', 'val_acc'], title='Variation in training and validation accuracy')
    return model

def evaluate_siamese_model(model, val_ds):
    model.eval()
    val_dl = DataLoader(val_ds, num_workers=6, batch_size=1, shuffle=True)
    dataiter = iter(val_dl)
    x0, _, _ = next(dataiter)

    for i in range(2):
        _, x1, label2 = next(dataiter)
        concatenated = torch.cat((x0 * 0.5 + 0.5, x1 * 0.5 + 0.5), 0)
        output1, output2 = model(x0.to(device), x1.to(device))
        euclidean_distance = F.pairwise_distance(output1, output2)
        output = 'Same Face' if euclidean_distance.item() < 0.6 else 'Different'
        show(torchvision.utils.make_grid(concatenated),
             title='Dissimilarity: {:.2f}\n{}'.format(euclidean_distance.item(), output))
        plt.show()

def run_siamese_network():
    print("Running Siamese Network...")
    trn_tfms, val_tfms = get_siamese_transforms()
    trn_ds = setup_siamese_data(folder="./data/faces/training/", transform=trn_tfms)
    val_ds = setup_siamese_data(folder="./data/faces/testing/", transform=val_tfms)
    trn_dl = DataLoader(trn_ds, shuffle=True, batch_size=64)
    val_dl = DataLoader(val_ds, shuffle=False, batch_size=64)
    model = train_siamese_network(trn_dl, val_dl, n_epochs=200)
    evaluate_siamese_model(model, val_ds)

# ==================== Zero-Shot Learning ====================

def load_zero_shot_data(data_path, word2vec_path, train_classes_file):
    with open(train_classes_file, 'r') as infile:
        train_classes = [str.strip(line) for line in infile]

    with gzip.GzipFile(data_path, 'rb') as infile:
        data = cPickle.load(infile)

    training_data = [instance for instance in data if instance[0] in train_classes]
    zero_shot_data = [instance for instance in data if instance[0] not in train_classes]
    np.random.shuffle(training_data)

    train_size = 300  # per class
    train_data, valid_data = [], []
    for class_label in train_classes:
        ctr = 0
        for instance in training_data:
            if instance[0] == class_label:
                if ctr < train_size:
                    train_data.append(instance)
                    ctr += 1
                else:
                    valid_data.append(instance)

    np.random.shuffle(train_data)
    np.random.shuffle(valid_data)
    vectors = {i: j for i, j in np.load(word2vec_path, allow_pickle=True)}

    train_data = [(feat, vectors[clss]) for clss, feat in train_data]
    valid_data = [(feat, vectors[clss]) for clss, feat in valid_data]

    train_clss = [clss for clss, feat in train_data]
    valid_clss = [clss for clss, feat in valid_data]
    zero_shot_clss = [clss for clss, feat in zero_shot_data]

    x_train, y_train = zip(*train_data)
    x_train, y_train = np.squeeze(np.asarray(x_train)), np.squeeze(np.asarray(y_train))
    x_train = normalize(x_train, norm='l2')

    x_valid, y_valid = zip(*valid_data)
    x_valid, y_valid = np.squeeze(np.asarray(x_valid)), np.squeeze(np.asarray(y_valid))
    x_valid = normalize(x_valid, norm='l2')

    y_zsl, x_zsl = zip(*zero_shot_data)
    x_zsl, y_zsl = np.squeeze(np.asarray(x_zsl)), np.squeeze(np.asarray(y_zsl))
    x_zsl = normalize(x_zsl, norm='l2')

    return (x_train, y_train), (x_valid, y_valid), (x_zsl, y_zsl), zero_shot_clss, vectors

def build_zsl_model():
    return nn.Sequential(
        nn.Linear(4096, 1024), nn.ReLU(inplace=True),
        nn.BatchNorm1d(1024), nn.Dropout(0.8),
        nn.Linear(1024, 512), nn.ReLU(inplace=True),
        nn.BatchNorm1d(512), nn.Dropout(0.8),
        nn.Linear(512, 256), nn.ReLU(inplace=True),
        nn.BatchNorm1d(256), nn.Dropout(0.8),
        nn.Linear(256, 300)
    )

def train_zsl_batch(model, data, optimizer, criterion):
    ims, labels = data
    _preds = model(ims)
    optimizer.zero_grad()
    loss = criterion(_preds, labels)
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def validate_zsl_batch(model, data, criterion):
    ims, labels = data
    _preds = model(ims)
    loss = criterion(_preds, labels)
    return loss.item()

def train_zero_shot_learning(trn_dl, val_dl, n_epochs=60):
    model = build_zsl_model().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    log = Report(n_epochs)

    for ex in range(n_epochs):
        N = len(trn_dl)
        for bx, data in enumerate(trn_dl):
            loss = train_zsl_batch(model, data, optimizer, criterion)
            log.record(ex + (bx + 1) / N, trn_loss=loss, end='\r')

        N = len(val_dl)
        for bx, data in enumerate(val_dl):
            loss = validate_zsl_batch(model, data, criterion)
            log.record(ex + (bx + 1) / N, val_loss=loss, end='\r')

        if ex == 10:
            optimizer = optim.Adam(model.parameters(), lr=1e-4)
        if ex == 40:
            optimizer = optim.Adam(model.parameters(), lr=1e-5)
        if not (ex + 1) % 10:
            log.report_avgs(ex + 1)

    log.plot(log=True)
    return model

def evaluate_zsl_model(model, x_zsl, zero_shot_clss, class_vectors_path):
    pred_zsl = model(torch.Tensor(x_zsl).to(device)).cpu().detach().numpy()
    class_vectors = sorted(np.load(class_vectors_path, allow_pickle=True), key=lambda x: x[0])
    classnames, vectors = zip(*class_vectors)
    classnames = list(classnames)
    vectors = np.array(vectors)

    dists = (pred_zsl[None] - vectors[:, None])
    dists = (dists ** 2).sum(-1).T

    best_classes = []
    for item in dists:
        best_classes.append([classnames[j] for j in np.argsort(item)[:5]])

    accuracy = np.mean([i in J for i, J in zip(zero_shot_clss, best_classes)])
    print(f"Zero-shot learning accuracy: {accuracy:.4f}")

def run_zero_shot_learning():
    print("Running Zero-Shot Learning...")
    data_path = "../data/zeroshot_data.pkl"
    word2vec_path = "../data/class_vectors.npy"
    train_classes_file = "train_classes.txt"

    (x_train, y_train), (x_valid, y_valid), (x_zsl, y_zsl), zero_shot_clss, vectors = load_zero_shot_data(
        data_path, word2vec_path, train_classes_file)

    trn_ds = torch.utils.data.TensorDataset(torch.Tensor(x_train).to(device), torch.Tensor(y_train).to(device))
    val_ds = torch.utils.data.TensorDataset(torch.Tensor(x_valid).to(device), torch.Tensor(y_valid).to(device))
    trn_dl = DataLoader(trn_ds, batch_size=32, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=32, shuffle=False)

    model = train_zero_shot_learning(trn_dl, val_dl, n_epochs=60)
    evaluate_zsl_model(model, x_zsl, zero_shot_clss, word2vec_path)

# ==================== Main ====================

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Uncomment the function you want to run:

    # Run Siamese Network
    # run_siamese_network()

    # Run Zero-Shot Learning
    # run_zero_shot_learning()