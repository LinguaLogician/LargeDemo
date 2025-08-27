# -*- coding: utf-8 -*-
# @project: LargeDemo
# @filename: ch05_image_classification_transfer_learning.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/8/26 18:24

# https://chat.deepseek.com/a/chat/s/51adf510-5122-400d-9ce7-8ba1c9a4ac72

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import cv2
import glob
import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from copy import deepcopy
import time
from google.colab import files
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ==================== 数据准备与下载 ====================

def setup_kaggle():
    """设置Kaggle API"""
    # !pip
    # install - q
    # kaggle
    # files.upload()
    # !mkdir - p
    # ~ /.kaggle
    # !cp
    # kaggle.json
    # ~ /.kaggle /
    # !ls
    # ~ /.kaggle
    # !chmod
    # 600 / root /.kaggle / kaggle.json
    pass


def download_cats_dogs_dataset():
    """下载猫狗数据集"""
    # !kaggle
    # datasets
    # download - d
    # tongpython / cat - and -dog
    # !unzip
    # cat - and -dog.zip
    pass


def setup_google_drive():
    """设置Google Drive认证"""
    auth.authenticate_user()
    gauth = GoogleAuth()
    gauth.credentials = GoogleCredentials.get_application_default()
    return GoogleDrive(gauth)


def download_from_drive(drive, file_id, name):
    """从Google Drive下载文件"""
    downloaded = drive.CreateFile({'id': file_id})
    downloaded.GetContentFile(name)


def download_fairface_dataset():
    """下载FairFace数据集"""
    drive = setup_google_drive()
    download_from_drive(drive, '1Z1RqRo0_JiavaZw2yzZG6WETdZQ8qX86', 'fairface-img-margin025-trainval.zip')
    download_from_drive(drive, '1k5vvyREmHDW5TSM9QgB04Bvc8C8_7dl-', 'fairface-label-train.csv')
    download_from_drive(drive, '1_rtz1M1zhvS0d5vVoXUamnohB6cJ02iJ', 'fairface-label-val.csv')
    # !unzip - qq
    # fairface - img - margin025 - trainval.zip


# ==================== 数据集类定义 ====================

class CatsDogs(Dataset):
    """猫狗分类数据集类"""

    def __init__(self, folder):
        cats = glob.glob(folder + '/cats/*.jpg')
        dogs = glob.glob(folder + '/dogs/*.jpg')
        self.fpaths = cats[:500] + dogs[:500]
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        from random import shuffle, seed;
        seed(10);
        shuffle(self.fpaths)
        self.targets = [fpath.split('/')[-1].startswith('dog') for fpath in self.fpaths]

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, ix):
        f = self.fpaths[ix]
        target = self.targets[ix]
        im = (cv2.imread(f)[:, :, ::-1])
        im = cv2.resize(im, (224, 224))
        im = torch.tensor(im / 255)
        im = im.permute(2, 0, 1)
        im = self.normalize(im)
        return im.float().to(device), torch.tensor([target]).float().to(device)


class FacesData(Dataset):
    """面部关键点检测数据集类"""

    def __init__(self, df):
        super(FacesData, self).__init__()
        self.df = df
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, ix):
        img_path = 'P1_Facial_Keypoints/data/training/' + self.df.iloc[ix, 0]
        img = cv2.imread(img_path) / 255.
        kp = deepcopy(self.df.iloc[ix, 1:].tolist())
        kp_x = (np.array(kp[0::2]) / img.shape[1]).tolist()
        kp_y = (np.array(kp[1::2]) / img.shape[0]).tolist()
        kp2 = kp_x + kp_y
        kp2 = torch.tensor(kp2)
        img = self.preprocess_input(img)
        return img, kp2

    def preprocess_input(self, img):
        img = cv2.resize(img, (224, 224))
        img = torch.tensor(img).permute(2, 0, 1)
        img = self.normalize(img).float()
        return img.to(device)

    def load_img(self, ix):
        img_path = 'P1_Facial_Keypoints/data/training/' + self.df.iloc[ix, 0]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.
        img = cv2.resize(img, (224, 224))
        return img


class GenderAgeClass(Dataset):
    """年龄性别分类数据集类"""

    def __init__(self, df, tfms=None):
        self.df = df
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, ix):
        f = self.df.iloc[ix].squeeze()
        file = f.file
        gen = f.gender == 'Female'
        age = f.age
        im = cv2.imread(file)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        return im, age, gen

    def preprocess_image(self, im):
        im = cv2.resize(im, (224, 224))
        im = torch.tensor(im).permute(2, 0, 1)
        im = self.normalize(im / 255.)
        return im[None]

    def collate_fn(self, batch):
        """预处理图像、年龄和性别"""
        ims, ages, genders = [], [], []
        for im, age, gender in batch:
            im = self.preprocess_image(im)
            ims.append(im)
            ages.append(float(int(age) / 80))
            genders.append(float(gender))

        ages, genders = [torch.tensor(x).to(device).float() for x in [ages, genders]]
        ims = torch.cat(ims).to(device)
        return ims, ages, genders


# ==================== 模型定义 ====================

def get_vgg16_model():
    """获取VGG16模型（用于猫狗分类）"""
    model = models.vgg16(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    model.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 1),
        nn.Sigmoid()
    )
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    return model.to(device), loss_fn, optimizer


def get_resnet18_model():
    """获取ResNet18模型（用于猫狗分类）"""
    model = models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    model.fc = nn.Sequential(
        nn.Flatten(),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 1),
        nn.Sigmoid()
    )
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    return model.to(device), loss_fn, optimizer


def get_vgg16_facial_keypoints_model():
    """获取VGG16模型（用于面部关键点检测）"""
    model = models.vgg16(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.avgpool = nn.Sequential(
        nn.Conv2d(512, 512, 3),
        nn.MaxPool2d(2),
        nn.Flatten()
    )
    model.classifier = nn.Sequential(
        nn.Linear(2048, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 136),
        nn.Sigmoid()
    )
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    return model.to(device), criterion, optimizer


def get_age_gender_model():
    """获取年龄性别预测模型"""
    model = models.vgg16(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.avgpool = nn.Sequential(
        nn.Conv2d(512, 512, kernel_size=3),
        nn.MaxPool2d(2),
        nn.ReLU(),
        nn.Flatten()
    )

    class AgeGenderClassifier(nn.Module):
        def __init__(self):
            super(AgeGenderClassifier, self).__init__()
            self.intermediate = nn.Sequential(
                nn.Linear(2048, 512),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(128, 64),
                nn.ReLU(),
            )
            self.age_classifier = nn.Sequential(
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
            self.gender_classifier = nn.Sequential(
                nn.Linear(64, 1),
                nn.Sigmoid()
            )

        def forward(self, x):
            x = self.intermediate(x)
            age = self.age_classifier(x)
            gender = self.gender_classifier(x)
            return gender, age

    model.classifier = AgeGenderClassifier()

    gender_criterion = nn.BCELoss()
    age_criterion = nn.L1Loss()
    loss_functions = gender_criterion, age_criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    return model.to(device), loss_functions, optimizer


# ==================== 训练函数 ====================

def train_batch_classification(x, y, model, opt, loss_fn):
    """训练批处理（分类任务）"""
    model.train()
    prediction = model(x)
    batch_loss = loss_fn(prediction, y)
    batch_loss.backward()
    opt.step()
    opt.zero_grad()
    return batch_loss.item()


def train_batch_facial_keypoints(img, kps, model, optimizer, criterion):
    """训练批处理（面部关键点检测）"""
    model.train()
    optimizer.zero_grad()
    _kps = model(img.to(device))
    loss = criterion(_kps, kps.to(device))
    loss.backward()
    optimizer.step()
    return loss


def train_batch_age_gender(data, model, optimizer, criteria):
    """训练批处理（年龄性别预测）"""
    model.train()
    ims, age, gender = data
    optimizer.zero_grad()
    pred_gender, pred_age = model(ims)
    gender_criterion, age_criterion = criteria
    gender_loss = gender_criterion(pred_gender.squeeze(), gender)
    age_loss = age_criterion(pred_age.squeeze(), age)
    total_loss = gender_loss + age_loss
    total_loss.backward()
    optimizer.step()
    return total_loss


# ==================== 验证/评估函数 ====================

@torch.no_grad()
def accuracy_classification(x, y, model):
    """计算分类准确率"""
    model.eval()
    prediction = model(x)
    is_correct = (prediction > 0.5) == y
    return is_correct.cpu().numpy().tolist()


@torch.no_grad()
def validate_batch_facial_keypoints(img, kps, model, criterion):
    """验证批处理（面部关键点检测）"""
    model.eval()
    _kps = model(img.to(device))
    loss = criterion(_kps, kps.to(device))
    return _kps, loss


@torch.no_grad()
def validate_batch_age_gender(data, model, criteria):
    """验证批处理（年龄性别预测）"""
    model.eval()
    ims, age, gender = data
    pred_gender, pred_age = model(ims)
    gender_criterion, age_criterion = criteria
    gender_loss = gender_criterion(pred_gender.squeeze(), gender)
    age_loss = age_criterion(pred_age.squeeze(), age)
    total_loss = gender_loss + age_loss
    pred_gender = (pred_gender > 0.5).squeeze()
    gender_acc = (pred_gender == gender).float().sum()
    age_mae = torch.abs(age - pred_age).float().sum()
    return total_loss, gender_acc, age_mae


# ==================== 数据加载器 ====================

def get_cats_dogs_data():
    """获取猫狗分类数据加载器"""
    train_data_dir = 'training_set/training_set'
    test_data_dir = 'test_set/test_set'

    train = CatsDogs(train_data_dir)
    trn_dl = DataLoader(train, batch_size=32, shuffle=True, drop_last=True)
    val = CatsDogs(test_data_dir)
    val_dl = DataLoader(val, batch_size=32, shuffle=True, drop_last=True)
    return trn_dl, val_dl


def get_facial_keypoints_data():
    """获取面部关键点检测数据"""
    # !git
    # clone
    # https: // github.com / udacity / P1_Facial_Keypoints.git
    root_dir = 'P1_Facial_Keypoints/data/training/'
    all_img_paths = glob.glob(os.path.join(root_dir, '*.jpg'))
    data = pd.read_csv('P1_Facial_Keypoints/data/training_frames_keypoints.csv')

    train, test = train_test_split(data, test_size=0.2, random_state=101)
    train_dataset = FacesData(train.reset_index(drop=True))
    test_dataset = FacesData(test.reset_index(drop=True))

    train_loader = DataLoader(train_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    return train_loader, test_loader


def get_age_gender_data():
    """获取年龄性别预测数据"""
    trn_df = pd.read_csv('fairface-label-train.csv')
    val_df = pd.read_csv('fairface-label-val.csv')

    trn = GenderAgeClass(trn_df)
    val = GenderAgeClass(val_df)

    train_loader = DataLoader(trn, batch_size=32, shuffle=True, drop_last=True, collate_fn=trn.collate_fn)
    test_loader = DataLoader(val, batch_size=32, collate_fn=val.collate_fn)

    return train_loader, test_loader


# ==================== 训练循环 ====================

def train_classification_model(model_name='vgg16', epochs=5):
    """训练分类模型（猫狗分类）"""
    # 设置和数据准备
    setup_kaggle()
    download_cats_dogs_dataset()

    # 获取模型和数据
    if model_name.lower() == 'vgg16':
        model, loss_fn, optimizer = get_vgg16_model()
    elif model_name.lower() == 'resnet18':
        model, loss_fn, optimizer = get_resnet18_model()
    else:
        raise ValueError("Unsupported model. Choose 'vgg16' or 'resnet18'")

    trn_dl, val_dl = get_cats_dogs_data()

    # 训练循环
    train_losses, train_accuracies = [], []
    val_accuracies = []

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_epoch_losses, train_epoch_accuracies = [], []
        val_epoch_accuracies = []

        # 训练阶段
        for ix, batch in enumerate(iter(trn_dl)):
            x, y = batch
            batch_loss = train_batch_classification(x, y, model, optimizer, loss_fn)
            train_epoch_losses.append(batch_loss)

        train_epoch_loss = np.array(train_epoch_losses).mean()

        # 训练准确率
        for ix, batch in enumerate(iter(trn_dl)):
            x, y = batch
            is_correct = accuracy_classification(x, y, model)
            train_epoch_accuracies.extend(is_correct)

        train_epoch_accuracy = np.mean(train_epoch_accuracies)

        # 验证准确率
        for ix, batch in enumerate(iter(val_dl)):
            x, y = batch
            val_is_correct = accuracy_classification(x, y, model)
            val_epoch_accuracies.extend(val_is_correct)

        val_epoch_accuracy = np.mean(val_epoch_accuracies)

        # 记录指标
        train_losses.append(train_epoch_loss)
        train_accuracies.append(train_epoch_accuracy)
        val_accuracies.append(val_epoch_accuracy)

        print(
            f"Train Loss: {train_epoch_loss:.4f}, Train Acc: {train_epoch_accuracy:.4f}, Val Acc: {val_epoch_accuracy:.4f}")

    # 绘制结果
    epochs_range = np.arange(epochs) + 1
    plt.plot(epochs_range, train_accuracies, 'bo', label='Training accuracy')
    plt.plot(epochs_range, val_accuracies, 'r', label='Validation accuracy')
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.title(f'Training and validation accuracy with {model_name.upper()}\nand 1K training data points')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.ylim(0.95, 1)
    plt.gca().set_yticklabels(['{:.0f}%'.format(x * 100) for x in plt.gca().get_yticks()])
    plt.legend()
    plt.grid('off')
    plt.show()

    return model, train_losses, train_accuracies, val_accuracies


def train_facial_keypoints_model(epochs=50):
    """训练面部关键点检测模型"""
    # 获取数据
    train_loader, test_loader = get_facial_keypoints_data()

    # 获取模型
    model, criterion, optimizer = get_vgg16_facial_keypoints_model()

    # 训练循环
    train_loss, test_loss = [], []

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        epoch_train_loss, epoch_test_loss = 0, 0

        # 训练阶段
        for ix, (img, kps) in enumerate(train_loader):
            loss = train_batch_facial_keypoints(img, kps, model, optimizer, criterion)
            epoch_train_loss += loss.item()

        epoch_train_loss /= (ix + 1)

        # 验证阶段
        for ix, (img, kps) in enumerate(test_loader):
            _, loss = validate_batch_facial_keypoints(img, kps, model, criterion)
            epoch_test_loss += loss.item()

        epoch_test_loss /= (ix + 1)

        # 记录损失
        train_loss.append(epoch_train_loss)
        test_loss.append(epoch_test_loss)

        print(f"Train Loss: {epoch_train_loss:.4f}, Test Loss: {epoch_test_loss:.4f}")

    # 绘制结果
    epochs_range = np.arange(epochs) + 1
    plt.plot(epochs_range, train_loss, 'bo', label='Training loss')
    plt.plot(epochs_range, test_loss, 'r', label='Test loss')
    plt.title('Training and Test loss over increasing epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid('off')
    plt.show()

    # 可视化结果
    ix = 0
    plt.figure(figsize=(10, 10))
    plt.subplot(221)
    plt.title('Original image')
    im = test_loader.dataset.load_img(ix)
    plt.imshow(im)
    plt.grid(False)
    plt.subplot(222)
    plt.title('Image with facial keypoints')
    x, _ = test_loader.dataset[ix]
    plt.imshow(im)
    kp = model(x[None]).flatten().detach().cpu()
    plt.scatter(kp[:68] * 224, kp[68:] * 224, c='r')
    plt.grid(False)
    plt.show()

    return model, train_loss, test_loss


def train_age_gender_model(epochs=5):
    """训练年龄性别预测模型"""
    # 下载数据
    download_fairface_dataset()

    # 获取数据
    train_loader, test_loader = get_age_gender_data()

    # 获取模型
    model, criteria, optimizer = get_age_gender_model()

    # 训练循环
    val_gender_accuracies = []
    val_age_maes = []
    train_losses = []
    val_losses = []

    best_test_loss = 1000
    start = time.time()

    for epoch in range(epochs):
        epoch_train_loss, epoch_test_loss = 0, 0
        val_age_mae, val_gender_acc, ctr = 0, 0, 0

        # 训练阶段
        for ix, data in enumerate(train_loader):
            loss = train_batch_age_gender(data, model, optimizer, criteria)
            epoch_train_loss += loss.item()

        epoch_train_loss /= len(train_loader)

        # 验证阶段
        for ix, data in enumerate(test_loader):
            loss, gender_acc, age_mae = validate_batch_age_gender(data, model, criteria)
            epoch_test_loss += loss.item()
            val_age_mae += age_mae
            val_gender_acc += gender_acc
            ctr += len(data[0])

        val_age_mae /= ctr
        val_gender_acc /= ctr
        epoch_test_loss /= len(test_loader)

        # 记录指标
        elapsed = time.time() - start
        best_test_loss = min(best_test_loss, epoch_test_loss)

        val_gender_accuracies.append(val_gender_acc.cpu().numpy())
        val_age_maes.append(val_age_mae.cpu().numpy())
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_test_loss)

        # 打印进度
        print('{}/{} ({:.2f}s - {:.2f}s remaining)'.format(
            epoch + 1, epochs, elapsed, (epochs - epoch) * (elapsed / (epoch + 1))))
        info = f'''Epoch: {epoch + 1:03d}\tTrain Loss: {epoch_train_loss:.3f}\tTest: {epoch_test_loss:.3f}\tBest Test Loss: {best_test_loss:.4f}'''
        info += f'\nGender Accuracy: {val_gender_acc * 100:.2f}%\tAge MAE: {val_age_mae:.2f}\n'
        print(info)

    # 绘制结果
    epochs_range = np.arange(1, len(val_gender_accuracies) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax = ax.flat
    ax[0].plot(epochs_range, val_gender_accuracies, 'bo')
    ax[1].plot(epochs_range, val_age_maes, 'r')
    ax[0].set_xlabel('Epochs')
    ax[1].set_xlabel('Epochs')
    ax[0].set_ylabel('Accuracy')
    ax[1].set_ylabel('MAE')
    ax[0].set_title('Validation Gender Accuracy')
    ax[1].set_title('Validation Age Mean-Absolute-Error')
    plt.show()

    return model, train_losses, val_losses, val_gender_accuracies, val_age_maes


# ==================== 辅助函数 ====================

def visualize_model_summary(model, input_size=(3, 224, 224)):
    """可视化模型架构摘要"""
    summary(model, input_size=input_size, device=device)


def test_age_gender_prediction(model, image_path, dataset):
    """测试年龄性别预测"""
    im = cv2.imread(image_path)
    im = dataset.preprocess_image(im).to(device)
    gender, age = model(im)
    pred_gender = gender.to('cpu').detach().numpy()
    pred_age = age.to('cpu').detach().numpy()
    im = cv2.imread(image_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    plt.imshow(im)
    print('Predicted gender:', np.where(pred_gender[0][0] < 0.5, 'Male', 'Female'),
          '; Predicted age:', int(pred_age[0][0] * 80))


# ==================== 主函数 ====================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Image Classification and Facial Analysis')
    parser.add_argument('--task', type=str, required=True,
                        choices=['cats_dogs', 'facial_keypoints', 'age_gender'],
                        help='Task to perform: cats_dogs, facial_keypoints, or age_gender')
    parser.add_argument('--model', type=str, default='vgg16',
                        help='Model architecture (for cats_dogs task): vgg16 or resnet18')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('--test_image', type=str, default=None,
                        help='Path to test image for age_gender prediction')

    args = parser.parse_args()

    if args.task == 'cats_dogs':
        print(f"Training {args.model} for cats vs dogs classification...")
        model, train_loss, train_acc, val_acc = train_classification_model(
            model_name=args.model, epochs=args.epochs)

    elif args.task == 'facial_keypoints':
        print("Training facial keypoints detection model...")
        model, train_loss, test_loss = train_facial_keypoints_model(epochs=args.epochs)

    elif args.task == 'age_gender':
        print("Training age and gender prediction model...")
        model, train_loss, val_loss, val_gender_acc, val_age_mae = train_age_gender_model(epochs=args.epochs)

        if args.test_image:
            # 下载测试图像
            # !wget
            # https: // www.dropbox.com / s / 6
            # kzr8l68e9kpjkf / 5_9.J
            # PG
            # 创建数据集实例以使用其预处理方法
            trn_df = pd.read_csv('fairface-label-train.csv')
            dataset = GenderAgeClass(trn_df)
            test_age_gender_prediction(model, args.test_image, dataset)

    else:
        print("Invalid task specified. Please choose from: cats_dogs, facial_keypoints, age_gender")