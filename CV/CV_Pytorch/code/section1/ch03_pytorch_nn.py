# -*- coding: utf-8 -*-
# @project: LargeDemo
# @filename: ch03_pytorch_nn.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/8/26 17:54

# https://chat.deepseek.com/a/chat/s/8b729f95-1975-4abb-8b19-78dde5327d43
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torch.optim import SGD, Adam
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cv2
import argparse
import sys
import os

# 设置设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ==================== 数据准备和可视化函数 ====================

def prepare_data(data_folder='~/data/FMNIST'):
    """准备FashionMNIST数据集"""
    fmnist = datasets.FashionMNIST(data_folder, download=True, train=True)
    tr_images = fmnist.data
    tr_targets = fmnist.targets

    val_fmnist = datasets.FashionMNIST(data_folder, download=True, train=False)
    val_images = val_fmnist.data
    val_targets = val_fmnist.targets

    return tr_images, tr_targets, val_images, val_targets, fmnist.classes


def inspect_grayscale_images():
    """检查灰度图像"""
    # 下载示例图像
    os.system('wget https://www.dropbox.com/s/l98leemr7r5stnm/Hemanvi.jpeg')

    # 读取和处理图像
    img = cv2.imread('Hemanvi.jpeg')
    img = img[50:250, 40:240]
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 显示原图
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.imshow(img_gray, cmap='gray')
    plt.title('Original Grayscale Image')

    # 显示缩小后的图像
    img_gray_small = cv2.resize(img_gray, (25, 25))
    plt.subplot(122)
    plt.imshow(img_gray_small, cmap='gray')
    plt.title('Resized Grayscale Image (25x25)')
    plt.tight_layout()
    plt.show()

    print("Resized image array:")
    print(img_gray_small)


def inspect_color_images():
    """检查彩色图像"""
    # 下载示例图像
    os.system('wget https://www.dropbox.com/s/l98leemr7r5stnm/Hemanvi.jpeg')

    # 读取和处理图像
    img = cv2.imread('Hemanvi.jpeg')
    img = img[50:250, 40:240, :]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 显示图像
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.title('Color Image')
    plt.axis('off')
    plt.show()

    print(f"Image shape: {img.shape}")


def visualize_dataset_samples(tr_images, tr_targets, classes):
    """可视化数据集样本"""
    unique_values = tr_targets.unique()
    print(
        f'tr_images & tr_targets:\n\tX - {tr_images.shape}\n\tY - {tr_targets.shape}\n\tY - Unique Values : {unique_values}')
    print(f'TASK:\n\t{len(unique_values)} class Classification')
    print(f'UNIQUE CLASSES:\n\t{classes}')

    # 可视化每个类别的样本
    R, C = len(tr_targets.unique()), 10
    fig, ax = plt.subplots(R, C, figsize=(10, 10))
    for label_class, plot_row in enumerate(ax):
        label_x_rows = np.where(tr_targets == label_class)[0]
        for plot_cell in plot_row:
            plot_cell.grid(False)
            plot_cell.axis('off')
            ix = np.random.choice(label_x_rows)
            x, y = tr_images[ix], tr_targets[ix]
            plot_cell.imshow(x, cmap='gray')
    plt.tight_layout()
    plt.show()


# ==================== 数据集类定义 ====================

class FMNISTDataset(Dataset):
    """FashionMNIST数据集类"""

    def __init__(self, x, y, scaled=True, view_flat=True):
        x = x.float()
        if scaled:
            x = x / 255
        if view_flat:
            x = x.view(-1, 28 * 28)
        self.x, self.y = x, y

    def __getitem__(self, ix):
        x, y = self.x[ix], self.y[ix]
        return x.to(device), y.to(device)

    def __len__(self):
        return len(self.x)


# ==================== 模型定义函数 ====================

def get_simple_model(input_size=28 * 28, hidden_size=1000, output_size=10, dropout_rate=0.0):
    """获取简单神经网络模型"""
    layers = []
    if dropout_rate > 0:
        layers.append(nn.Dropout(dropout_rate))

    layers.extend([
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
    ])

    if dropout_rate > 0:
        layers.append(nn.Dropout(dropout_rate))

    layers.append(nn.Linear(hidden_size, output_size))

    return nn.Sequential(*layers).to(device)


def get_multi_layer_model(hidden_layers=1, hidden_size=1000, input_size=28 * 28, output_size=10):
    """获取多层神经网络模型"""
    layers = [nn.Linear(input_size, hidden_size), nn.ReLU()]

    for _ in range(hidden_layers - 1):
        layers.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])

    layers.append(nn.Linear(hidden_size, output_size))

    return nn.Sequential(*layers).to(device)


def get_batchnorm_model():
    """获取带BatchNorm的模型"""

    class NeuralNet(nn.Module):
        def __init__(self, use_batchnorm=False):
            super().__init__()
            self.input_to_hidden_layer = nn.Linear(784, 1000)
            self.use_batchnorm = use_batchnorm
            if use_batchnorm:
                self.batch_norm = nn.BatchNorm1d(1000)
            self.hidden_layer_activation = nn.ReLU()
            self.hidden_to_output_layer = nn.Linear(1000, 10)

        def forward(self, x):
            x = self.input_to_hidden_layer(x)
            if self.use_batchnorm:
                x = self.batch_norm(x)
            x1 = self.hidden_layer_activation(x)
            x2 = self.hidden_to_output_layer(x1)
            return x2, x1

    return NeuralNet


# ==================== 训练和评估函数 ====================

def train_batch(x, y, model, optimizer, loss_fn, regularization=None, reg_lambda=0.0):
    """训练一个批次"""
    model.train()
    prediction = model(x)

    # 计算损失
    if regularization == 'l1':
        l1_regularization = 0
        for param in model.parameters():
            l1_regularization += torch.norm(param, 1)
        batch_loss = loss_fn(prediction, y) + reg_lambda * l1_regularization
    elif regularization == 'l2':
        l2_regularization = 0
        for param in model.parameters():
            l2_regularization += torch.norm(param, 2)
        batch_loss = loss_fn(prediction, y) + reg_lambda * l2_regularization
    else:
        batch_loss = loss_fn(prediction, y)

    # 反向传播和优化
    batch_loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return batch_loss.item()


@torch.no_grad()
def accuracy(x, y, model):
    """计算准确率"""
    model.eval()
    prediction = model(x)
    max_values, argmaxes = prediction.max(-1)
    is_correct = argmaxes == y
    return is_correct.cpu().numpy().tolist()


@torch.no_grad()
def val_loss(x, y, model, loss_fn):
    """计算验证损失"""
    model.eval()
    prediction = model(x)
    return loss_fn(prediction, y).item()


def get_data_loader(tr_images, tr_targets, val_images, val_targets, batch_size=32, scaled=True, view_flat=True):
    """获取数据加载器"""
    train = FMNISTDataset(tr_images, tr_targets, scaled=scaled, view_flat=view_flat)
    trn_dl = DataLoader(train, batch_size=batch_size, shuffle=True)

    val = FMNISTDataset(val_images, val_targets, scaled=scaled, view_flat=view_flat)
    val_dl = DataLoader(val, batch_size=len(val_images), shuffle=False)

    return trn_dl, val_dl


def train_model(trn_dl, val_dl, model, loss_fn, optimizer, epochs=5,
                scheduler=None, regularization=None, reg_lambda=0.0):
    """训练模型"""
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []

    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')

        # 训练阶段
        train_epoch_losses, train_epoch_accuracies = [], []
        for ix, batch in enumerate(trn_dl):
            x, y = batch
            batch_loss = train_batch(x, y, model, optimizer, loss_fn, regularization, reg_lambda)
            train_epoch_losses.append(batch_loss)

        train_epoch_loss = np.array(train_epoch_losses).mean()

        # 计算训练准确率
        for ix, batch in enumerate(trn_dl):
            x, y = batch
            is_correct = accuracy(x, y, model)
            train_epoch_accuracies.extend(is_correct)

        train_epoch_accuracy = np.mean(train_epoch_accuracies)

        # 验证阶段
        for ix, batch in enumerate(val_dl):
            x, y = batch
            val_is_correct = accuracy(x, y, model)
            validation_loss = val_loss(x, y, model, loss_fn)

        val_epoch_accuracy = np.mean(val_is_correct)

        # 学习率调度
        if scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(validation_loss)
            else:
                scheduler.step()

        # 记录指标
        train_losses.append(train_epoch_loss)
        train_accuracies.append(train_epoch_accuracy)
        val_losses.append(validation_loss)
        val_accuracies.append(val_epoch_accuracy)

        print(f'Train Loss: {train_epoch_loss:.4f}, Train Acc: {train_epoch_accuracy:.4f}, '
              f'Val Loss: {validation_loss:.4f}, Val Acc: {val_epoch_accuracy:.4f}')

    return train_losses, train_accuracies, val_losses, val_accuracies


def plot_results(epochs, train_losses, val_losses, train_accuracies, val_accuracies, title_suffix=""):
    """绘制训练结果"""
    plt.figure(figsize=(12, 8))

    plt.subplot(211)
    plt.plot(epochs, train_losses, 'bo', label='Training loss')
    plt.plot(epochs, val_losses, 'r', label='Validation loss')
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.title(f'Training and validation loss {title_suffix}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid('off')

    plt.subplot(212)
    plt.plot(epochs, train_accuracies, 'bo', label='Training accuracy')
    plt.plot(epochs, val_accuracies, 'r', label='Validation accuracy')
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.title(f'Training and validation accuracy {title_suffix}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.gca().set_yticklabels(['{:.0f}%'.format(x * 100) for x in plt.gca().get_yticks()])
    plt.legend()
    plt.grid('off')

    plt.tight_layout()
    plt.show()


def plot_weight_distribution(model, title_suffix=""):
    """绘制权重分布"""
    for ix, par in enumerate(model.parameters()):
        plt.figure(figsize=(8, 4))
        plt.hist(par.cpu().detach().numpy().flatten(), bins=50)
        plt.xlim(-2, 2)

        if ix == 0:
            plt.title(f'Distribution of weights connecting input to hidden layer {title_suffix}')
        elif ix == 1:
            plt.title(f'Distribution of biases of hidden layer {title_suffix}')
        elif ix == 2:
            plt.title(f'Distribution of weights connecting hidden to output layer {title_suffix}')
        elif ix == 3:
            plt.title(f'Distribution of biases of output layer {title_suffix}')

        plt.xlabel('Weight Value')
        plt.ylabel('Frequency')
        plt.show()


# ==================== 实验函数 ====================

def experiment_basic_nn(data_folder='~/data/FMNIST', epochs=5, batch_size=32, scaled=True):
    """基础神经网络实验"""
    print("Running basic neural network experiment...")

    # 准备数据
    tr_images, tr_targets, val_images, val_targets, classes = prepare_data(data_folder)

    # 获取数据加载器
    trn_dl, val_dl = get_data_loader(tr_images, tr_targets, val_images, val_targets,
                                     batch_size=batch_size, scaled=scaled)

    # 获取模型和优化器
    model = get_simple_model()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=1e-2)

    # 训练模型
    train_losses, train_accuracies, val_losses, val_accuracies = train_model(
        trn_dl, val_dl, model, loss_fn, optimizer, epochs=epochs)

    # 绘制结果
    epoch_range = np.arange(epochs) + 1
    plot_results(epoch_range, train_losses, val_losses, train_accuracies, val_accuracies,
                 "with basic neural network")

    return model, train_losses, train_accuracies, val_losses, val_accuracies


def experiment_varying_batch_size(data_folder='~/data/FMNIST', epochs=5):
    """不同批次大小实验"""
    print("Running varying batch size experiment...")

    # 准备数据
    tr_images, tr_targets, val_images, val_targets, classes = prepare_data(data_folder)

    # 测试不同批次大小
    batch_sizes = [32, 10000]

    for batch_size in batch_sizes:
        print(f"\nTraining with batch size: {batch_size}")

        # 获取数据加载器
        trn_dl, val_dl = get_data_loader(tr_images, tr_targets, val_images, val_targets,
                                         batch_size=batch_size, scaled=True)

        # 获取模型和优化器
        model = get_simple_model()
        loss_fn = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=1e-2)

        # 训练模型
        train_losses, train_accuracies, val_losses, val_accuracies = train_model(
            trn_dl, val_dl, model, loss_fn, optimizer, epochs=epochs)

        # 绘制结果
        epoch_range = np.arange(epochs) + 1
        plot_results(epoch_range, train_losses, val_losses, train_accuracies, val_accuracies,
                     f"when batch size is {batch_size}")


def experiment_varying_optimizer(data_folder='~/data/FMNIST', epochs=10):
    """不同优化器实验"""
    print("Running varying optimizer experiment...")

    # 准备数据
    tr_images, tr_targets, val_images, val_targets, classes = prepare_data(data_folder)

    # 测试不同优化器
    optimizers = [
        ('SGD', lambda params: SGD(params, lr=1e-2)),
        ('Adam', lambda params: Adam(params, lr=1e-2))
    ]

    for opt_name, opt_func in optimizers:
        print(f"\nTraining with {opt_name} optimizer...")

        # 获取数据加载器
        trn_dl, val_dl = get_data_loader(tr_images, tr_targets, val_images, val_targets,
                                         batch_size=32, scaled=True)

        # 获取模型和优化器
        model = get_simple_model()
        loss_fn = nn.CrossEntropyLoss()
        optimizer = opt_func(model.parameters())

        # 训练模型
        train_losses, train_accuracies, val_losses, val_accuracies = train_model(
            trn_dl, val_dl, model, loss_fn, optimizer, epochs=epochs)

        # 绘制结果
        epoch_range = np.arange(epochs) + 1
        plot_results(epoch_range, train_losses, val_losses, train_accuracies, val_accuracies,
                     f"with {opt_name} optimizer")


def experiment_varying_learning_rate(data_folder='~/data/FMNIST', epochs=5, scaled=True):
    """不同学习率实验"""
    print("Running varying learning rate experiment...")

    # 准备数据
    tr_images, tr_targets, val_images, val_targets, classes = prepare_data(data_folder)

    # 测试不同学习率
    learning_rates = [1e-1, 1e-3, 1e-5]

    for lr in learning_rates:
        print(f"\nTraining with learning rate: {lr}")

        # 获取数据加载器
        trn_dl, val_dl = get_data_loader(tr_images, tr_targets, val_images, val_targets,
                                         batch_size=32, scaled=scaled)

        # 获取模型和优化器
        model = get_simple_model()
        loss_fn = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=lr)

        # 训练模型
        train_losses, train_accuracies, val_losses, val_accuracies = train_model(
            trn_dl, val_dl, model, loss_fn, optimizer, epochs=epochs)

        # 绘制结果
        epoch_range = np.arange(epochs) + 1
        scale_status = "on scaled data" if scaled else "on non-scaled data"
        plot_results(epoch_range, train_losses, val_losses, train_accuracies, val_accuracies,
                     f"with {lr} learning rate {scale_status}")

        # 绘制权重分布
        plot_weight_distribution(model, f"with learning rate {lr}")


def experiment_learning_rate_annealing(data_folder='~/data/FMNIST', epochs=30):
    """学习率衰减实验"""
    print("Running learning rate annealing experiment...")

    # 准备数据
    tr_images, tr_targets, val_images, val_targets, classes = prepare_data(data_folder)

    # 获取数据加载器
    trn_dl, val_dl = get_data_loader(tr_images, tr_targets, val_images, val_targets,
                                     batch_size=32, scaled=True)

    # 获取模型和优化器
    model = get_simple_model()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-3)

    # 设置学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=0, threshold=0.001,
        verbose=True, min_lr=1e-5, threshold_mode='abs')

    # 训练模型
    train_losses, train_accuracies, val_losses, val_accuracies = train_model(
        trn_dl, val_dl, model, loss_fn, optimizer, epochs=epochs, scheduler=scheduler)

    # 绘制结果
    epoch_range = np.arange(epochs) + 1
    plot_results(epoch_range, train_losses, val_losses, train_accuracies, val_accuracies,
                 "with learning rate scheduler")


def experiment_varying_depth(data_folder='~/data/FMNIST', epochs=5):
    """不同网络深度实验"""
    print("Running varying network depth experiment...")

    # 准备数据
    tr_images, tr_targets, val_images, val_targets, classes = prepare_data(data_folder)

    # 测试不同网络深度
    hidden_layers_list = [0, 1, 2]

    for hidden_layers in hidden_layers_list:
        print(f"\nTraining with {hidden_layers} hidden layers...")

        # 获取数据加载器
        trn_dl, val_dl = get_data_loader(tr_images, tr_targets, val_images, val_targets,
                                         batch_size=32, scaled=True)

        # 获取模型和优化器
        if hidden_layers == 0:
            model = nn.Sequential(nn.Linear(28 * 28, 10)).to(device)
        else:
            model = get_multi_layer_model(hidden_layers=hidden_layers)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=1e-3)

        # 训练模型
        train_losses, train_accuracies, val_losses, val_accuracies = train_model(
            trn_dl, val_dl, model, loss_fn, optimizer, epochs=epochs)

        # 绘制结果
        epoch_range = np.arange(epochs) + 1
        layer_text = "no hidden layer" if hidden_layers == 0 else f"{hidden_layers} hidden layer{'s' if hidden_layers > 1 else ''}"
        plot_results(epoch_range, train_losses, val_losses, train_accuracies, val_accuracies,
                     f"with {layer_text}")


def experiment_batch_normalization(data_folder='~/data/FMNIST', epochs=100):
    """批归一化实验"""
    print("Running batch normalization experiment...")

    # 准备数据
    tr_images, tr_targets, val_images, val_targets, classes = prepare_data(data_folder)

    # 测试有和无BatchNorm的情况
    for use_batchnorm in [False, True]:
        print(f"\nTraining with BatchNorm: {use_batchnorm}")

        # 获取数据加载器（使用非常小的输入值来突出BatchNorm的效果）
        trn_dl, val_dl = get_data_loader(tr_images, tr_targets, val_images, val_targets,
                                         batch_size=32, scaled=False, view_flat=True)

        # 手动缩放数据到非常小的值
        for dl in [trn_dl, val_dl]:
            for i, (x, y) in enumerate(dl):
                dl.dataset.x = dl.dataset.x / (255 * 10000)

        # 获取模型和优化器
        model_class = get_batchnorm_model()
        model = model_class(use_batchnorm=use_batchnorm).to(device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=1e-3)

        # 修改accuracy函数以处理返回两个值的模型
        @torch.no_grad()
        def accuracy_bn(x, y, model):
            model.eval()
            prediction, _ = model(x)
            max_values, argmaxes = prediction.max(-1)
            is_correct = argmaxes == y
            return is_correct.cpu().numpy().tolist()

        # 修改val_loss函数以处理返回两个值的模型
        @torch.no_grad()
        def val_loss_bn(x, y, model, loss_fn):
            model.eval()
            prediction, _ = model(x)
            return loss_fn(prediction, y).item()

        # 修改train_batch函数以处理返回两个值的模型
        def train_batch_bn(x, y, model, optimizer, loss_fn):
            model.train()
            prediction, _ = model(x)
            batch_loss = loss_fn(prediction, y)
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            return batch_loss.item()

        # 训练模型
        train_losses, train_accuracies, val_losses, val_accuracies = [], [], [], []

        for epoch in range(epochs):
            if epoch % 10 == 0:
                print(f'Epoch {epoch + 1}/{epochs}')

            # 训练阶段
            train_epoch_losses, train_epoch_accuracies = [], []
            for ix, batch in enumerate(trn_dl):
                x, y = batch
                batch_loss = train_batch_bn(x, y, model, optimizer, loss_fn)
                train_epoch_losses.append(batch_loss)

            train_epoch_loss = np.array(train_epoch_losses).mean()

            # 计算训练准确率
            for ix, batch in enumerate(trn_dl):
                x, y = batch
                is_correct = accuracy_bn(x, y, model)
                train_epoch_accuracies.extend(is_correct)

            train_epoch_accuracy = np.mean(train_epoch_accuracies)

            # 验证阶段
            for ix, batch in enumerate(val_dl):
                x, y = batch
                val_is_correct = accuracy_bn(x, y, model)
                validation_loss = val_loss_bn(x, y, model, loss_fn)

            val_epoch_accuracy = np.mean(val_is_correct)

            # 记录指标
            train_losses.append(train_epoch_loss)
            train_accuracies.append(train_epoch_accuracy)
            val_losses.append(validation_loss)
            val_accuracies.append(val_epoch_accuracy)

        # 绘制结果
        epoch_range = np.arange(epochs) + 1
        bn_status = "with batch normalization" if use_batchnorm else "without batch normalization"
        plot_results(epoch_range, train_losses, val_losses, train_accuracies, val_accuracies,
                     f"with very small input values {bn_status}")

        # 可视化隐藏层激活值分布
        model.eval()
        with torch.no_grad():
            sample_batch = next(iter(trn_dl))
            x, y = sample_batch
            _, hidden_activations = model(x)
            plt.hist(hidden_activations.cpu().detach().numpy().flatten(), bins=50)
            plt.title(f"Hidden layer node values' distribution {bn_status}")
            plt.xlabel('Activation Value')
            plt.ylabel('Frequency')
            plt.show()


def experiment_dropout(data_folder='~/data/FMNIST', epochs=30):
    """Dropout实验"""
    print("Running dropout experiment...")

    # 准备数据
    tr_images, tr_targets, val_images, val_targets, classes = prepare_data(data_folder)

    # 测试有和无Dropout的情况
    dropout_rates = [0.0, 0.25]

    for dropout_rate in dropout_rates:
        print(f"\nTraining with dropout rate: {dropout_rate}")

        # 获取数据加载器
        trn_dl, val_dl = get_data_loader(tr_images, tr_targets, val_images, val_targets,
                                         batch_size=32, scaled=True)

        # 获取模型和优化器
        model = get_simple_model(dropout_rate=dropout_rate)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=1e-3)

        # 训练模型
        train_losses, train_accuracies, val_losses, val_accuracies = train_model(
            trn_dl, val_dl, model, loss_fn, optimizer, epochs=epochs)

        # 绘制结果
        epoch_range = np.arange(epochs) + 1
        dropout_status = "with dropout" if dropout_rate > 0 else "without dropout"
        plot_results(epoch_range, train_losses, val_losses, train_accuracies, val_accuracies,
                     dropout_status)

        # 绘制权重分布
        plot_weight_distribution(model, dropout_status)


def experiment_regularization(data_folder='~/data/FMNIST', epochs=30):
    """正则化实验"""
    print("Running regularization experiment...")

    # 准备数据
    tr_images, tr_targets, val_images, val_targets, classes = prepare_data(data_folder)

    # 测试不同的正则化方法
    regularization_configs = [
        (None, 0.0, "no regularization"),
        ('l1', 0.0001, "L1 regularization (1e-4)"),
        ('l2', 0.01, "L2 regularization (1e-2)")
    ]

    for reg_type, reg_lambda, reg_name in regularization_configs:
        print(f"\nTraining with {reg_name}...")

        # 获取数据加载器
        trn_dl, val_dl = get_data_loader(tr_images, tr_targets, val_images, val_targets,
                                         batch_size=32, scaled=True)

        # 获取模型和优化器
        model = get_simple_model()
        loss_fn = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=1e-3)

        # 训练模型
        train_losses, train_accuracies, val_losses, val_accuracies = train_model(
            trn_dl, val_dl, model, loss_fn, optimizer, epochs=epochs,
            regularization=reg_type, reg_lambda=reg_lambda)

        # 绘制结果
        epoch_range = np.arange(epochs) + 1
        plot_results(epoch_range, train_losses, val_losses, train_accuracies, val_accuracies,
                     f"with {reg_name}")

        # 绘制权重分布
        plot_weight_distribution(model, f"with {reg_name}")


# ==================== 主函数 ====================

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Neural Network Experiments on FashionMNIST')
    parser.add_argument('--experiment', type=str, default='basic',
                        choices=['basic', 'batch_size', 'optimizer', 'lr_scaled',
                                 'lr_non_scaled', 'lr_annealing', 'depth',
                                 'batchnorm', 'dropout', 'regularization',
                                 'inspect_grayscale', 'inspect_color', 'all'],
                        help='Experiment to run')
    parser.add_argument('--data_folder', type=str, default='~/data/FMNIST',
                        help='Directory to store/load FashionMNIST data')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')

    args = parser.parse_args()

    # 运行选择的实验
    if args.experiment == 'basic' or args.experiment == 'all':
        experiment_basic_nn(args.data_folder, args.epochs, args.batch_size)

    if args.experiment == 'batch_size' or args.experiment == 'all':
        experiment_varying_batch_size(args.data_folder, args.epochs)

    if args.experiment == 'optimizer' or args.experiment == 'all':
        experiment_varying_optimizer(args.data_folder, args.epochs)

    if args.experiment == 'lr_scaled' or args.experiment == 'all':
        experiment_varying_learning_rate(args.data_folder, args.epochs, scaled=True)

    if args.experiment == 'lr_non_scaled' or args.experiment == 'all':
        experiment_varying_learning_rate(args.data_folder, args.epochs, scaled=False)

    if args.experiment == 'lr_annealing' or args.experiment == 'all':
        experiment_learning_rate_annealing(args.data_folder, args.epochs)

    if args.experiment == 'depth' or args.experiment == 'all':
        experiment_varying_depth(args.data_folder, args.epochs)

    if args.experiment == 'batchnorm' or args.experiment == 'all':
        experiment_batch_normalization(args.data_folder, min(args.epochs, 100))

    if args.experiment == 'dropout' or args.experiment == 'all':
        experiment_dropout(args.data_folder, args.epochs)

    if args.experiment == 'regularization' or args.experiment == 'all':
        experiment_regularization(args.data_folder, args.epochs)

    if args.experiment == 'inspect_grayscale' or args.experiment == 'all':
        inspect_grayscale_images()

    if args.experiment == 'inspect_color' or args.experiment == 'all':
        inspect_color_images()


if __name__ == '__main__':
    main()