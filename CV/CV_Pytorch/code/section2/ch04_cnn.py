# -*- coding: utf-8 -*-
# @project: LargeDemo
# @filename: ch04_cnn.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/8/26 18:04
# https://chat.deepseek.com/a/chat/s/3aaec5c2-df17-4da1-9cf8-d36006ef6ff5

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torchvision import datasets
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imgaug import augmenters as iaa
import cv2
from glob import glob
from random import shuffle, seed
import matplotlib.ticker as mticker
from torchsummary import summary

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ==================== FMNIST Dataset and Model ====================

class FMNISTDataset(Dataset):
    def __init__(self, x, y, aug=None, view_flat=False):
        x = x.float() / 255
        if view_flat:
            x = x.view(-1, 28 * 28)
        else:
            x = x.view(-1, 1, 28, 28)
        self.x, self.y = x, y
        self.aug = aug

    def __getitem__(self, ix):
        x, y = self.x[ix], self.y[ix]
        return x.to(device), y.to(device)

    def __len__(self):
        return len(self.x)


def get_fmnist_model(model_type='cnn'):
    if model_type == 'mlp':
        model = nn.Sequential(
            nn.Linear(28 * 28, 1000),
            nn.ReLU(),
            nn.Linear(1000, 10)
        ).to(device)
    elif model_type == 'cnn':
        model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3200, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        ).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-3)
    return model, loss_fn, optimizer


def get_fmnist_data(aug=None, view_flat=False):
    data_folder = './data'
    fmnist_train = datasets.FashionMNIST(data_folder, download=True, train=True)
    fmnist_val = datasets.FashionMNIST(data_folder, download=True, train=False)

    train_dataset = FMNISTDataset(fmnist_train.data, fmnist_train.targets, aug, view_flat)
    val_dataset = FMNISTDataset(fmnist_val.data, fmnist_val.targets, view_flat=view_flat)

    train_dl = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=True)

    return train_dl, val_dl, fmnist_train.classes


# ==================== Training Functions ====================

def train_batch(x, y, model, optimizer, loss_fn):
    model.train()
    prediction = model(x)
    batch_loss = loss_fn(prediction, y)
    batch_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return batch_loss.item()


@torch.no_grad()
def accuracy(x, y, model):
    model.eval()
    prediction = model(x)
    max_values, argmaxes = prediction.max(-1)
    is_correct = argmaxes == y
    return is_correct.cpu().numpy().tolist()


@torch.no_grad()
def val_loss(x, y, model, loss_fn):
    model.eval()
    prediction = model(x)
    val_loss = loss_fn(prediction, y)
    return val_loss.item()


def train_model(model, loss_fn, optimizer, train_dl, val_dl, epochs=5):
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []

    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')

        # Training
        model.train()
        train_epoch_losses, train_epoch_accuracies = [], []
        for batch in train_dl:
            x, y = batch
            batch_loss = train_batch(x, y, model, optimizer, loss_fn)
            train_epoch_losses.append(batch_loss)

            is_correct = accuracy(x, y, model)
            train_epoch_accuracies.extend(is_correct)

        train_epoch_loss = np.mean(train_epoch_losses)
        train_epoch_accuracy = np.mean(train_epoch_accuracies)

        # Validation
        model.eval()
        val_epoch_accuracies = []
        for batch in val_dl:
            x, y = batch
            val_is_correct = accuracy(x, y, model)
            val_epoch_accuracies.extend(val_is_correct)
            validation_loss = val_loss(x, y, model, loss_fn)

        val_epoch_accuracy = np.mean(val_epoch_accuracies)

        # Store metrics
        train_losses.append(train_epoch_loss)
        train_accuracies.append(train_epoch_accuracy)
        val_losses.append(validation_loss)
        val_accuracies.append(val_epoch_accuracy)

        print(f'Train Loss: {train_epoch_loss:.4f}, Train Acc: {train_epoch_accuracy:.4f}')
        print(f'Val Loss: {validation_loss:.4f}, Val Acc: {val_epoch_accuracy:.4f}')

    return train_losses, train_accuracies, val_losses, val_accuracies


def plot_training_results(train_losses, train_accuracies, val_losses, val_accuracies, title_suffix=''):
    epochs = np.arange(len(train_losses)) + 1

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot losses
    ax1.plot(epochs, train_losses, 'bo', label='Training loss')
    ax1.plot(epochs, val_losses, 'r', label='Validation loss')
    ax1.xaxis.set_major_locator(mticker.MultipleLocator(1))
    ax1.set_title(f'Training and validation loss {title_suffix}')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(False)

    # Plot accuracies
    ax2.plot(epochs, train_accuracies, 'bo', label='Training accuracy')
    ax2.plot(epochs, val_accuracies, 'r', label='Validation accuracy')
    ax2.xaxis.set_major_locator(mticker.MultipleLocator(1))
    ax2.set_title(f'Training and validation accuracy {title_suffix}')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_yticklabels(['{:.0f}%'.format(x * 100) for x in ax2.get_yticks()])
    ax2.legend()
    ax2.grid(False)

    plt.tight_layout()
    plt.show()


# ==================== Translation Analysis ====================

def analyze_translation_effect(model, images, targets, classes, sample_idx=24300):
    """Analyze how translation affects model predictions"""
    img = images[sample_idx] / 255.
    original_img = img.view(28, 28)
    original_label = targets[sample_idx]

    print(f"Original image label: {classes[original_label]}")
    plt.imshow(original_img, cmap='gray')
    plt.title(f'Original: {classes[original_label]}')
    plt.show()

    preds = []
    for px in range(-5, 6):
        img_translated = np.roll(original_img, px, axis=1)

        if len(model._modules) > 0 and isinstance(list(model.children())[0], nn.Conv2d):
            # CNN model expects 4D input
            img_tensor = torch.Tensor(img_translated).view(-1, 1, 28, 28).to(device)
        else:
            # MLP model expects flattened input
            img_tensor = torch.Tensor(img_translated).view(28 * 28).to(device)

        with torch.no_grad():
            output = model(img_tensor)
            if output.dim() > 1:
                output = output[0]  # Handle batch dimension
            probs = torch.softmax(output, dim=-1).cpu().numpy()

        preds.append(probs)

        plt.imshow(img_translated, cmap='gray')
        predicted_class = classes[np.argmax(probs)]
        plt.title(f'Translation {px}px: {predicted_class}')
        plt.show()

    # Create heatmap of predictions
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    plt.title('Probability of each class for various translations')
    sns.heatmap(np.array(preds), annot=True, ax=ax, fmt='.2f',
                xticklabels=classes,
                yticklabels=[f'{i} pixels' for i in range(-5, 6)],
                cmap='gray')
    plt.show()


# ==================== CNN Working Details ====================

def demonstrate_cnn_working():
    """Demonstrate how CNN works with a simple example"""
    X_train = torch.tensor([[[[1, 2, 3, 4], [2, 3, 4, 5], [5, 6, 7, 8], [1, 3, 4, 5]]],
                            [[[-1, 2, 3, -4], [2, -3, 4, 5], [-5, 6, -7, 8], [-1, -3, -4, -5]]]]).to(device).float()
    X_train /= 8
    y_train = torch.tensor([0, 1]).to(device).float()

    model = nn.Sequential(
        nn.Conv2d(1, 1, kernel_size=3),
        nn.MaxPool2d(2),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(1, 1),
        nn.Sigmoid(),
    ).to(device)

    loss_fn = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=1e-2)

    # Train the model
    train_dl = DataLoader(list(zip(X_train, y_train)), batch_size=2)
    for epoch in range(2000):
        for x, y in train_dl:
            train_batch(x, y, model, optimizer, loss_fn)

    # Demonstrate manual computation
    cnn_layer = list(model.children())[0]
    linear_layer = list(model.children())[4]

    cnn_w, cnn_b = cnn_layer.weight.data, cnn_layer.bias.data
    lin_w, lin_b = linear_layer.weight.data, linear_layer.bias.data

    # Manual convolution
    h_im, w_im = X_train.shape[2:]
    h_conv, w_conv = cnn_w.shape[2:]
    sumprod = torch.zeros((h_im - h_conv + 1, w_im - w_conv + 1))

    for i in range(h_im - h_conv + 1):
        for j in range(w_im - w_conv + 1):
            img_subset = X_train[0, 0, i:(i + 3), j:(j + 3)]
            model_filter = cnn_w.reshape(3, 3)
            val = torch.sum(img_subset * model_filter) + cnn_b
            sumprod[i, j] = val

    # ReLU and pooling
    sumprod.clamp_min_(0)
    pooling_output = torch.max(sumprod)

    # Linear layer
    intermediate_output = pooling_output * lin_w + lin_b
    final_output = torch.sigmoid(intermediate_output)

    print("Manual computation result:", final_output.item())
    print("Model output:", model(X_train[:1]).item())


# ==================== Data Augmentation ====================

class AugmentedFMNISTDataset(FMNISTDataset):
    def __init__(self, x, y, aug=None):
        super().__init__(x, y, aug=aug, view_flat=False)

    def collate_fn(self, batch):
        ims, classes = list(zip(*batch))
        if self.aug:
            ims_np = [im.cpu().numpy().transpose(1, 2, 0) for im in ims]
            ims_aug = self.aug.augment_images(images=ims_np)
            ims = [torch.tensor(im.transpose(2, 0, 1)) for im in ims_aug]

        ims = torch.stack(ims).to(device).float() / 255.
        classes = torch.tensor(classes).to(device)
        return ims, classes


def get_augmented_data():
    aug = iaa.Sequential([
        iaa.Affine(translate_px={'x': (-10, 10)}, mode='constant'),
    ])

    data_folder = './data'
    fmnist_train = datasets.FashionMNIST(data_folder, download=True, train=True)
    fmnist_val = datasets.FashionMNIST(data_folder, download=True, train=False)

    train_dataset = AugmentedFMNISTDataset(fmnist_train.data, fmnist_train.targets, aug=aug)
    val_dataset = FMNISTDataset(fmnist_val.data, fmnist_val.targets, view_flat=False)

    train_dl = DataLoader(train_dataset, batch_size=64,
                          collate_fn=train_dataset.collate_fn, shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=True)

    return train_dl, val_dl, fmnist_train.classes


# ==================== Filter Visualization ====================

class XODataset(Dataset):
    def __init__(self, folder):
        self.files = glob(folder)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, ix):
        f = self.files[ix]
        im = cv2.imread(f)[:, :, 0]  # Grayscale
        im = cv2.resize(im, (28, 28))
        cl = f.split('/')[-1].split('@')[0] == 'x'
        return torch.tensor(1 - im / 255).unsqueeze(0).to(device).float(), torch.tensor([cl]).float().to(device)


def visualize_filters(model, dataset, sample_idx=2):
    """Visualize filters of a trained model"""
    im, c = dataset[sample_idx]
    plt.imshow(im[0].cpu(), cmap='gray')
    plt.title(f'Class: {"X" if c.item() else "O"}')
    plt.show()

    # First layer filters
    first_layer = nn.Sequential(*list(model.children())[:1])
    intermediate_output = first_layer(im.unsqueeze(0))[0].detach()

    n = 8
    fig, ax = plt.subplots(n, n, figsize=(10, 10))
    for i in range(n * n):
        if i < intermediate_output.shape[0]:
            ax.flat[i].imshow(intermediate_output[i].cpu())
            ax.flat[i].set_title(f'Filter: {i}')
        ax.flat[i].axis('off')
    plt.tight_layout()
    plt.show()

    # Second layer filters
    second_layer = nn.Sequential(*list(model.children())[:4])
    second_output = second_layer(im.unsqueeze(0))[0].detach()

    n = 11
    fig, ax = plt.subplots(n, n, figsize=(10, 10))
    for i in range(n * n):
        if i < second_output.shape[0]:
            ax.flat[i].imshow(second_output[i].cpu())
            ax.flat[i].set_title(str(i))
        ax.flat[i].axis('off')
    plt.tight_layout()
    plt.show()


# ==================== Cats vs Dogs ====================

class CatsDogsDataset(Dataset):
    def __init__(self, folder, max_samples=None):
        cats = glob(folder + '/cats/*.jpg')
        dogs = glob(folder + '/dogs/*.jpg')

        if max_samples:
            cats = cats[:max_samples // 2]
            dogs = dogs[:max_samples // 2]

        self.fpaths = cats + dogs
        seed(10)
        shuffle(self.fpaths)
        self.targets = [fpath.split('/')[-1].startswith('dog') for fpath in self.fpaths]

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, ix):
        f = self.fpaths[ix]
        target = self.targets[ix]
        im = cv2.imread(f)[:, :, ::-1]  # BGR to RGB
        im = cv2.resize(im, (224, 224))
        return torch.tensor(im / 255).permute(2, 0, 1).to(device).float(), torch.tensor([target]).float().to(device)


def get_cats_dogs_model():
    def conv_layer(ni, no, kernel_size, stride=1):
        return nn.Sequential(
            nn.Conv2d(ni, no, kernel_size, stride),
            nn.ReLU(),
            nn.BatchNorm2d(no),
            nn.MaxPool2d(2)
        )

    model = nn.Sequential(
        conv_layer(3, 64, 3),
        conv_layer(64, 512, 3),
        conv_layer(512, 512, 3),
        conv_layer(512, 512, 3),
        conv_layer(512, 512, 3),
        conv_layer(512, 512, 3),
        nn.Flatten(),
        nn.Linear(512, 1),
        nn.Sigmoid(),
    ).to(device)

    loss_fn = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=1e-3)
    return model, loss_fn, optimizer


def get_cats_dogs_data(train_dir, test_dir, max_samples=None):
    train_dataset = CatsDogsDataset(train_dir, max_samples)
    test_dataset = CatsDogsDataset(test_dir)

    train_dl = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
    test_dl = DataLoader(test_dataset, batch_size=32, shuffle=True, drop_last=True)

    return train_dl, test_dl


def run_cats_dogs_experiment(data_sizes=[1000, 2000, 4000, 8000]):
    train_data_dir = '/content/training_set/training_set'
    test_data_dir = '/content/test_set/test_set'

    results = {}

    for size in data_sizes:
        print(f"Training with {size} samples...")
        train_dl, test_dl = get_cats_dogs_data(train_data_dir, test_data_dir, size)
        model, loss_fn, optimizer = get_cats_dogs_model()

        _, _, _, val_accuracies = train_model(model, loss_fn, optimizer, train_dl, test_dl, epochs=5)
        results[size] = val_accuracies

    # Plot results
    epochs = np.arange(5) + 1
    plt.figure(figsize=(10, 6))

    for size, accuracies in results.items():
        plt.plot(epochs, accuracies, label=f'{size} samples')

    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.title('Validation accuracy with different dataset sizes')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.gca().set_yticklabels(['{:.0f}%'.format(x * 100) for x in plt.gca().get_yticks()])
    plt.legend()
    plt.grid(False)
    plt.show()

    return results


# ==================== Main Execution ====================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run different computer vision experiments')
    parser.add_argument('--experiment', type=str, default='fmnist_cnn',
                        choices=['fmnist_mlp', 'fmnist_cnn', 'cnn_demo',
                                 'translation_analysis', 'augmentation',
                                 'filter_visualization', 'cats_dogs'],
                        help='Which experiment to run')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs')

    args = parser.parse_args()

    if args.experiment == 'fmnist_mlp':
        # FMNIST with MLP
        train_dl, val_dl, classes = get_fmnist_data(view_flat=True)
        model, loss_fn, optimizer = get_fmnist_model('mlp')
        train_losses, train_accuracies, val_losses, val_accuracies = train_model(
            model, loss_fn, optimizer, train_dl, val_dl, args.epochs)
        plot_training_results(train_losses, train_accuracies, val_losses, val_accuracies, 'with MLP')

    elif args.experiment == 'fmnist_cnn':
        # FMNIST with CNN
        train_dl, val_dl, classes = get_fmnist_data()
        model, loss_fn, optimizer = get_fmnist_model('cnn')
        summary(model, (1, 28, 28))
        train_losses, train_accuracies, val_losses, val_accuracies = train_model(
            model, loss_fn, optimizer, train_dl, val_dl, args.epochs)
        plot_training_results(train_losses, train_accuracies, val_losses, val_accuracies, 'with CNN')

    elif args.experiment == 'cnn_demo':
        # CNN working details demonstration
        demonstrate_cnn_working()

    elif args.experiment == 'translation_analysis':
        # Translation effect analysis
        train_dl, val_dl, classes = get_fmnist_data()
        model, loss_fn, optimizer = get_fmnist_model('cnn')
        # Load or train model here...
        data_folder = './data'
        fmnist = datasets.FashionMNIST(data_folder, download=True, train=True)
        analyze_translation_effect(model, fmnist.data, fmnist.targets, fmnist.classes)

    elif args.experiment == 'augmentation':
        # Data augmentation experiment
        train_dl, val_dl, classes = get_augmented_data()
        model, loss_fn, optimizer = get_fmnist_model('cnn')
        train_losses, train_accuracies, val_losses, val_accuracies = train_model(
            model, loss_fn, optimizer, train_dl, val_dl, args.epochs)
        plot_training_results(train_losses, train_accuracies, val_losses, val_accuracies, 'with augmentation')

    elif args.experiment == 'filter_visualization':
        # Filter visualization (requires XO dataset)
        print("This experiment requires the XO dataset")
        # dataset = XODataset('/content/all/*')
        # model, loss_fn, optimizer = get_model()  # Would need appropriate model
        # visualize_filters(model, dataset)

    elif args.experiment == 'cats_dogs':
        # Cats vs Dogs experiment
        print("This experiment requires the Cats vs Dogs dataset")
        # results = run_cats_dogs_experiment()

    print("Experiment completed!")