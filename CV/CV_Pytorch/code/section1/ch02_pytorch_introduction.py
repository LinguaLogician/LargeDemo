# -*- coding: utf-8 -*-
# @project: LargeDemo
# @filename: ch02_pytorch_introduction.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/8/26 16:45

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD
import time
import os
# https://chat.deepseek.com/a/chat/s/4351ebf7-a7e3-4407-a705-3d26cc088766

def initialize_tensor():
    """Initialize tensors and demonstrate basic properties."""
    x = torch.tensor([[1, 2]])
    y = torch.tensor([[1], [2]])
    print(f"x.shape: {x.shape}")
    print(f"y.shape: {y.shape}")
    print(f"x.dtype: {x.dtype}")

    x = torch.tensor([False, 1, 2.0])
    print(f"Mixed tensor: {x}")

    zeros = torch.zeros((3, 4))
    ones = torch.ones((3, 4))
    rand_int = torch.randint(low=0, high=10, size=(3, 4))
    rand = torch.rand(3, 4)
    randn = torch.randn((3, 4))

    print(f"Zeros:\n{zeros}")
    print(f"Ones:\n{ones}")
    print(f"Random integers:\n{rand_int}")
    print(f"Random uniform:\n{rand}")
    print(f"Random normal:\n{randn}")

    np_array = np.array([[10, 20, 30], [2, 3, 4]])
    torch_tensor = torch.tensor(np_array)
    print(f"Type of np_array: {type(np_array)}, Type of torch_tensor: {type(torch_tensor)}")


def tensor_operations():
    """Demonstrate operations on tensors."""
    x = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    print(f"x * 10:\n{x * 10}")

    y = x.add(10)
    print(f"x.add(10):\n{y}")

    y = torch.tensor([2, 3, 1, 0])
    y = y.view(4, 1)
    print(f"y after view(4,1): {y.shape}")

    x = torch.randn(10, 1, 10)
    z1 = torch.squeeze(x, 1)
    z2 = x.squeeze(1)
    assert torch.all(z1 == z2)
    print(f'Squeeze: {x.shape} -> {z1.shape}')

    x = torch.randn(10, 10)
    z1 = x.unsqueeze(0)
    z2, z3, z4 = x[None], x[:, None], x[:, :, None]
    print(f"Unsqueeze shapes: {z1.shape}, {z2.shape}, {z3.shape}, {z4.shape}")

    x = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    y = torch.tensor([[2], [3], [1], [0]])
    print(f"matmul(x, y):\n{torch.matmul(x, y)}")
    print(f"x @ y:\n{x @ y}")

    x = torch.randn(10, 10, 10)
    z = torch.cat([x, x], axis=0)
    print(f'Cat axis 0: {x.shape} -> {z.shape}')
    z = torch.cat([x, x], axis=1)
    print(f'Cat axis 1: {x.shape} -> {z.shape}')

    x = torch.arange(25).reshape(5, 5)
    print(f'Max: {x.max()}')
    m, argm = x.max(dim=1)
    print(f'Max in axis 1: {m}, {argm}')

    x = torch.randn(10, 20, 30)
    z = x.permute(2, 0, 1)
    print(f'Permute dimensions: {x.shape} -> {z.shape}')


def auto_gradient():
    """Demonstrate auto gradient computation."""
    x = torch.tensor([[2., -1.], [1., 1.]], requires_grad=True)
    out = x.pow(2).sum()
    out.backward()
    print(f"Gradient of x:\n{x.grad}")

    x_np = np.array([[1, 1]])
    y_np = np.array([[0]])
    x_tensor, y_tensor = [torch.tensor(i).float() for i in [x_np, y_np]]

    W = [
        np.array([[-0.0053, 0.3793],
                  [-0.5820, -0.5204],
                  [-0.2723, 0.1896]], dtype=np.float32).T,
        np.array([-0.0140, 0.5607, -0.0628], dtype=np.float32),
        np.array([[0.1528, -0.1745, -0.1135]], dtype=np.float32).T,
        np.array([-0.5516], dtype=np.float32)
    ]
    W = [torch.tensor(i, requires_grad=True) for i in W]

    def feed_forward(inputs, outputs, weights):
        pre_hidden = torch.matmul(inputs, weights[0]) + weights[1]
        hidden = 1 / (1 + torch.exp(-pre_hidden))
        out = torch.matmul(hidden, weights[2]) + weights[3]
        mean_squared_error = torch.mean(torch.square(out - outputs))
        return mean_squared_error

    loss = feed_forward(x_tensor, y_tensor, W)
    print(f"Initial loss: {loss.item()}")

    loss.backward()
    gradients = [w.grad for w in W]
    print("Gradients of W:")
    for i, grad in enumerate(gradients):
        print(f"W[{i}]: {grad}")

    updated_W = [w - w.grad for w in W]
    print("Updated weights:")
    for i, w in enumerate(updated_W):
        print(f"W[{i}]: {w}")


def speed_comparison():
    """Compare computation speed between CPU and GPU."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    assert device == 'cuda', "This exercise assumes the notebook is on a GPU machine"

    x = torch.rand(1, 6400)
    y = torch.rand(6400, 5000)
    x, y = x.to(device), y.to(device)

    start = time.time()
    z = x @ y
    gpu_time = time.time() - start
    print(f"GPU time: {gpu_time:.6f} seconds")

    x, y = x.cpu(), y.cpu()
    start = time.time()
    z = x @ y
    cpu_time = time.time() - start
    print(f"CPU time: {cpu_time:.6f} seconds")

    x_np = np.random.random((1, 6400))
    y_np = np.random.random((6400, 5000))
    start = time.time()
    z_np = np.matmul(x_np, y_np)
    numpy_time = time.time() - start
    print(f"NumPy time: {numpy_time:.6f} seconds")


class MyDataset(Dataset):
    def __init__(self, x, y, device='cpu'):
        self.x = torch.tensor(x).float().to(device)
        self.y = torch.tensor(y).float().to(device)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class MyNeuralNet(nn.Module):
    def __init__(self, input_size=2, hidden_size=8, output_size=1):
        super().__init__()
        self.input_to_hidden_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layer_activation = nn.ReLU()
        self.hidden_to_output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.input_to_hidden_layer(x)
        x = self.hidden_layer_activation(x)
        x = self.hidden_to_output_layer(x)
        return x


def build_and_train_model(batch_size=None, return_intermediate=False):
    """Build and train a neural network on toy data."""
    x = [[1, 2], [3, 4], [5, 6], [7, 8]]
    y = [[3], [7], [11], [15]]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if batch_size is None:
        X = torch.tensor(x).float().to(device)
        Y = torch.tensor(y).float().to(device)

        model = MyNeuralNet().to(device)
        loss_func = nn.MSELoss()
        opt = SGD(model.parameters(), lr=0.001)

        loss_history = []
        for _ in range(50):
            opt.zero_grad()
            loss_value = loss_func(model(X), Y)
            loss_value.backward()
            opt.step()
            loss_history.append(loss_value.item())

        plt.plot(loss_history)
        plt.title('Loss variation over increasing epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss value')
        plt.show()

        val_x = torch.tensor([[10, 11]]).float().to(device)
        print(f"Prediction for [10, 11]: {model(val_x).item()}")

    else:
        ds = MyDataset(x, y, device)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

        model = MyNeuralNet().to(device)
        loss_func = nn.MSELoss()
        opt = SGD(model.parameters(), lr=0.001)

        loss_history = []
        start_time = time.time()
        for _ in range(50):
            for batch_x, batch_y in dl:
                opt.zero_grad()
                loss_value = loss_func(model(batch_x), batch_y)
                loss_value.backward()
                opt.step()
                loss_history.append(loss_value.item())

        training_time = time.time() - start_time
        print(f"Training time with batch size {batch_size}: {training_time:.4f} seconds")

        val_x = torch.tensor([[10, 11]]).float().to(device)
        print(f"Prediction for [10, 11]: {model(val_x).item()}")

    if return_intermediate:
        # Demonstrate accessing intermediate layers
        X = torch.tensor(x).float().to(device)
        intermediate_output = model.input_to_hidden_layer(X)
        print(f"Intermediate layer output:\n{intermediate_output}")
        return model, intermediate_output

    return model


def custom_loss_function():
    """Implement and test a custom loss function."""

    def my_mean_squared_error(_y, y):
        loss = (_y - y) ** 2
        loss = loss.mean()
        return loss

    x = [[1, 2], [3, 4], [5, 6], [7, 8]]
    y = [[3], [7], [11], [15]]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    X = torch.tensor(x).float().to(device)
    Y = torch.tensor(y).float().to(device)

    model = MyNeuralNet().to(device)

    # Standard MSE loss
    standard_loss = nn.MSELoss()(model(X), Y)
    print(f"Standard MSE loss: {standard_loss.item()}")

    # Custom MSE loss
    custom_loss = my_mean_squared_error(model(X), Y)
    print(f"Custom MSE loss: {custom_loss.item()}")


def sequential_model():
    """Build and train a model using nn.Sequential."""
    x = [[1, 2], [3, 4], [5, 6], [7, 8]]
    y = [[3], [7], [11], [15]]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ds = MyDataset(x, y, device)
    dl = DataLoader(ds, batch_size=2, shuffle=True)

    model = nn.Sequential(
        nn.Linear(2, 8),
        nn.ReLU(),
        nn.Linear(8, 1)
    ).to(device)

    try:
        from torchsummary import summary
        summary(model, (1, 2))
    except ImportError:
        print("torch_summary not installed, skipping model summary")

    loss_func = nn.MSELoss()
    opt = SGD(model.parameters(), lr=0.001)

    loss_history = []
    start_time = time.time()
    for _ in range(50):
        for batch_x, batch_y in dl:
            opt.zero_grad()
            loss_value = loss_func(model(batch_x), batch_y)
            loss_value.backward()
            opt.step()
            loss_history.append(loss_value.item())

    training_time = time.time() - start_time
    print(f"Training time: {training_time:.4f} seconds")

    val = [[8, 9], [10, 11], [1.5, 2.5]]
    val_tensor = torch.tensor(val).float().to(device)
    predictions = model(val_tensor)
    print(f"Predictions for {val}:")
    for i, pred in enumerate(predictions):
        print(f"  {val[i]} -> {pred.item():.4f} (expected: {sum(val[i])})")

    return model


def save_and_load_model():
    """Demonstrate saving and loading a model."""
    model = sequential_model()  # Reuse the sequential model function

    # Save model
    save_path = 'mymodel.pth'
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    # Check file size
    if os.path.exists(save_path):
        file_size = os.path.getsize(save_path) / 1024  # Size in KB
        print(f"Model file size: {file_size:.2f} KB")

    # Create a new model instance
    loaded_model = nn.Sequential(
        nn.Linear(2, 8),
        nn.ReLU(),
        nn.Linear(8, 1)
    ).to(next(model.parameters()).device)

    # Load weights
    loaded_model.load_state_dict(torch.load(save_path))
    loaded_model.eval()
    print("Model loaded successfully")

    # Test loaded model
    val = [[8, 9], [10, 11], [1.5, 2.5]]
    val_tensor = torch.tensor(val).float().to(next(model.parameters()).device)
    predictions = loaded_model(val_tensor)
    print(f"Predictions from loaded model for {val}:")
    for i, pred in enumerate(predictions):
        print(f"  {val[i]} -> {pred.item():.4f} (expected: {sum(val[i])})")

    return loaded_model


if __name__ == "__main__":
    # Uncomment the functions you want to run

    # Basic tensor operations
    print("=== Initializing Tensors ===")
    initialize_tensor()

    print("\n=== Tensor Operations ===")
    tensor_operations()

    print("\n=== Auto Gradient ===")
    auto_gradient()

    print("\n=== Speed Comparison ===")
    speed_comparison()

    print("\n=== Building and Training Model (No Batch) ===")
    build_and_train_model()

    print("\n=== Building and Training Model (With Batch) ===")
    build_and_train_model(batch_size=2)

    print("\n=== Custom Loss Function ===")
    custom_loss_function()

    print("\n=== Sequential Model ===")
    sequential_model()

    print("\n=== Save and Load Model ===")
    save_and_load_model()

    print("\n=== Access Intermediate Layers ===")
    build_and_train_model(return_intermediate=True)
