# -*- coding: utf-8 -*-
# @project: LargeDemo
# @filename: ch01_neural_networks.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/8/26 16:08

import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
# https://chat.deepseek.com/a/chat/s/5cf24fc4-b063-45b9-83c1-0bf155838504

def feed_forward_nn(inputs, outputs, weights):
    """
    神经网络前向传播
    """
    pre_hidden = np.dot(inputs, weights[0]) + weights[1]
    hidden = 1 / (1 + np.exp(-pre_hidden))
    out = np.dot(hidden, weights[2]) + weights[3]
    mean_squared_error = np.mean(np.square(out - outputs))
    return mean_squared_error, out, hidden, pre_hidden


def feed_forward_linear(inputs, outputs, weights):
    """
    线性模型前向传播
    """
    out = np.dot(inputs, weights[0]) + weights[1]
    mean_squared_error = np.mean(np.square(out - outputs))
    return mean_squared_error, out


def update_weights_numerical(inputs, outputs, weights, lr, verbose=False):
    """
    数值法更新权重（有限差分法）
    """
    original_weights = deepcopy(weights)
    temp_weights = deepcopy(weights)
    updated_weights = deepcopy(weights)

    if len(weights) == 4:  # 神经网络
        original_loss, _, _, _ = feed_forward_nn(inputs, outputs, original_weights)
    else:  # 线性模型
        original_loss, _ = feed_forward_linear(inputs, outputs, original_weights)

    for i, layer in enumerate(original_weights):
        for index, weight in np.ndenumerate(layer):
            temp_weights = deepcopy(weights)
            temp_weights[i][index] += 0.0001

            if len(weights) == 4:  # 神经网络
                _loss_plus, _, _, _ = feed_forward_nn(inputs, outputs, temp_weights)
            else:  # 线性模型
                _loss_plus, _ = feed_forward_linear(inputs, outputs, temp_weights)

            grad = (_loss_plus - original_loss) / 0.0001
            updated_weights[i][index] -= grad * lr

            if verbose and i % 2 == 0:
                print(f'weight value: {np.round(original_weights[i][index], 2)}, '
                      f'original loss: {np.round(original_loss, 2)}, '
                      f'loss_plus: {np.round(_loss_plus, 2)}, '
                      f'gradient: {np.round(grad, 2)}, '
                      f'updated_weights: {np.round(updated_weights[i][index], 2)}')

    return updated_weights, original_loss


def chain_rule_update_demo():
    """
    链式法则更新权重演示
    """
    x = np.array([[1, 1]])
    y = np.array([[0]])

    W = [
        np.array([[-0.0053, 0.3793],
                  [-0.5820, -0.5204],
                  [-0.2723, 0.1896]], dtype=np.float32).T,
        np.array([-0.0140, 0.5607, -0.0628], dtype=np.float32),
        np.array([[0.1528, -0.1745, -0.1135]], dtype=np.float32).T,
        np.array([-0.5516], dtype=np.float32)
    ]

    # 前向传播
    pre_hidden = np.dot(x, W[0]) + W[1]
    hidden = 1 / (1 + np.exp(-pre_hidden))
    predicted_value = np.dot(hidden, W[2]) + W[3]

    # 使用链式法则计算梯度并更新权重
    updated_weights = deepcopy(W)

    # 更新W[0]的权重
    for i in range(W[0].shape[0]):
        for j in range(W[0].shape[1]):
            tmp = W[0][i][j] - (-2 * (0 - predicted_value[0][0]) * W[2][j][0] *
                                hidden[0, j] * (1 - hidden[0, j]) * x[0][i])
            updated_weights[0][i][j] = tmp

    # 更新W[1]的偏置
    for j in range(W[1].shape[0]):
        tmp = W[1][j] - (-2 * (0 - predicted_value[0][0]) * hidden[0, j] *
                         (1 - hidden[0, j]) * W[2][j][0])
        updated_weights[1][j] = tmp

    # 更新W[2]的权重
    for j in range(W[2].shape[0]):
        tmp = W[2][j][0] - (-2 * (0 - predicted_value[0][0]) * hidden[0][j])
        updated_weights[2][j][0] = tmp

    # 更新W[3]的偏置
    tmp = W[3][0] - (-2 * (0 - predicted_value[0][0]))
    updated_weights[3][0] = tmp

    return W, updated_weights, predicted_value


def train_neural_network(epochs=100, lr=0.01):
    """
    训练神经网络
    """
    x = np.array([[1, 1]])
    y = np.array([[0]])

    W = [
        np.array([[-0.0053, 0.3793],
                  [-0.5820, -0.5204],
                  [-0.2723, 0.1896]], dtype=np.float32).T,
        np.array([-0.0140, 0.5607, -0.0628], dtype=np.float32),
        np.array([[0.1528, -0.1745, -0.1135]], dtype=np.float32).T,
        np.array([-0.5516], dtype=np.float32)
    ]

    losses = []
    for epoch in range(epochs):
        W, loss = update_weights_numerical(x, y, W, lr)
        losses.append(loss)

    # 绘制损失曲线
    plt.plot(losses)
    plt.title('Loss over increasing number of epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

    # 最终预测
    pre_hidden = np.dot(x, W[0]) + W[1]
    hidden = 1 / (1 + np.exp(-pre_hidden))
    out = np.dot(hidden, W[2]) + W[3]

    print(f"Final prediction: {out}")
    return W, losses


def train_linear_model(epochs=1000, lr=0.01, verbose=False):
    """
    训练线性模型
    """
    x = np.array([[1], [2], [3], [4]])
    y = np.array([[3], [6], [9], [12]])

    W = [np.array([[0]], dtype=np.float32), np.array([[0]], dtype=np.float32)]

    weight_values = []
    for epoch in range(epochs):
        W, _ = update_weights_numerical(x, y, W, lr, verbose=verbose and epoch < 10)
        weight_values.append(W[0][0][0])

    # 绘制权重变化曲线
    plt.plot(weight_values)
    plt.title(f'Weight value over increasing epochs (lr={lr})')
    plt.xlabel('Epochs')
    plt.ylabel('Weight value')
    plt.show()

    return W, weight_values


def learning_rate_comparison():
    """
    比较不同学习率的效果
    """
    learning_rates = [0.01, 0.1, 1.0]

    for lr in learning_rates:
        print(f"Training with learning rate: {lr}")
        W, weight_values = train_linear_model(epochs=1000, lr=lr)
        print(f"Final weight: {W[0][0][0]:.4f}, Final bias: {W[1][0][0]:.4f}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Neural Network Basic Operations')
    parser.add_argument('--operation', type=str, default='all',
                        choices=['forward', 'gradient', 'chain', 'backprop', 'lr', 'all'],
                        help='Operation to perform')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate')

    args = parser.parse_args()

    if args.operation == 'forward' or args.operation == 'all':
        print("=== Forward Propagation ===")
        x = np.array([[1, 1]])
        y = np.array([[0]])
        W = [
            np.array([[-0.0053, 0.3793],
                      [-0.5820, -0.5204],
                      [-0.2723, 0.1896]], dtype=np.float32).T,
            np.array([-0.0140, 0.5607, -0.0628], dtype=np.float32),
            np.array([[0.1528, -0.1745, -0.1135]], dtype=np.float32).T,
            np.array([-0.5516], dtype=np.float32)
        ]

        loss, out, hidden, pre_hidden = feed_forward_nn(x, y, W)
        print(f"Loss: {loss}")
        print(f"Output: {out}")

    if args.operation == 'gradient' or args.operation == 'all':
        print("\n=== Gradient Descent ===")
        x = np.array([[1, 1]])
        y = np.array([[0]])
        W = [
            np.array([[-0.0053, 0.3793],
                      [-0.5820, -0.5204],
                      [-0.2723, 0.1896]], dtype=np.float32).T,
            np.array([-0.0140, 0.5607, -0.0628], dtype=np.float32),
            np.array([[0.1528, -0.1745, -0.1135]], dtype=np.float32).T,
            np.array([-0.5516], dtype=np.float32)
        ]

        updated_W, loss = update_weights_numerical(x, y, W, args.lr)
        print(f"Original loss: {loss}")
        print("Updated weights:")
        for i, w in enumerate(updated_W):
            print(f"W[{i}]:\n{w}")

    if args.operation == 'chain' or args.operation == 'all':
        print("\n=== Chain Rule Demonstration ===")
        W, updated_W, predicted_value = chain_rule_update_demo()
        print(f"Original prediction: {predicted_value}")
        print("Chain rule updated weights:")
        for i, w in enumerate(updated_W):
            print(f"W[{i}]:\n{w}")

    if args.operation == 'backprop' or args.operation == 'all':
        print("\n=== Backpropagation Training ===")
        W, losses = train_neural_network(epochs=args.epochs, lr=args.lr)
        print("Training completed")

    if args.operation == 'lr' or args.operation == 'all':
        print("\n=== Learning Rate Comparison ===")
        learning_rate_comparison()
