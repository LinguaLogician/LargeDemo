# -*- coding: utf-8 -*-
# @project: LargeDemo
# @filename: ch18_style_transfer.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/8/26 10:27
# https://chat.deepseek.com/a/chat/s/cec5ad74-e921-45ff-8ef1-689ea0ee7c9b

import torch
import torchvision
from datasets import load_dataset
from transformers import PreTrainedModel, PretrainedConfig
from matplotlib import pyplot as plt
import PIL.Image
import argparse


def load_dataset_module():
    """加载数据集模块"""
    dataset = load_dataset('lansinuote/gen.5.flower.book', split='train')
    dataset = dataset.remove_columns(['cls'])

    compose = torchvision.transforms.Compose([
        torchvision.transforms.Resize(224),
        torchvision.transforms.ToTensor(),
    ])

    def transform_function(data):
        image = compose(data['image'][0]).unsqueeze(dim=0)
        return {'image': image}

    dataset = dataset.with_transform(transform_function)

    dataset_tensor = torch.empty(len(dataset), 3, 224, 224)
    for i in range(len(dataset)):
        dataset_tensor[i] = dataset[i]['image']

    return dataset_tensor


def create_data_loader(dataset, batch_size=4):
    """创建数据加载器"""
    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )


def show_images(images):
    """显示图像"""
    images = images.to('cpu').detach()[:4]
    images = images.permute(0, 2, 3, 1)

    plt.figure(figsize=(20, 10))
    for i in range(len(images)):
        plt.subplot(1, 4, i + 1)
        plt.imshow(images[i])
        plt.axis('off')
    plt.show()


class GEN(PreTrainedModel):
    """生成器模型"""
    config_class = PretrainedConfig

    def __init__(self, config):
        super().__init__(config)

        self.encoder = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(padding=4),
            torch.nn.Conv2d(3, 32, kernel_size=9, stride=1, padding=0),
            torch.nn.InstanceNorm2d(32, affine=True),
            torch.nn.ReLU(),
            torch.nn.ReflectionPad2d(padding=1),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=0),
            torch.nn.InstanceNorm2d(64, affine=True),
            torch.nn.ReLU(),
            torch.nn.ReflectionPad2d(padding=1),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0),
            torch.nn.InstanceNorm2d(128, affine=True),
            torch.nn.ReLU(),
        )

        self.middle = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.ReflectionPad2d(padding=1),
                torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0),
                torch.nn.InstanceNorm2d(128, affine=True),
                torch.nn.ReLU(),
                torch.nn.ReflectionPad2d(padding=1),
                torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0),
                torch.nn.InstanceNorm2d(128, affine=True),
            ) for _ in range(5)
        ])

        self.decoder = torch.nn.Sequential(
            torch.nn.UpsamplingNearest2d(scale_factor=2),
            torch.nn.ReflectionPad2d(padding=1),
            torch.nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=0),
            torch.nn.InstanceNorm2d(64, affine=True),
            torch.nn.ReLU(),
            torch.nn.UpsamplingNearest2d(scale_factor=2),
            torch.nn.ReflectionPad2d(padding=1),
            torch.nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=0),
            torch.nn.InstanceNorm2d(32, affine=True),
            torch.nn.ReLU(),
            torch.nn.ReflectionPad2d(padding=4),
            torch.nn.Conv2d(32, 3, kernel_size=9, stride=1, padding=0),
        )

    def forward(self, data):
        data = self.encoder(data)
        for layer in self.middle:
            data = data + layer(data)
        data = self.decoder(data)
        return data


def load_pretrained_cnn():
    """加载预训练的CNN模型"""
    cnn = torchvision.models.vgg16(weights='IMAGENET1K_V1').features[:19]
    cnn.eval()
    return cnn


def get_feature_content(image, cnn_model):
    """获取内容特征"""
    for i, layer in enumerate(cnn_model):
        image = layer(image)
        if i == 6:
            return image


def get_feature_style(image, cnn_model):
    """获取风格特征"""
    features = []
    for i, layer in enumerate(cnn_model):
        image = layer(image)
        if i in [1, 6, 11, 18]:
            num_elements = image.shape[1] * image.shape[2] * image.shape[3]
            data = image.flatten(start_dim=2)
            data = torch.matmul(data, data.permute(0, 2, 1))
            data = data / num_elements
            features.append(data)
    return features


def load_style_image(style_image_path):
    """加载风格图像并提取特征"""
    image_style = PIL.Image.open(style_image_path)

    compose = torchvision.transforms.Compose([
        torchvision.transforms.Resize(250),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        lambda x: x.unsqueeze(dim=0),
    ])
    image_style = compose(image_style)

    show_images(image_style)
    return image_style


def setup_training_environment(gen_model, cnn_model, style_features, device=None):
    """设置训练环境"""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(gen_model.parameters(), lr=2e-4)

    cnn_model.to(device)
    gen_model.to(device)
    style_features = [feature.to(device).detach() for feature in style_features]

    cnn_model.eval()
    gen_model.train()

    return criterion, optimizer, device, style_features


def train_model(gen_model, cnn_model, data_loader, criterion, optimizer,
                style_features, device, num_epochs=20, save_path='save/gen.model'):
    """训练模型"""
    for epoch in range(num_epochs):
        for i, image in enumerate(data_loader):
            image = image.to(device)

            with torch.no_grad():
                target_content = get_feature_content(image, cnn_model)

            image_gen = gen_model(image)

            feature_content = get_feature_content(image_gen, cnn_model)
            feature_style = get_feature_style(image_gen, cnn_model)

            loss_style = sum([criterion(feat, style_feat)
                              for feat, style_feat in zip(feature_style, style_features)])
            loss_content = criterion(feature_content, target_content)

            loss = 1e5 * loss_content + 1e10 * loss_style
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if epoch % 2 == 0:
            print(f"Epoch {epoch}: Content Loss {1e5 * loss_content.item():.2f}, "
                  f"Style Loss {1e10 * loss_style.item():.2f}")
            show_images(image_gen.clip(0, 1))

    torch.save(gen_model.to('cpu'), save_path)
    return gen_model


def test_model(model, data_loader, device=None):
    """测试模型"""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.to(device)
    model.eval()

    with torch.no_grad():
        sample_batch = next(iter(data_loader)).to(device)
        pred = model(sample_batch).clip(0, 1)

    show_images(pred)
    return pred


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Style Transfer Training and Testing')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['all', 'load_data', 'train', 'test', 'test_pretrained'],
                        help='运行模式: all, load_data, train, test, test_pretrained')
    parser.add_argument('--style_image', type=str, default='./datas/style.jpg',
                        help='风格图像路径')
    parser.add_argument('--model_path', type=str, default='save/gen.model',
                        help='模型保存/加载路径')
    parser.add_argument('--epochs', type=int, default=20,
                        help='训练轮数')
    args = parser.parse_args()

    # 根据模式执行相应的操作
    if args.mode in ['all', 'load_data']:
        print("加载数据集...")
        dataset = load_dataset_module()
        data_loader = create_data_loader(dataset)
        print(f"数据集大小: {dataset.shape}")

    if args.mode in ['all', 'train']:
        print("开始训练...")
        dataset = load_dataset_module()
        data_loader = create_data_loader(dataset)

        gen_model = GEN(PretrainedConfig())
        cnn_model = load_pretrained_cnn()

        style_image = load_style_image(args.style_image)
        style_features = get_feature_style(style_image, cnn_model)

        criterion, optimizer, device, style_features = setup_training_environment(
            gen_model, cnn_model, style_features)

        gen_model = train_model(gen_model, cnn_model, data_loader, criterion,
                                optimizer, style_features, device, args.epochs, args.model_path)

    if args.mode in ['all', 'test']:
        print("测试训练好的模型...")
        dataset = load_dataset_module()
        data_loader = create_data_loader(dataset)

        gen_model = torch.load(args.model_path)
        test_model(gen_model, data_loader)

    if args.mode == 'test_pretrained':
        print("测试预训练模型...")
        dataset = load_dataset_module()
        data_loader = create_data_loader(dataset)

        gen_model = GEN.from_pretrained('lansinuote/gen.8.style_transfer.book')
        test_model(gen_model, data_loader)


if __name__ == "__main__":
    main()