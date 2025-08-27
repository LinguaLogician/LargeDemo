# -*- coding: utf-8 -*-
# @project: LargeDemo
# @filename: ch17_cyclegan_model.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/8/26 10:27
# https://chat.deepseek.com/a/chat/s/925c0357-6d5a-477c-927b-59b4abae804f

import torch
import torchvision
from datasets import load_dataset
from matplotlib import pyplot as plt
from transformers import PreTrainedModel, PretrainedConfig


class CycleGAN:
    def __init__(self, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.cls_A = self.get_cls()
        self.cls_B = self.get_cls()
        self.gen_A = self.UNet()
        self.gen_B = self.UNet()

        self.optimizer_cls_A = torch.optim.Adam(self.cls_A.parameters(), lr=2e-4)
        self.optimizer_cls_B = torch.optim.Adam(self.cls_B.parameters(), lr=2e-4)
        self.optimizer_gen_A = torch.optim.Adam(self.gen_A.parameters(), lr=2e-4)
        self.optimizer_gen_B = torch.optim.Adam(self.gen_B.parameters(), lr=2e-4)

        self.criterion_mse = torch.nn.MSELoss()
        self.criterion_l1 = torch.nn.L1Loss()

        # Move models to device
        self.cls_A.to(self.device).train()
        self.cls_B.to(self.device).train()
        self.gen_A.to(self.device).train()
        self.gen_B.to(self.device).train()

        self.loader_A = None
        self.loader_B = None

    class UNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.down = torch.nn.ModuleList([
                torch.nn.Sequential(
                    torch.nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
                    torch.nn.InstanceNorm2d(num_features=32),
                    torch.nn.ReLU(),
                ),
                torch.nn.Sequential(
                    torch.nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
                    torch.nn.InstanceNorm2d(num_features=64),
                    torch.nn.ReLU(),
                ),
                torch.nn.Sequential(
                    torch.nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
                    torch.nn.InstanceNorm2d(num_features=128),
                    torch.nn.ReLU(),
                ),
                torch.nn.Sequential(
                    torch.nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
                    torch.nn.InstanceNorm2d(num_features=256),
                    torch.nn.ReLU(),
                ),
            ])

            self.up = torch.nn.ModuleList([
                torch.nn.Sequential(
                    torch.nn.UpsamplingNearest2d(size=16),
                    torch.nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
                    torch.nn.InstanceNorm2d(num_features=128),
                    torch.nn.ReLU(),
                ),
                torch.nn.Sequential(
                    torch.nn.UpsamplingNearest2d(size=32),
                    torch.nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1),
                    torch.nn.InstanceNorm2d(num_features=64),
                    torch.nn.ReLU(),
                ),
                torch.nn.Sequential(
                    torch.nn.UpsamplingNearest2d(size=64),
                    torch.nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
                    torch.nn.InstanceNorm2d(num_features=32),
                    torch.nn.ReLU(),
                ),
                torch.nn.Sequential(
                    torch.nn.UpsamplingNearest2d(size=128),
                    torch.nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
                    torch.nn.Tanh(),
                ),
            ])

        def forward(self, x):
            down_out = []
            for layer in self.down:
                x = layer(x)
                down_out.append(x)

            up_out = torch.cat((self.up[0](down_out[3]), down_out[2]), dim=1)
            up_out = torch.cat((self.up[1](up_out), down_out[1]), dim=1)
            up_out = torch.cat((self.up[2](up_out), down_out[0]), dim=1)
            up_out = self.up[3](up_out)

            return up_out

    def get_cls(self):
        return torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            torch.nn.InstanceNorm2d(num_features=64),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            torch.nn.InstanceNorm2d(num_features=128),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=1),
            torch.nn.InstanceNorm2d(num_features=256),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=1),
        )

    def set_requires_grad(self, model, requires_grad):
        for param in model.parameters():
            param.requires_grad_(requires_grad)

    def load_data(self, cls_A=0, cls_B=1):
        def get_data(cls):
            dataset = load_dataset('lansinuote/gen.5.flower.book', split='train')

            def f(data):
                return [i == cls for i in data['cls']]

            dataset = dataset.filter(f, batched=True, batch_size=100, num_proc=1)
            dataset = dataset.remove_columns(['cls'])

            compose = torchvision.transforms.Compose([
                torchvision.transforms.Resize(128),
                torchvision.transforms.ToTensor(),
                lambda x: x * 2 - 1,
            ])

            def f(data):
                image = compose(data['image'][0]).unsqueeze(dim=0)
                return {'image': image}

            dataset = dataset.with_transform(f)

            dataset_tensor = torch.empty(len(dataset), 3, 128, 128)
            for i in range(len(dataset)):
                dataset_tensor[i] = dataset[i]['image']

            return dataset_tensor

        data_A = get_data(cls=cls_A)
        data_B = get_data(cls=cls_B)

        self.loader_A = torch.utils.data.DataLoader(
            dataset=data_A, batch_size=1, shuffle=True
        )
        self.loader_B = torch.utils.data.DataLoader(
            dataset=data_B, batch_size=1, shuffle=True
        )

        return data_A.shape, data_B.shape, data_A.dtype

    def show(self, images, title=None):
        images = images.to('cpu').detach()[:50]
        images = images.permute(0, 2, 3, 1)
        images = (images + 1) / 2

        plt.figure(figsize=(20, 10))

        for i in range(min(len(images), 50)):
            plt.subplot(5, 10, i + 1)
            plt.imshow(images[i])
            plt.axis('off')

        if title:
            plt.suptitle(title)

        plt.show()

    def stack(self, loader, n):
        datas = []
        for _ in range(n):
            datas.append(next(iter(loader)))
        return torch.cat(datas, dim=0)

    def train_cls(self):
        self.set_requires_grad(self.cls_A, True)
        self.set_requires_grad(self.cls_B, True)
        self.set_requires_grad(self.gen_A, False)
        self.set_requires_grad(self.gen_B, False)

        def update(cls, optimizer, image, label):
            pred = cls(image)
            label = torch.full((1, 1, 14, 14), label, device=self.device).float()
            loss = self.criterion_mse(pred, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            return loss.item()

        image_A = next(iter(self.loader_A)).to(self.device)
        image_B = next(iter(self.loader_B)).to(self.device)

        with torch.no_grad():
            image_A_gen = self.gen_A(image_B)
            image_B_gen = self.gen_B(image_A)

        loss_sum = 0

        loss = update(self.cls_A, self.optimizer_cls_A, image_A, 1)
        loss_sum += loss

        loss = update(self.cls_A, self.optimizer_cls_A, image_A_gen, 0)
        loss_sum += loss

        loss = update(self.cls_B, self.optimizer_cls_B, image_B, 1)
        loss_sum += loss

        loss = update(self.cls_B, self.optimizer_cls_B, image_B_gen, 0)
        loss_sum += loss

        return loss_sum / 4

    def train_gen(self):
        self.set_requires_grad(self.cls_A, False)
        self.set_requires_grad(self.cls_B, False)
        self.set_requires_grad(self.gen_A, True)
        self.set_requires_grad(self.gen_B, True)

        image_A = next(iter(self.loader_A)).to(self.device)
        image_B = next(iter(self.loader_B)).to(self.device)

        # A变B,B变A
        image_gen_B = self.gen_B(image_A)
        image_gen_A = self.gen_A(image_B)

        losses = []

        # 计算转换的成绩
        loss = self.criterion_mse(
            self.cls_A(image_gen_A),
            torch.ones(1, 1, 14, 14, device=self.device)
        )
        losses.append(loss)

        loss = self.criterion_mse(
            self.cls_B(image_gen_B),
            torch.ones(1, 1, 14, 14, device=self.device)
        )
        losses.append(loss)

        # 把转换过的图片再转换回来
        loss = self.criterion_l1(self.gen_A(image_gen_B), image_A) * 10
        losses.append(loss)

        loss = self.criterion_l1(self.gen_B(image_gen_A), image_B) * 10
        losses.append(loss)

        # A变A,B变B,这次转换应该是不变的
        loss = self.criterion_l1(self.gen_A(image_A), image_A) * 2
        losses.append(loss)

        loss = self.criterion_l1(self.gen_B(image_B), image_B) * 2
        losses.append(loss)

        loss = sum(losses)
        loss.backward()

        self.optimizer_gen_A.step()
        self.optimizer_gen_B.step()

        self.optimizer_gen_A.zero_grad()
        self.optimizer_gen_B.zero_grad()

        return loss.item()

    def train(self, epochs=40000, save_interval=8000, save_path='save/'):
        for epoch in range(epochs):
            loss_cls = self.train_cls()
            loss_gen = self.train_gen()

            if epoch % save_interval == 0:
                print(f"Epoch {epoch}: CLS Loss = {loss_cls:.6f}, GEN Loss = {loss_gen:.6f}")

                image_A = self.stack(self.loader_A, 10)
                image_B = self.stack(self.loader_B, 10)

                print('A')
                self.show(image_A, 'Original A Images')

                print('A to B')
                self.show(self.gen_B(image_A.to(self.device)), 'A to B Translation')

                print('B')
                self.show(image_B, 'Original B Images')

                print('B to A')
                self.show(self.gen_A(image_B.to(self.device)), 'B to A Translation')

        # Save models
        torch.save(self.cls_A.to('cpu'), f'{save_path}cls_A.model')
        torch.save(self.cls_B.to('cpu'), f'{save_path}cls_B.model')
        torch.save(self.gen_A.to('cpu'), f'{save_path}gen_A.model')
        torch.save(self.gen_B.to('cpu'), f'{save_path}gen_B.model')

        # Move back to device
        self.cls_A.to(self.device)
        self.cls_B.to(self.device)
        self.gen_A.to(self.device)
        self.gen_B.to(self.device)

    def test(self):
        with torch.no_grad():
            image_A = self.stack(self.loader_A, 10)
            print('A')
            self.show(image_A, 'Original A Images')

            print('A to B')
            self.show(self.gen_B(image_A.to(self.device)), 'A to B Translation')

            image_B = self.stack(self.loader_B, 10)
            print('B')
            self.show(image_B, 'Original B Images')

            print('B to A')
            self.show(self.gen_A(image_B.to(self.device)), 'B to A Translation')

    def load_pretrained_models(self, path='save/'):
        self.cls_A = torch.load(f'{path}cls_A.model').to(self.device)
        self.cls_B = torch.load(f'{path}cls_B.model').to(self.device)
        self.gen_A = torch.load(f'{path}gen_A.model').to(self.device)
        self.gen_B = torch.load(f'{path}gen_B.model').to(self.device)

    def load_huggingface_model(self, model_name='lansinuote/gen.6.cyclegan.book'):
        class Model(PreTrainedModel):
            config_class = PretrainedConfig

            def __init__(self, config):
                super().__init__(config)
                self.cls_A = None
                self.cls_B = None
                self.gen_A = None
                self.gen_B = None

        model = Model.from_pretrained(model_name)
        self.cls_A = model.cls_A.to(self.device)
        self.cls_B = model.cls_B.to(self.device)
        self.gen_A = model.gen_A.to(self.device)
        self.gen_B = model.gen_B.to(self.device)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='CycleGAN Training and Testing')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'test', 'load_pretrained', 'load_huggingface'],
                        help='Mode to run: train, test, load_pretrained, or load_huggingface')
    parser.add_argument('--epochs', type=int, default=40000, help='Number of training epochs')
    parser.add_argument('--save_interval', type=int, default=8000, help='Interval for saving and displaying results')
    parser.add_argument('--save_path', type=str, default='save/', help='Path to save models')
    parser.add_argument('--model_path', type=str, default='save/', help='Path to load pretrained models')
    parser.add_argument('--hf_model', type=str, default='lansinuote/gen.6.cyclegan.book',
                        help='HuggingFace model name')

    args = parser.parse_args()

    # Initialize CycleGAN
    cyclegan = CycleGAN()

    # Load data
    print("Loading data...")
    cyclegan.load_data()

    if args.mode == 'train':
        print("Starting training...")
        cyclegan.train(epochs=args.epochs, save_interval=args.save_interval, save_path=args.save_path)

    elif args.mode == 'test':
        print("Testing with current models...")
        cyclegan.test()

    elif args.mode == 'load_pretrained':
        print("Loading pretrained models...")
        cyclegan.load_pretrained_models(path=args.model_path)
        print("Testing pretrained models...")
        cyclegan.test()

    elif args.mode == 'load_huggingface':
        print("Loading HuggingFace model...")
        cyclegan.load_huggingface_model(model_name=args.hf_model)
        print("Testing HuggingFace model...")
        cyclegan.test()


if __name__ == '__main__':
    main()
