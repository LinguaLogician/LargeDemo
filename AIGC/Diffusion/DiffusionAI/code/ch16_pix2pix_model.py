# -*- coding: utf-8 -*-
# @project: LargeDemo
# @filename: ch16_pix2pix_model.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/8/26 10:26
# https://chat.deepseek.com/a/chat/s/1ecaf300-d868-4954-af06-1fb210f082e5

import random
import torch
import torchvision
from datasets import load_dataset
from matplotlib import pyplot as plt
from transformers import PreTrainedModel, PretrainedConfig


class Dataset(torch.utils.data.Dataset):
    def __init__(self, part, facades_dict):
        super().__init__()
        self.dataset = facades_dict[part]
        self.to_tensor = torchvision.transforms.ToTensor()
        self.resize = torchvision.transforms.Resize(
            size=286,
            interpolation=torchvision.transforms.functional.InterpolationMode.BICUBIC
        )
        self.norm = torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.flip = torchvision.transforms.RandomHorizontalFlip(p=1)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        a = self.dataset[idx]['a']
        b = self.dataset[idx]['b']

        a = self.resize(a)
        b = self.resize(b)

        a = self.to_tensor(a)
        b = self.to_tensor(b)

        w = random.randint(0, 29)
        h = random.randint(0, 29)
        a = a[:, h:h + 256, w:w + 256]
        b = b[:, h:h + 256, w:w + 256]

        a = self.norm(a)
        b = self.norm(b)

        if random.random() < 0.5:
            a = self.flip(a)
            b = self.flip(b)

        return b, a


class Res(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.s = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=0, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(True),
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=0, bias=False),
            torch.nn.BatchNorm2d(256),
        )

    def forward(self, x):
        out = x + self.s(x)
        return torch.nn.ReLU(True)(out)


def show(images):
    images = images.to('cpu').detach().numpy()
    images = images[:3]
    images = (images + 1) / 2
    images = images.transpose(0, 2, 3, 1)

    plt.figure(figsize=(14, 10))
    for i in range(len(images)):
        plt.subplot(1, 3, i + 1)
        plt.imshow(images[i])
        plt.axis('off')
    plt.show()


def set_requires_grad(model, requires_grad):
    for param in model.parameters():
        param.requires_grad_(requires_grad)


def load_facades_dataset():
    facades = load_dataset('lansinuote/gen.3.facades', split='train')
    facades = facades.train_test_split(test_size=15)
    return facades


def initialize_models_and_optimizers(device):
    cls = torch.nn.Sequential(
        torch.nn.Conv2d(6, 64, kernel_size=4, stride=2, padding=1),
        torch.nn.LeakyReLU(0.2),
        torch.nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
        torch.nn.BatchNorm2d(128),
        torch.nn.LeakyReLU(0.2),
        torch.nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
        torch.nn.BatchNorm2d(256),
        torch.nn.LeakyReLU(0.2),
        torch.nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1, bias=False),
        torch.nn.BatchNorm2d(512),
        torch.nn.LeakyReLU(0.2),
        torch.nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
    )

    gen = torch.nn.Sequential(
        torch.nn.Sequential(
            torch.nn.ReflectionPad2d(3),
            torch.nn.Conv2d(3, 64, kernel_size=7, padding=0, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
        ),
        torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
        ),
        torch.nn.Sequential(*[Res() for _ in range(9)]),
        torch.nn.Sequential(
            torch.nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
        ),
        torch.nn.Sequential(
            torch.nn.ReflectionPad2d(3),
            torch.nn.Conv2d(64, 3, kernel_size=7, padding=0),
            torch.nn.Tanh(),
        ),
    )

    criterion_l1 = torch.nn.L1Loss()
    criterion_mse = torch.nn.MSELoss()

    optimizer_gen = torch.optim.Adam(gen.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optimizer_cls = torch.optim.Adam(cls.parameters(), lr=2e-4, betas=(0.5, 0.999))

    scheduler_gen = torch.optim.lr_scheduler.StepLR(optimizer_gen, step_size=1, gamma=0.99)
    scheduler_cls = torch.optim.lr_scheduler.StepLR(optimizer_cls, step_size=1, gamma=0.99)

    gen.to(device)
    cls.to(device)

    gen.train()
    cls.train()

    return cls, gen, criterion_l1, criterion_mse, optimizer_gen, optimizer_cls, scheduler_gen, scheduler_cls


def train_cls(cls, gen, loader, device, criterion_mse, optimizer_cls, optimizer_gen):
    set_requires_grad(cls, True)
    set_requires_grad(gen, False)

    a_real, b_real = next(iter(loader))
    a_real = a_real.to(device)
    b_real = b_real.to(device)

    with torch.no_grad():
        b_fake = gen(a_real)

    pred = cls(torch.cat((a_real, b_fake), dim=1))
    loss_fake = criterion_mse(pred, torch.zeros(1, 1, 30, 30, device=device))

    pred = cls(torch.cat((a_real, b_real), 1))
    loss_real = criterion_mse(pred, torch.ones(1, 1, 30, 30, device=device))

    loss = loss_fake + loss_real
    loss.backward()
    optimizer_cls.step()
    optimizer_cls.zero_grad()
    optimizer_gen.zero_grad()

    return loss.item()


def train_gen(cls, gen, loader, device, criterion_mse, criterion_l1, optimizer_gen, optimizer_cls):
    set_requires_grad(cls, False)
    set_requires_grad(gen, True)

    a_real, b_real = next(iter(loader))
    a_real = a_real.to(device)
    b_real = b_real.to(device)

    b_fake = gen(a_real)

    pred = cls(torch.cat((a_real, b_fake), dim=1))
    loss_mse = criterion_mse(pred, torch.ones(1, 1, 30, 30, device=device))
    loss_l1 = criterion_l1(b_fake, b_real)
    loss = loss_mse + loss_l1 * 10

    loss.backward()
    optimizer_gen.step()
    optimizer_cls.zero_grad()
    optimizer_gen.zero_grad()

    return loss.item()


def train_models(gen, cls, loader, device, criterion_l1, criterion_mse, optimizer_gen, optimizer_cls, scheduler_gen, scheduler_cls):
    for epoch in range(200):
        for _ in range(len(loader)):
            loss_cls = train_cls(cls, gen, loader, device, criterion_mse, optimizer_cls, optimizer_gen)
            loss_gen = train_gen(cls, gen, loader, device, criterion_mse, criterion_l1, optimizer_gen, optimizer_cls)

        if epoch > 100:
            scheduler_gen.step()
            scheduler_cls.step()

        lr_gen = optimizer_gen.param_groups[0]['lr']
        lr_cls = optimizer_cls.param_groups[0]['lr']

        if epoch % 20 == 0:
            print(epoch, loss_cls, loss_gen, lr_gen, lr_cls)

            a_real, b_real = next(iter(loader))
            a_real = a_real.to(device)
            b_real = b_real.to(device)

            with torch.no_grad():
                b_fake = gen(a_real)

            show(torch.cat((a_real, b_real, b_fake), dim=0))

    torch.save(cls.to('cpu'), 'save/cls.model')
    torch.save(gen.to('cpu'), 'save/gen.model')


def test_model(gen, facades):
    loader_test = torch.utils.data.DataLoader(
        dataset=Dataset('test', facades),
        batch_size=1,
        shuffle=True
    )

    for i, (a_real, b_real) in enumerate(loader_test):
        with torch.no_grad():
            b_fake = gen(a_real)
            show(torch.cat((a_real, b_real, b_fake), dim=0))
            if i == 5:
                break


def load_pretrained_model():
    class Model(PreTrainedModel):
        config_class = PretrainedConfig

        def __init__(self, config):
            super().__init__(config)
            self.cls = None
            self.gen = None

    model = Model.from_pretrained('lansinuote/gen.11.pix2pix')
    return model.gen


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run Pix2Pix training or testing.')
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'test_pretrained'], default='train',
                        help='Mode to run: train, test, or test_pretrained')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    facades = load_facades_dataset()

    if args.mode == 'train':
        dataset = Dataset('train', facades)
        loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=True)
        cls, gen, criterion_l1, criterion_mse, optimizer_gen, optimizer_cls, scheduler_gen, scheduler_cls = initialize_models_and_optimizers(device)
        train_models(gen, cls, loader, device, criterion_l1, criterion_mse, optimizer_gen, optimizer_cls, scheduler_gen, scheduler_cls)

    elif args.mode == 'test':
        gen = torch.load('save/gen.model')
        test_model(gen, facades)

    elif args.mode == 'test_pretrained':
        gen = load_pretrained_model()
        test_model(gen, facades)


if __name__ == '__main__':
    main()