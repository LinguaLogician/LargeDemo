# -*- coding: utf-8 -*-
# @project: LargeDemo
# @filename: ch4_text_training.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/8/25 22:09
# https://chat.deepseek.com/a/chat/s/352c1ead-bfc9-45a9-9cfc-d81152f601b4

import torch
import torch.nn as nn
import torchvision
import numpy as np
import PIL.Image
from matplotlib import pyplot as plt
from datasets import Dataset, load_dataset
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import (
    StableDiffusionPipeline,
    AutoencoderKL,
    UNet2DConditionModel,
    DDPMScheduler
)


class TextualInversionTrainer:
    """文本反转训练器"""

    def __init__(self, checkpoint='runwayml/stable-diffusion-v1-5'):
        self.checkpoint = checkpoint
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 初始化组件
        self.tokenizer = None
        self.encoder = None
        self.vae = None
        self.unet = None
        self.scheduler = None
        self.optimizer = None
        self.criterion = None
        self.dataset = None
        self.loader = None

        # 描述文本
        self.texts = [
            'a photo of a <cat-toy>', 'a rendering of a <cat-toy>',
            'a cropped photo of the <cat-toy>', 'the photo of a <cat-toy>',
            'a photo of a clean <cat-toy>', 'a photo of a dirty <cat-toy>',
            'a dark photo of the <cat-toy>', 'a photo of my <cat-toy>',
            'a photo of the cool <cat-toy>', 'a close-up photo of a <cat-toy>',
            'a bright photo of the <cat-toy>', 'a cropped photo of a <cat-toy>',
            'a photo of the <cat-toy>', 'a good photo of the <cat-toy>',
            'a photo of one <cat-toy>', 'a close-up photo of the <cat-toy>',
            'a rendition of the <cat-toy>', 'a photo of the clean <cat-toy>',
            'a rendition of a <cat-toy>', 'a photo of a nice <cat-toy>',
            'a good photo of a <cat-toy>', 'a photo of the nice <cat-toy>',
            'a photo of the small <cat-toy>', 'a photo of the weird <cat-toy>',
            'a photo of the large <cat-toy>', 'a photo of a cool <cat-toy>',
            'a photo of a small <cat-toy>'
        ]

    def load_models(self):
        """加载预训练模型"""
        print("Loading models...")
        self.tokenizer = CLIPTokenizer.from_pretrained(self.checkpoint, subfolder='tokenizer')
        self.encoder = CLIPTextModel.from_pretrained(self.checkpoint, subfolder='text_encoder')
        self.vae = AutoencoderKL.from_pretrained(self.checkpoint, subfolder='vae')
        self.unet = UNet2DConditionModel.from_pretrained(self.checkpoint, subfolder='unet')
        self.scheduler = DDPMScheduler.from_pretrained(self.checkpoint, subfolder='scheduler')

        # 移动到设备
        self.encoder.to(self.device)
        self.vae.to(self.device)
        self.unet.to(self.device)

        # 打印模型大小
        self._print_model_sizes()

    def _print_model_sizes(self):
        """打印模型参数量"""

        def print_model_size(name, model):
            print(f"{name}: {sum(i.numel() for i in model.parameters()) / 10000:.4f}")

        print_model_size('encoder', self.encoder)
        print_model_size('vae', self.vae)
        print_model_size('unet', self.unet)

    def load_local_dataset(self):
        """加载本地数据集"""
        print("Loading local dataset...")
        images = [{'image': PIL.Image.open(f'images/{i}.jpeg')} for i in range(6)]
        self.dataset = Dataset.from_list(images)
        return self.dataset

    def load_online_dataset(self, repo_id='lansinuote/diffusion.2.textual_inversion'):
        """加载在线数据集"""
        print("Loading online dataset...")
        self.dataset = load_dataset(path=repo_id, split='train')
        return self.dataset

    def preprocess_dataset(self):
        """预处理数据集"""
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_local_dataset() or load_online_dataset() first.")

        print("Preprocessing dataset...")

        # 数据增强
        compose = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
        ])

        def transform_func(data):
            # 编码文字
            input_ids = self.tokenizer(
                np.random.choice(self.texts),
                padding='max_length',
                truncation=True,
                max_length=77,
                return_tensors='pt'
            )['input_ids']

            # 编码图片
            pixel_values = []
            for i in range(len(data['image'])):
                image = data['image'][i]

                # 数据增强
                image = compose(image)

                # 尺寸缩放
                image = image.resize((512, 512), resample=PIL.Image.Resampling.BICUBIC)

                # 数值操作
                image = np.array(image).astype(np.uint8)
                image = image / 127.5 - 1.0
                image = image.astype(np.float32)

                # 转tensor,把通道维度放在前面
                image = torch.from_numpy(image).permute(2, 0, 1)
                pixel_values.append(image)

            return {'input_ids': input_ids, 'pixel_values': pixel_values}

        self.dataset = self.dataset.with_transform(transform_func)
        return self.dataset

    def create_data_loader(self, batch_size=1, shuffle=True):
        """创建数据加载器"""
        if self.dataset is None:
            raise ValueError("Dataset not prepared. Call preprocess_dataset() first.")

        print("Creating data loader...")
        self.loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle
        )
        return self.loader

    def add_new_token(self, new_token='<cat-toy>', init_token='toy'):
        """添加新词到tokenizer和encoder"""
        print(f"Adding new token: {new_token}")

        # 字典里添加新词
        self.tokenizer.add_tokens(new_token)

        # 扩展encoder的embed层
        self.encoder.resize_token_embeddings(len(self.tokenizer))

        # 取新旧两个词的id
        old_id = self.tokenizer.convert_tokens_to_ids(init_token)
        new_id = self.tokenizer.convert_tokens_to_ids(new_token)

        embed = self.encoder.get_input_embeddings().weight.data

        # 以旧词来初始化新词
        embed[new_id] = embed[old_id]

    def freeze_models(self):
        """冻结不需要训练的模型参数"""
        print("Freezing models...")

        # 这两个模型不更新参数
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)

        # 只训练encoder.text_model.embeddings.token_embedding层
        self.encoder.text_model.encoder.requires_grad_(False)
        self.encoder.text_model.final_layer_norm.requires_grad_(False)
        self.encoder.text_model.embeddings.position_embedding.requires_grad_(False)

    def setup_optimizer(self, lr=5e-4, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-8):
        """设置优化器"""
        print("Setting up optimizer...")
        self.optimizer = torch.optim.AdamW(
            self.encoder.get_input_embeddings().parameters(),
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            eps=eps
        )
        self.criterion = nn.MSELoss()
        return self.optimizer, self.criterion

    def compute_loss(self, data):
        """计算损失"""
        device = data['input_ids'].device

        # 编码文字
        out_encoder = self.encoder(data['input_ids'])[0]

        # 计算特征图
        out_vae = self.vae.encode(data['pixel_values']).latent_dist.sample().detach()
        out_vae = out_vae * 0.18215  # vae.config.scaling_factor

        # 随机噪声
        noise = torch.randn_like(out_vae)

        # 随机噪声步
        noise_step = torch.randint(0, 1000, (1,), device=device).long()

        # 添加噪声
        out_vae_noise = self.scheduler.add_noise(out_vae, noise, noise_step)

        # 从噪声图中把噪声计算出来
        out_unet = self.unet(out_vae_noise, noise_step, out_encoder).sample

        return self.criterion(out_unet, noise)

    def train(self, epochs=800, accumulation_steps=4, save_path='./save'):
        """训练模型"""
        if None in [self.loader, self.optimizer, self.criterion]:
            raise ValueError("Please setup data loader and optimizer first.")

        print("Starting training...")
        self.encoder.train()
        loss_sum = 0

        for epoch in range(epochs):
            for i, data in enumerate(self.loader):
                # 移动到设备
                for k in data.keys():
                    data[k] = data[k].to(self.device)

                # 计算损失并反向传播
                loss = self.compute_loss(data) / accumulation_steps
                loss.backward()

                # 积累更新
                if (epoch * len(self.loader) + i) % accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                loss_sum += loss.item()

            # 打印损失
            if epoch % 20 == 0:
                print(f"Epoch {epoch}: Loss = {loss_sum}")
                loss_sum = 0

        # 保存模型
        print("Saving model...")
        pipeline = StableDiffusionPipeline.from_pretrained(
            self.checkpoint,
            text_encoder=self.encoder,
            vae=self.vae,
            unet=self.unet,
            tokenizer=self.tokenizer,
            safety_checker=None
        )
        pipeline.save_pretrained(save_path)
        print(f"Model saved to {save_path}")


class TextualInversionTester:
    """文本反转测试器"""

    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def test(self, pipeline):
        """测试模型"""
        pipeline = pipeline.to(self.device)

        texts = [
            'a photo of a <cat-toy>', 'a photo of a clean <cat-toy>',
            'a photo of a dirty <cat-toy>', 'a dark photo of the <cat-toy>'
        ]

        images = []
        for text in texts:
            image = pipeline(text, num_inference_steps=25).images[0]
            images.append(image)

        # 显示图像
        plt.figure(figsize=(20, 10))
        for i in range(4):
            plt.subplot(1, 4, i + 1)
            plt.imshow(images[i])
            plt.axis('off')

        plt.show()

    def test_pretrained(self, model_path='runwayml/stable-diffusion-v1-5'):
        """测试预训练模型"""
        print(f"Testing pretrained model: {model_path}")
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_path,
            safety_checker=None
        )
        self.test(pipeline)

    def test_custom(self, model_path='./save'):
        """测试自定义模型"""
        print(f"Testing custom model: {model_path}")
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_path,
            safety_checker=None
        )
        self.test(pipeline)

    def test_online(self, repo_id='lansinuote/diffusion.2.textual_inversion'):
        """测试在线模型"""
        print(f"Testing online model: {repo_id}")
        pipeline = StableDiffusionPipeline.from_pretrained(
            repo_id,
            safety_checker=None
        )
        self.test(pipeline)


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='Textual Inversion Training and Testing')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['train', 'test_pretrained', 'test_custom', 'test_online'],
                        help='Mode to run: train or test')
    parser.add_argument('--checkpoint', type=str, default='runwayml/stable-diffusion-v1-5',
                        help='Pretrained model checkpoint')
    parser.add_argument('--dataset_source', type=str, default='local',
                        choices=['local', 'online'],
                        help='Dataset source: local or online')
    parser.add_argument('--repo_id', type=str,
                        default='lansinuote/diffusion.2.textual_inversion',
                        help='HuggingFace repository ID for online dataset/model')
    parser.add_argument('--epochs', type=int, default=800,
                        help='Number of training epochs')
    parser.add_argument('--save_path', type=str, default='./save',
                        help='Path to save trained model')

    args = parser.parse_args()

    if args.mode == 'train':
        # 训练模式
        trainer = TextualInversionTrainer(checkpoint=args.checkpoint)
        trainer.load_models()

        # 加载数据集
        if args.dataset_source == 'local':
            trainer.load_local_dataset()
        else:
            trainer.load_online_dataset(repo_id=args.repo_id)

        # 预处理数据
        trainer.preprocess_dataset()
        trainer.create_data_loader()

        # 设置训练
        trainer.add_new_token()
        trainer.freeze_models()
        trainer.setup_optimizer()

        # 开始训练
        trainer.train(epochs=args.epochs, save_path=args.save_path)

    else:
        # 测试模式
        tester = TextualInversionTester()

        if args.mode == 'test_pretrained':
            tester.test_pretrained(model_path=args.checkpoint)
        elif args.mode == 'test_custom':
            tester.test_custom(model_path=args.save_path)
        elif args.mode == 'test_online':
            tester.test_online(repo_id=args.repo_id)


if __name__ == '__main__':
    main()