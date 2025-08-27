# -*- coding: utf-8 -*-
# @project: LargeDemo
# @filename: ch14_instruct_pix2pix.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/8/26 10:24
# https://chat.deepseek.com/a/chat/s/aa076cba-ec6a-466d-9537-c6b6748816e5
import torch
import torchvision
from datasets import load_dataset
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import (
    StableDiffusionInstructPix2PixPipeline,
    AutoencoderKL,
    UNet2DConditionModel,
    DDPMScheduler,
    StableDiffusionPipeline
)
from matplotlib import pyplot as plt
from typing import Dict, List, Tuple, Optional
import argparse


class InstructPix2PixConfig:
    """配置类，存储所有全局常量"""

    def __init__(self):
        self.repo_id = 'lansinuote/diffusion.8.instruct_pix2pix'
        self.checkpoint = 'runwayml/stable-diffusion-v1-5'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.batch_size = 4
        self.num_epochs = 4000
        self.learning_rate = 5e-5
        self.save_dir = './save'


class DataProcessor:
    """数据处理类，负责数据加载和预处理"""

    def __init__(self, config: InstructPix2PixConfig):
        self.config = config
        self.tokenizer = CLIPTokenizer.from_pretrained(
            config.checkpoint, subfolder='tokenizer'
        )
        self.compose = self._create_transform()

    def _create_transform(self):
        """创建图像预处理转换"""
        return torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.ToTensor(),
            lambda x: (x * 2) - 1,
        ])

    def load_and_preprocess_data(self) -> torch.utils.data.DataLoader:
        """加载并预处理数据集"""
        dataset = load_dataset(path=self.config.repo_id, split='train')
        dataset = dataset.with_transform(self._transform_function)

        # 打印数据集信息
        for k, v in dataset[0].items():
            print(f"{k} {v.shape} {v.dtype}")
        print(dataset)

        # 创建数据加载器
        loader = torch.utils.data.DataLoader(
            dataset,
            shuffle=True,
            batch_size=self.config.batch_size
        )

        # 打印批次信息
        batch = next(iter(loader))
        for k, v in batch.items():
            print(f"{k} {v.shape} {v.dtype}")
        print(f"Number of batches: {len(loader)}")

        return loader

    def _transform_function(self, data: Dict) -> Dict:
        """数据转换函数"""
        # 图像编码
        input_images = [self.compose(img) for img in data['input']]
        output_images = [self.compose(img) for img in data['output']]

        # 文字编码
        text_tokens = self.tokenizer.batch_encode_plus(
            data['text'],
            max_length=77,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).input_ids

        return {
            'input': input_images,
            'output': output_images,
            'text': text_tokens
        }


class ModelManager:
    """模型管理类，负责加载和管理所有模型"""

    def __init__(self, config: InstructPix2PixConfig):
        self.config = config
        self.encoder = None
        self.vae = None
        self.unet = None
        self.scheduler = None
        self.optimizer = None
        self.criterion = None

    def load_models(self):
        """加载所有预训练模型"""
        # 加载3个主要模型
        self.encoder = CLIPTextModel.from_pretrained(
            self.config.checkpoint, subfolder='text_encoder'
        )
        self.vae = AutoencoderKL.from_pretrained(
            self.config.checkpoint, subfolder='vae'
        )
        self.unet = UNet2DConditionModel.from_pretrained(
            self.config.checkpoint, subfolder='unet'
        )

        # 修改unet.conv_in层的形状
        self._modify_unet_input_channels()

        # 打印模型大小
        self._print_model_sizes()

        # 加载调度器
        self.scheduler = DDPMScheduler.from_pretrained(
            self.config.checkpoint, subfolder='scheduler'
        )

        # 初始化优化器和损失函数
        self.optimizer = torch.optim.AdamW(
            self.unet.parameters(),
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01,
            eps=1e-8
        )
        self.criterion = torch.nn.MSELoss()

        return self.encoder, self.vae, self.unet, self.scheduler, self.optimizer, self.criterion

    def _modify_unet_input_channels(self):
        """修改UNet输入通道数"""
        self.unet.register_to_config(in_channels=8)
        with torch.no_grad():
            new_conv_in = torch.nn.Conv2d(8, 320, 3, 1, 1)
            new_conv_in.weight.zero_()
            new_conv_in.weight[:, :4, :, :].copy_(self.unet.conv_in.weight)
            self.unet.conv_in = new_conv_in
        print(f"UNet input channels: {self.unet.config.in_channels}")

    def _print_model_sizes(self):
        """打印模型参数量"""

        def print_model_size(name, model):
            param_count = sum(p.numel() for p in model.parameters()) / 10000
            print(f"{name} {param_count:.4f}")

        print_model_size('encoder', self.encoder)
        print_model_size('vae', self.vae)
        print_model_size('unet', self.unet)


class TrainingUtils:
    """训练工具类，包含训练过程中的辅助函数"""

    def __init__(self, config: InstructPix2PixConfig, model_manager: ModelManager):
        self.config = config
        self.model_manager = model_manager

    def dropout_data(self, out_encoder: torch.Tensor, input_images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """随机丢弃部分数据以增加鲁棒性"""
        # 输入图编码
        out_vae_input = self.model_manager.vae.encode(input_images).latent_dist.mode()

        # 生成mask
        r = torch.rand(self.config.batch_size, device=out_encoder.device)
        mask_text = (r > 0.1).reshape(self.config.batch_size, 1, 1)
        mask_image = torch.logical_or(r < 0.05, r > 0.15).float().reshape(self.config.batch_size, 1, 1, 1)

        # 编码负采样的文本
        tokenizer = CLIPTokenizer.from_pretrained(self.config.checkpoint, subfolder='tokenizer')
        out_encoder_neg = tokenizer.batch_encode_plus(
            [''],
            max_length=77,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).input_ids.to(out_encoder.device)
        out_encoder_neg = self.model_manager.encoder(out_encoder_neg)[0]

        # 使用mask混合正负编码
        out_encoder = torch.where(mask_text, out_encoder, out_encoder_neg)
        out_vae_input = mask_image * out_vae_input

        return out_encoder, out_vae_input

    def get_loss(self, data: Dict) -> torch.Tensor:
        """计算损失函数"""
        # 文字编码
        out_encoder = self.model_manager.encoder(data['text'])[0]

        # 输出图编码
        out_vae_output = self.model_manager.vae.encode(data['output']).latent_dist.sample()
        out_vae_output = out_vae_output * 0.18215  # vae.config.scaling_factor

        # 随机噪声
        noise = torch.randn_like(out_vae_output)

        # 往特征图中添加噪声
        noise_step = torch.randint(0, 1000, (self.config.batch_size,)).long().to(out_encoder.device)
        out_vae_noise = self.model_manager.scheduler.add_noise(out_vae_output, noise, noise_step)

        # 使用mask组合正负采样的文本编码数据
        out_encoder, out_vae_input = self.dropout_data(out_encoder, data['input'])

        # 向out_vae_noise中组合输入图的数据
        out_vae_noise = torch.cat([out_vae_noise, out_vae_input], dim=1)

        # 根据文字信息，把特征图中的噪声计算出来
        out_unet = self.model_manager.unet(
            out_vae_noise,
            noise_step,
            encoder_hidden_states=out_encoder
        ).sample

        # 计算loss
        return self.model_manager.criterion(out_unet, noise)


class Trainer:
    """训练器类，负责模型训练"""

    def __init__(self, config: InstructPix2PixConfig, model_manager: ModelManager, training_utils: TrainingUtils):
        self.config = config
        self.model_manager = model_manager
        self.training_utils = training_utils

    def train(self, data_loader: torch.utils.data.DataLoader):
        """训练模型"""
        # 移动模型到设备
        self.model_manager.unet.to(self.config.device)
        self.model_manager.encoder.to(self.config.device)
        self.model_manager.vae.to(self.config.device)

        # 冻结不需要训练的模型
        self.model_manager.vae.requires_grad_(False)
        self.model_manager.encoder.requires_grad_(False)
        self.model_manager.unet.train()

        loss_sum = 0
        for epoch in range(self.config.num_epochs):
            for i, data in enumerate(data_loader):
                # 移动数据到设备
                for k in data.keys():
                    data[k] = data[k].to(self.config.device)

                # 计算损失并反向传播
                loss = self.training_utils.get_loss(data) / self.config.batch_size
                loss.backward()
                loss_sum += loss.item()

                # 每4个批次更新一次参数
                if i % 4 == 0:
                    torch.nn.utils.clip_grad_norm_(self.model_manager.unet.parameters(), 1.0)
                    self.model_manager.optimizer.step()
                    self.model_manager.optimizer.zero_grad()

            # 每20个epoch打印一次损失并保存模型
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch}, Loss: {loss_sum}")
                loss_sum = 0
                self.save_model()

    def save_model(self):
        """保存模型"""
        pipeline = StableDiffusionPipeline.from_pretrained(
            self.config.checkpoint,
            text_encoder=self.model_manager.encoder,
            vae=self.model_manager.vae,
            unet=self.model_manager.unet
        )
        pipeline.save_pretrained(self.config.save_dir)
        print(f"Model saved to {self.config.save_dir}")


class Tester:
    """测试器类，负责模型测试"""

    def __init__(self, config: InstructPix2PixConfig):
        self.config = config
        self.dataset = None

    def load_test_dataset(self):
        """加载测试数据集"""
        self.dataset = load_dataset(path=self.config.repo_id, split='train')
        return self.dataset

    def test_model(self, model_path: Optional[str] = None):
        """测试模型性能"""
        # 加载模型
        if model_path:
            pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                model_path, safety_checker=None
            )
        else:
            pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                self.config.repo_id, safety_checker=None
            )

        pipeline = pipeline.to(self.config.device)

        # 测试指定的样本
        test_indices = [
            0, 14, 16, 23, 43, 63, 65, 66, 72, 78, 79, 80, 82, 91, 101, 115,
            121, 133, 135, 136, 142, 143, 145, 148, 156, 169, 171, 175, 176
        ]

        for i in test_indices:
            input_image = self.dataset[i]['input']
            text = self.dataset[i]['text']
            output_image = self.dataset[i]['output']

            # 生成预测
            pred = pipeline(
                text,
                image=input_image,
                num_inference_steps=250,
                image_guidance_scale=1.5,
                guidance_scale=10
            ).images[0]

            print(f"{i} {text}")

            # 显示结果
            plt.figure(figsize=(20, 10))

            plt.subplot(1, 3, 1)
            plt.imshow(input_image)
            plt.axis('off')
            plt.title('Input')

            plt.subplot(1, 3, 2)
            plt.imshow(output_image)
            plt.axis('off')
            plt.title('Ground Truth')

            plt.subplot(1, 3, 3)
            plt.imshow(pred)
            plt.axis('off')
            plt.title('Prediction')

            plt.show()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='InstructPix2Pix Training and Testing')
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'],
                        help='运行模式: train 或 test')
    parser.add_argument('--model_path', type=str, default=None,
                        help='测试时使用的模型路径（可选）')

    args = parser.parse_args()

    # 初始化配置
    config = InstructPix2PixConfig()

    if args.mode == 'train':
        # 训练模式
        data_processor = DataProcessor(config)
        data_loader = data_processor.load_and_preprocess_data()

        model_manager = ModelManager(config)
        model_manager.load_models()

        training_utils = TrainingUtils(config, model_manager)

        trainer = Trainer(config, model_manager, training_utils)
        trainer.train(data_loader)

    elif args.mode == 'test':
        # 测试模式
        tester = Tester(config)
        tester.load_test_dataset()
        tester.test_model(args.model_path)


if __name__ == '__main__':
    main()
