# -*- coding: utf-8 -*-
# @project: LargeDemo
# @filename: chapter8_audio_diffusion.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/8/25 15:01
# https://chat.deepseek.com/a/chat/s/c30ae953-d3e0-4823-9d0e-7c1227e47472

import torch
import random
import numpy as np
import torch.nn.functional as F
from tqdm.auto import tqdm
from IPython.display import Audio, display
from matplotlib import pyplot as plt
from diffusers import DiffusionPipeline
from torchaudio import transforms as AT
from torchvision import transforms as IT
from datasets import load_dataset
from huggingface_hub import get_full_repo_name, HfApi, create_repo, ModelCard
import argparse
import os


class AudioDiffusionTrainer:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pipe = None
        self.sample_rate_pipeline = None
        self.resampler = None
        self.to_t = IT.ToTensor()

    def load_pretrained_pipeline(self, model_name="teticio/audio-diffusion-instrumental-hiphop-256"):
        """加载预训练的音频扩散管道"""
        print(f"Loading pretrained pipeline: {model_name}")
        self.pipe = DiffusionPipeline.from_pretrained(model_name).to(self.device)
        self.sample_rate_pipeline = self.pipe.mel.get_sample_rate()
        return self.pipe

    def generate_sample(self, noise_shape=None):
        """生成音频样本"""
        if self.pipe is None:
            self.load_pretrained_pipeline()

        if noise_shape is not None:
            noise = torch.randn(noise_shape).to(self.device)
            output = self.pipe(noise=noise)
        else:
            output = self.pipe()

        # 显示结果
        display(output.images[0])
        display(Audio(output.audios[0], rate=self.pipe.mel.get_sample_rate()))

        return output

    def inspect_audio_output(self, output):
        """检查音频输出"""
        print("Audio array shape:", output.audios[0].shape)
        print("Image size:", output.images[0].size)

        # 计算并显示频谱图
        spec_transform = AT.Spectrogram(power=2)
        spectrogram = spec_transform(torch.tensor(output.audios[0]))
        print("Spectrogram range:", spectrogram.min().item(), spectrogram.max().item())

        log_spectrogram = spectrogram.log()
        plt.imshow(log_spectrogram[0], cmap='gray')
        plt.show()

        return spectrogram

    def load_and_explore_dataset(self, dataset_name='lewtun/music_genres'):
        """加载和探索数据集"""
        print(f"Loading dataset: {dataset_name}")
        dataset = load_dataset(dataset_name, split='train')
        print(f"Dataset size: {len(dataset)}")

        # 显示不同流派的数量
        genres = list(set(dataset['genre']))
        for g in genres:
            count = sum(x == g for x in dataset['genre'])
            print(f"{g}: {count}")

        return dataset

    def preview_dataset_audio(self, dataset, index=0):
        """预览数据集中的音频样本"""
        audio_array = dataset[index]['audio']['array']
        sample_rate = dataset[index]['audio']['sampling_rate']

        print(f'Audio array shape: {audio_array.shape}')
        print(f'Sample rate: {sample_rate}')
        display(Audio(audio_array, rate=sample_rate))

        return audio_array, sample_rate

    def audio_to_spectrogram_image(self, audio_array, sample_rate):
        """将音频转换为频谱图图像"""
        if self.resampler is None or self.resampler.orig_freq != sample_rate:
            self.resampler = AT.Resample(sample_rate, self.sample_rate_pipeline, dtype=torch.float32)

        audio_tensor = torch.tensor(audio_array).to(torch.float32)
        audio_tensor = self.resampler(audio_tensor)

        self.pipe.mel.load_audio(raw_audio=np.array(audio_tensor))
        num_slices = self.pipe.mel.get_number_of_slices()

        # 随机选择一个切片（排除最后一个短切片）
        slice_idx = random.randint(0, num_slices - 2) if num_slices > 1 else 0
        im = self.pipe.mel.audio_slice_to_image(slice_idx)

        return im

    def create_data_loader(self, dataset, genre, batch_size=4):
        """创建指定流派的数据加载器"""

        def to_image(audio_array):
            return self.audio_to_spectrogram_image(audio_array, dataset[0]['audio']['sampling_rate'])

        def collate_fn(examples):
            # 转换为图像 -> 转换为张量 -> 缩放到(-1, 1) -> 堆叠成批次
            audio_ims = [self.to_t(to_image(x['audio']['array'])) * 2 - 1 for x in examples]
            return torch.stack(audio_ims)

        # 创建指定流派的数据集
        indexes = [i for i, g in enumerate(dataset['genre']) if g == genre]
        filtered_dataset = dataset.select(indexes)

        dl = torch.utils.data.DataLoader(
            filtered_dataset.shuffle(),
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=True
        )

        return dl

    def train_model(self, data_loader, epochs=3, lr=1e-4):
        """训练模型"""
        if self.pipe is None:
            self.load_pretrained_pipeline()

        self.pipe.unet.train()
        self.pipe.scheduler.set_timesteps(1000)

        optimizer = torch.optim.AdamW(self.pipe.unet.parameters(), lr=lr)

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            for step, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
                # 准备输入图像
                clean_images = batch.to(self.device)
                bs = clean_images.shape[0]

                # 为每个图像采样随机时间步
                timesteps = torch.randint(
                    0, self.pipe.scheduler.num_train_timesteps, (bs,), device=clean_images.device
                ).long()

                # 根据每个时间步的噪声幅度向干净图像添加噪声
                noise = torch.randn(clean_images.shape).to(clean_images.device)
                noisy_images = self.pipe.scheduler.add_noise(clean_images, noise, timesteps)

                # 获取模型预测
                noise_pred = self.pipe.unet(noisy_images, timesteps, return_dict=False)[0]

                # 计算损失
                loss = F.mse_loss(noise_pred, noise)
                loss.backward()

                # 使用优化器更新模型参数
                optimizer.step()
                optimizer.zero_grad()

                if step % 50 == 0:
                    print(f"Step {step}, Loss: {loss.item():.4f}")

    def save_and_upload_model(self, model_name, chosen_genre, hub_model_id=None):
        """保存并上传模型到Hugging Face Hub"""
        if hub_model_id is None:
            hub_model_id = get_full_repo_name(model_name)

        # 保存管道到本地
        print(f"Saving model to {model_name}")
        self.pipe.save_pretrained(model_name)

        # 检查文件夹内容
        print("Model contents:")
        print(os.listdir(model_name))

        # 创建仓库
        try:
            create_repo(hub_model_id)
            print(f"Created repository: {hub_model_id}")
        except Exception as e:
            print(f"Repository may already exist: {e}")

        # 上传文件
        api = HfApi()
        api.upload_folder(
            folder_path=f"{model_name}/scheduler",
            path_in_repo="scheduler",
            repo_id=hub_model_id
        )
        api.upload_folder(
            folder_path=f"{model_name}/mel",
            path_in_repo="mel",
            repo_id=hub_model_id
        )
        api.upload_folder(
            folder_path=f"{model_name}/unet",
            path_in_repo="unet",
            repo_id=hub_model_id
        )
        api.upload_file(
            path_or_fileobj=f"{model_name}/model_index.json",
            path_in_repo="model_index.json",
            repo_id=hub_model_id,
        )

        # 推送模型卡片
        content = f"""
---
license: mit
tags:
- pytorch
- diffusers
- unconditional-audio-generation
- diffusion-models-class
---

# Model Card for Audio Diffusion Model

This model is a diffusion model for unconditional audio generation of music in the genre {chosen_genre}

## Usage

```python
from IPython.display import Audio
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("{hub_model_id}")
output = pipe()
display(output.images[0])
display(Audio(output.audios[0], rate=pipe.mel.get_sample_rate()))
```
"""

        card = ModelCard(content)
        card.push_to_hub(hub_model_id)
        print(f"Model uploaded to: {hub_model_id}")


def main():
    parser = argparse.ArgumentParser(description='Audio Diffusion Training')
    parser.add_argument('--mode', type=str, default='generate',
                        choices=['generate', 'explore', 'train', 'upload'],
                        help='运行模式: generate(生成样本), explore(探索数据), train(训练模型), upload(上传模型)')
    parser.add_argument('--model', type=str, default='teticio/audio-diffusion-instrumental-hiphop-256',
                        help='预训练模型名称')
    parser.add_argument('--genre', type=str, default='Electronic',
                        help='要训练的音乐流派')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='训练批量大小')
    parser.add_argument('--epochs', type=int, default=3,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='学习率')
    parser.add_argument('--output_model', type=str, default='audio-diffusion-electronic',
                        help='输出模型名称')

    args = parser.parse_args()

    trainer = AudioDiffusionTrainer()

    if args.mode == 'generate':
        # 生成样本
        trainer.load_pretrained_pipeline(args.model)
        output = trainer.generate_sample()
        trainer.inspect_audio_output(output)

        # 生成长样本
        print("\nGenerating longer sample...")
        noise_shape = (1, 1, trainer.pipe.unet.sample_size[0], trainer.pipe.unet.sample_size[1] * 4)
        output_long = trainer.generate_sample(noise_shape=noise_shape)

    elif args.mode == 'explore':
        # 探索数据集
        dataset = trainer.load_and_explore_dataset()
        trainer.preview_dataset_audio(dataset)

        # 显示音频的频谱图
        audio_array, sample_rate = trainer.preview_dataset_audio(dataset, 0)
        trainer.load_pretrained_pipeline(args.model)
        spectrogram_img = trainer.audio_to_spectrogram_image(audio_array, sample_rate)
        display(spectrogram_img)

    elif args.mode == 'train':
        # 训练模型
        trainer.load_pretrained_pipeline(args.model)
        dataset = trainer.load_and_explore_dataset()
        data_loader = trainer.create_data_loader(dataset, args.genre, args.batch_size)

        # 测试数据加载器
        batch = next(iter(data_loader))
        print(f"Batch shape: {batch.shape}")

        # 训练模型
        trainer.train_model(data_loader, epochs=args.epochs, lr=args.lr)

        # 测试训练后的模型
        print("Testing trained model...")
        output = trainer.generate_sample()

    elif args.mode == 'upload':
        # 上传模型到Hugging Face Hub
        trainer.load_pretrained_pipeline(args.model)
        trainer.save_and_upload_model(args.output_model, args.genre)


if __name__ == "__main__":
    main()

