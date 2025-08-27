# -*- coding: utf-8 -*-
# @project: LargeDemo
# @filename: ch2_introduction.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/8/25 21:52
# https://chat.deepseek.com/a/chat/s/ead5b442-d2f9-4279-9e2f-a4b6ea3d46cc

import torch
from diffusers import DiffusionPipeline, DDPMScheduler, AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTokenizer, CLIPTextModel
from matplotlib import pyplot as plt
import PIL.Image
import numpy as np
import torchvision
import argparse

# 全局模型变量（可选，也可以在每个函数内加载）
checkpoint = 'CompVis/stable-diffusion-v1-4'


def test_pipeline():
    """测试完整Diffusion流程生成图像"""
    pipeline = DiffusionPipeline.from_pretrained('runwayml/stable-diffusion-v1-5')
    pipeline.to('cuda' if torch.cuda.is_available() else 'cpu')

    # 测试画图
    image = pipeline('An image of a squirrel in Picasso style').images[0]

    # 保存图片
    image.save('save/sample.jpg')

    return image


def test_tokenizer():
    """测试CLIP Tokenizer"""
    tokenizer = CLIPTokenizer.from_pretrained(checkpoint, subfolder='tokenizer')

    text = 'the quick brown fox jumps over a lazy dog'
    out = tokenizer(text,
                    max_length=13,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt')

    for k, v in out.items():
        print(k, v)

    print(tokenizer.decode(out['input_ids'][0]))


def test_scheduler():
    """测试DDPMScheduler噪声调度器"""
    scheduler = DDPMScheduler.from_pretrained(checkpoint, subfolder='scheduler')

    plt.figure(figsize=(20, 10))

    # 加载测试图片
    image = PIL.Image.open('测试图片.jpeg').resize((512, 512))
    image = torch.FloatTensor(np.array(image)) / 255.0

    # 随机噪声
    noise = torch.randn(image.shape)

    for i, step in enumerate([0, 100, 500]):
        # 添加噪声
        image_noise = scheduler.add_noise(image, noise,
                                          torch.LongTensor([step])).clip(0, 1)

        # 保存图片
        torchvision.utils.save_image(image_noise.permute(2, 0, 1),
                                     'save/%d.jpg' % step)

        # 显示图片
        plt.subplot(1, 3, i + 1)
        plt.imshow(image_noise)
        plt.axis('off')
        plt.title('step %d' % step)

    plt.show()


def test_encoder():
    """测试CLIP文本编码器"""
    encoder = CLIPTextModel.from_pretrained(checkpoint, subfolder='text_encoder')

    input_ids = torch.LongTensor(
        [[49406, 320, 1746, 537, 1449, 14115, 593, 1237, 3095, 49407, 49407]])

    out_encoder = encoder(input_ids)[0]

    print(out_encoder.shape, out_encoder.dtype)


def test_vae():
    """测试VAE编码器和解码器"""
    vae = AutoencoderKL.from_pretrained(checkpoint, subfolder='vae')

    pixel_values = torch.randn(1, 3, 512, 512).clip(min=-1.0, max=1.0)

    # 根据数据计算出正态分布的mean和std
    out_vae = vae.encode(pixel_values).latent_dist

    # [1, 4, 64, 64], [1, 4, 64, 64]
    print(out_vae.mean.shape, out_vae.std.shape)

    # 根据mean和std采样
    # 等价写法:sample = out_vae.mean + out_vae.std * torch.randn(1, 4, 64, 64)
    # [1, 4, 64, 64]
    out_vae = out_vae.sample()

    # 乘以系数,0.18215 = vae.config.scaling_factor
    out_vae = out_vae * 0.18215

    # [1, 4, 64, 64]
    print(out_vae.shape, out_vae.dtype)

    # 特征图解码成图片
    out_vae = vae.decode(torch.randn(1, 4, 64, 64)).sample

    # [1, 3, 512, 512]
    print(out_vae.shape, out_vae.dtype)


def test_unet():
    """测试U-Net模型"""
    scheduler = DDPMScheduler.from_pretrained(checkpoint, subfolder='scheduler')
    unet = UNet2DConditionModel.from_pretrained(checkpoint, subfolder='unet')

    # 虚拟Encoder和VAE的输出
    out_encoder = torch.randn(1, 77, 768)
    out_vae = torch.randn(1, 4, 64, 64)

    # 随机噪声,也就是U-Net计算的目标
    noise = torch.randn_like(out_vae)

    # 这个数字实际应该是随机得到的,值域0~999
    noise_step = torch.LongTensor([500])

    # 往VAE输出的特征图中添加噪声
    out_vae_noise = scheduler.add_noise(out_vae, noise, noise_step)

    # U-Net的目标是把添加的噪声计算出来
    out_unet = unet(out_vae_noise, noise_step, out_encoder).sample

    print(out_unet.shape, out_unet.dtype)


def main():
    """主函数，根据命令行参数选择运行哪个测试"""
    parser = argparse.ArgumentParser(description='测试Stable Diffusion组件')
    parser.add_argument('--test', type=str, choices=[
        'pipeline', 'tokenizer', 'scheduler', 'encoder', 'vae', 'unet', 'all'
    ], default='all', help='选择要运行的测试')

    args = parser.parse_args()

    test_mapping = {
        'pipeline': test_pipeline,
        'tokenizer': test_tokenizer,
        'scheduler': test_scheduler,
        'encoder': test_encoder,
        'vae': test_vae,
        'unet': test_unet
    }

    if args.test == 'all':
        for test_name, test_func in test_mapping.items():
            print(f"\n=== 运行 {test_name} 测试 ===")
            test_func()
    else:
        print(f"\n=== 运行 {args.test} 测试 ===")
        test_mapping[args.test]()


if __name__ == "__main__":
    main()
