# -*- coding: utf-8 -*-
# @project: LargeDemo
# @filename: chapter5_finetune_and_guidance.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/8/25 14:59
# https://chat.deepseek.com/a/chat/s/e3dd9198-4aa5-4f82-9d7a-b95c58f021b3

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from datasets import load_dataset
from diffusers import DDIMScheduler, DDPMPipeline, DDPMScheduler, UNet2DModel
from matplotlib import pyplot as plt
from PIL import Image, ImageColor
from torchvision import transforms
from tqdm.auto import tqdm
import open_clip
import gradio as gr
from torch import nn
from torch.utils.data import DataLoader
import os
from huggingface_hub import notebook_login, HfApi, ModelCard, create_repo, get_full_repo_name


class DiffusionModel:
    def __init__(self):
        self.device = self.setup_device()
        self.image_pipe = None
        self.scheduler = None

    def setup_device(self):
        """设置计算设备"""
        device = (
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )
        print(f"Using device: {device}")
        return device

    def load_pretrained_model(self, model_name="google/ddpm-celebahq-256"):
        """加载预训练模型"""
        self.image_pipe = DDPMPipeline.from_pretrained(model_name)
        self.image_pipe.to(self.device)
        return self.image_pipe

    def setup_scheduler(self, num_inference_steps=40):
        """设置调度器"""
        self.scheduler = DDIMScheduler.from_pretrained("google/ddpm-celebahq-256")
        self.scheduler.set_timesteps(num_inference_steps=num_inference_steps)
        return self.scheduler

    def generate_images(self, num_images=1):
        """生成图像"""
        if self.image_pipe is None:
            self.load_pretrained_model()

        images = self.image_pipe(num_inference_steps=40).images
        return images

    def custom_sampling_loop(self, batch_size=4):
        """自定义采样循环"""
        if self.scheduler is None:
            self.setup_scheduler()

        # 随机起点
        x = torch.randn(batch_size, 3, 256, 256).to(self.device)

        # 循环采样时间步
        for i, t in tqdm(enumerate(self.scheduler.timesteps)):
            # 准备模型输入
            model_input = self.scheduler.scale_model_input(x, t)

            # 获取预测
            with torch.no_grad():
                noise_pred = self.image_pipe.unet(model_input, t)["sample"]

            # 使用调度器计算更新后的样本
            scheduler_output = self.scheduler.step(noise_pred, t, x)

            # 更新x
            x = scheduler_output.prev_sample

            # 偶尔显示x和预测的去噪图像
            if i % 10 == 0 or i == len(self.scheduler.timesteps) - 1:
                fig, axs = plt.subplots(1, 2, figsize=(12, 5))

                grid = torchvision.utils.make_grid(x, nrow=4).permute(1, 2, 0)
                axs[0].imshow(grid.cpu().clip(-1, 1) * 0.5 + 0.5)
                axs[0].set_title(f"Current x (step {i})")

                if hasattr(scheduler_output, 'pred_original_sample'):
                    pred_x0 = scheduler_output.pred_original_sample
                    grid = torchvision.utils.make_grid(pred_x0, nrow=4).permute(1, 2, 0)
                    axs[1].imshow(grid.cpu().clip(-1, 1) * 0.5 + 0.5)
                    axs[1].set_title(f"Predicted denoised images (step {i})")

                plt.show()

        return x


class FineTuner:
    def __init__(self, device):
        self.device = device
        self.image_pipe = None
        self.train_dataloader = None

    def prepare_dataset(self, dataset_name="huggan/smithsonian_butterflies_subset",
                        image_size=256, batch_size=4):
        """准备数据集"""
        dataset = load_dataset(dataset_name, split="train")

        preprocess = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        def transform(examples):
            images = [preprocess(image.convert("RGB")) for image in examples["image"]]
            return {"images": images}

        dataset.set_transform(transform)

        self.train_dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        # 预览批次
        print("Previewing batch:")
        batch = next(iter(self.train_dataloader))
        grid = torchvision.utils.make_grid(batch["images"], nrow=4)
        plt.imshow(grid.permute(1, 2, 0).cpu().clip(-1, 1) * 0.5 + 0.5)
        plt.show()

        return self.train_dataloader

    def load_model_for_finetuning(self, model_name="google/ddpm-celebahq-256"):
        """加载用于微调的模型"""
        self.image_pipe = DDPMPipeline.from_pretrained(model_name)
        self.image_pipe.to(self.device)
        return self.image_pipe

    def finetune(self, num_epochs=2, lr=1e-5, grad_accumulation_steps=2):
        """微调模型"""
        if self.image_pipe is None:
            self.load_model_for_finetuning()

        if self.train_dataloader is None:
            self.prepare_dataset()

        optimizer = torch.optim.AdamW(self.image_pipe.unet.parameters(), lr=lr)
        losses = []

        for epoch in range(num_epochs):
            for step, batch in tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader)):
                clean_images = batch["images"].to(self.device)
                # 采样要添加到图像的噪声
                noise = torch.randn(clean_images.shape).to(clean_images.device)
                bs = clean_images.shape[0]

                # 为每个图像采样一个随机时间步
                timesteps = torch.randint(
                    0,
                    self.image_pipe.scheduler.num_train_timesteps,
                    (bs,),
                    device=clean_images.device,
                ).long()

                # 根据每个时间步的噪声幅度向干净图像添加噪声
                # (这是前向扩散过程)
                noisy_images = self.image_pipe.scheduler.add_noise(clean_images, noise, timesteps)

                # 获取噪声的模型预测
                noise_pred = self.image_pipe.unet(noisy_images, timesteps, return_dict=False)[0]

                # 将预测与实际噪声进行比较：
                loss = F.mse_loss(noise_pred, noise)

                # 存储以供后续绘图
                losses.append(loss.item())

                # 基于此损失使用优化器更新模型参数
                loss.backward()

                # 梯度累积：
                if (step + 1) % grad_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            print(
                f"Epoch {epoch} average loss: {sum(losses[-len(self.train_dataloader):]) / len(self.train_dataloader)}"
            )

        # 绘制损失曲线：
        plt.plot(losses)
        plt.title("Training Loss")
        plt.show()

        return losses

    def save_model(self, save_path="my-finetuned-model"):
        """保存模型"""
        if self.image_pipe is not None:
            self.image_pipe.save_pretrained(save_path)
            print(f"Model saved to {save_path}")

    def upload_to_hub(self, model_name, local_folder_name, description):
        """上传模型到Hugging Face Hub"""
        hub_model_id = get_full_repo_name(model_name)
        create_repo(hub_model_id)
        api = HfApi()

        # 上传文件
        api.upload_folder(
            folder_path=f"{local_folder_name}/scheduler", path_in_repo="", repo_id=hub_model_id
        )
        api.upload_folder(
            folder_path=f"{local_folder_name}/unet", path_in_repo="", repo_id=hub_model_id
        )
        api.upload_file(
            path_or_fileobj=f"{local_folder_name}/model_index.json",
            path_in_repo="model_index.json",
            repo_id=hub_model_id,
        )

        # 添加模型卡片
        content = f"""---
license: mit
tags:
- pytorch
- diffusers
- unconditional-image-generation
- diffusion-models-class
---

# Example Fine-Tuned Model for Unit 2 of the [Diffusion Models Class 🧨](https://github.com/huggingface/diffusion-models-class)

{description}

## Usage

```python
from diffusers import DDPMPipeline

pipeline = DDPMPipeline.from_pretrained('{hub_model_id}')
image = pipeline().images[0]
image
```
"""

        card = ModelCard(content)
        card.push_to_hub(hub_model_id)
        print(f"Model uploaded to https://huggingface.co/{hub_model_id}")


class GuidedSampling:
    def __init__(self, device):
        self.device = device
        self.image_pipe = None
        self.scheduler = None

    def load_pretrained_pipeline(self, pipeline_name="johnowhitaker/sd-class-wikiart-from-bedrooms"):
        """加载预训练管道"""
        self.image_pipe = DDPMPipeline.from_pretrained(pipeline_name).to(self.device)
        self.scheduler = DDIMScheduler.from_pretrained(pipeline_name)
        return self.image_pipe, self.scheduler

    def color_loss(self, images, target_color=(0.1, 0.9, 0.5)):
        """颜色损失函数"""
        target = (
                torch.tensor(target_color).to(images.device) * 2 - 1
        )  # 将目标颜色映射到(-1, 1)
        target = target[
                 None, :, None, None
                 ]  # 调整形状以与图像(b, c, h, w)一起使用
        error = torch.abs(
            images - target
        ).mean()  # 图像像素与目标颜色之间的平均绝对差
        return error

    def guided_sampling_v1(self, guidance_loss_scale=40, num_images=8):
        """引导采样方法1"""
        if self.scheduler is None:
            self.load_pretrained_pipeline()

        self.scheduler.set_timesteps(50)
        x = torch.randn(num_images, 3, 256, 256).to(self.device)

        for i, t in tqdm(enumerate(self.scheduler.timesteps)):
            # 准备模型输入
            model_input = self.scheduler.scale_model_input(x, t)

            # 预测噪声残差
            with torch.no_grad():
                noise_pred = self.image_pipe.unet(model_input, t)["sample"]

            # 设置x.requires_grad为True
            x = x.detach().requires_grad_()

            # 获取预测的x0
            x0 = self.scheduler.step(noise_pred, t, x).pred_original_sample

            # 计算损失
            loss = self.color_loss(x0) * guidance_loss_scale
            if i % 10 == 0:
                print(i, "loss:", loss.item())

            # 获取梯度
            cond_grad = -torch.autograd.grad(loss, x)[0]

            # 基于此梯度修改x
            x = x.detach() + cond_grad

            # 现在使用调度器进行步骤
            x = self.scheduler.step(noise_pred, t, x).prev_sample

        # 查看输出
        grid = torchvision.utils.make_grid(x, nrow=4)
        im = grid.permute(1, 2, 0).cpu().clip(-1, 1) * 0.5 + 0.5
        return Image.fromarray(np.array(im * 255).astype(np.uint8))

    def guided_sampling_v2(self, guidance_loss_scale=40, num_images=4):
        """引导采样方法2"""
        if self.scheduler is None:
            self.load_pretrained_pipeline()

        self.scheduler.set_timesteps(50)
        x = torch.randn(num_images, 3, 256, 256).to(self.device)

        for i, t in tqdm(enumerate(self.scheduler.timesteps)):
            # 在模型前向传递之前设置requires_grad
            x = x.detach().requires_grad_()
            model_input = self.scheduler.scale_model_input(x, t)

            # 预测（这次有梯度）
            noise_pred = self.image_pipe.unet(model_input, t)["sample"]

            # 获取预测的x0：
            x0 = self.scheduler.step(noise_pred, t, x).pred_original_sample

            # 计算损失
            loss = self.color_loss(x0) * guidance_loss_scale
            if i % 10 == 0:
                print(i, "loss:", loss.item())

            # 获取梯度
            cond_grad = -torch.autograd.grad(loss, x)[0]

            # 基于此梯度修改x
            x = x.detach() + cond_grad

            # 现在使用调度器进行步骤
            x = self.scheduler.step(noise_pred, t, x).prev_sample

        grid = torchvision.utils.make_grid(x, nrow=4)
        im = grid.permute(1, 2, 0).cpu().clip(-1, 1) * 0.5 + 0.5
        return Image.fromarray(np.array(im * 255).astype(np.uint8))


class CLIPGuidance:
    def __init__(self, device):
        self.device = device
        self.image_pipe = None
        self.scheduler = None
        self.clip_model = None
        self.tfms = None

    def setup_clip_model(self):
        """设置CLIP模型"""
        self.clip_model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        self.clip_model.to(self.device)

        # 转换以调整图像大小和增强+标准化以匹配CLIP的训练数据
        self.tfms = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(224),  # 每次随机裁剪
            torchvision.transforms.RandomAffine(5),  # 一种可能的随机增强：倾斜图像
            torchvision.transforms.RandomHorizontalFlip(),  # 可以添加其他增强
            torchvision.transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ])

        return self.clip_model, self.tfms

    def clip_loss(self, image, text_features):
        """CLIP损失函数"""
        image_features = self.clip_model.encode_image(self.tfms(image))
        input_normed = torch.nn.functional.normalize(image_features.unsqueeze(1), dim=2)
        embed_normed = torch.nn.functional.normalize(text_features.unsqueeze(0), dim=2)
        dists = (
            input_normed.sub(embed_normed).norm(dim=2).div(2).arcsin().pow(2).mul(2)
        )  # 平方大圆距离
        return dists.mean()

    def clip_guided_sampling(self, prompt="Red Rose (still life), red flower painting",
                             guidance_scale=8, n_cuts=4, num_inference_steps=50):
        """CLIP引导采样"""
        if self.clip_model is None:
            self.setup_clip_model()

        if self.image_pipe is None:
            self.image_pipe = DDPMPipeline.from_pretrained("google/ddpm-celebahq-256").to(self.device)
            self.scheduler = DDIMScheduler.from_pretrained("google/ddpm-celebahq-256")

        # 更多步骤->更多时间让引导生效
        self.scheduler.set_timesteps(num_inference_steps)

        # 使用CLIP嵌入提示作为目标
        text = open_clip.tokenize([prompt]).to(self.device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            text_features = self.clip_model.encode_text(text)

        x = torch.randn(4, 3, 256, 256).to(self.device)

        for i, t in tqdm(enumerate(self.scheduler.timesteps)):
            model_input = self.scheduler.scale_model_input(x, t)

            # 预测噪声残差
            with torch.no_grad():
                noise_pred = self.image_pipe.unet(model_input, t)["sample"]

            cond_grad = 0

            for cut in range(n_cuts):
                # 在x上设置requires grad
                x = x.detach().requires_grad_()

                # 获取预测的x0：
                x0 = self.scheduler.step(noise_pred, t, x).pred_original_sample

                # 计算损失
                loss = self.clip_loss(x0, text_features) * guidance_scale

                # 获取梯度（按n_cuts缩放，因为我们想要平均值）
                cond_grad -= torch.autograd.grad(loss, x)[0] / n_cuts

            if i % 25 == 0:
                print("Step:", i, ", Guidance loss:", loss.item())

            # 基于此梯度修改x
            alpha_bar = self.scheduler.alphas_cumprod[i]
            x = (
                    x.detach() + cond_grad * alpha_bar.sqrt()
            )  # 注意这里的额外缩放因子！

            # 现在使用调度器进行步骤
            x = self.scheduler.step(noise_pred, t, x).prev_sample

        grid = torchvision.utils.make_grid(x.detach(), nrow=4)
        im = grid.permute(1, 2, 0).cpu().clip(-1, 1) * 0.5 + 0.5
        return Image.fromarray(np.array(im * 255).astype(np.uint8))


class GradioInterface:
    def __init__(self, device):
        self.device = device
        self.scheduler = None
        self.image_pipe = None

    def setup_models(self):
        """设置模型"""
        if self.image_pipe is None:
            self.image_pipe = DDPMPipeline.from_pretrained("google/ddpm-celebahq-256").to(self.device)
            self.scheduler = DDIMScheduler.from_pretrained("google/ddpm-celebahq-256")
            self.scheduler.set_timesteps(40)

    def color_loss(self, images, target_color=(0.1, 0.9, 0.5)):
        """颜色损失函数"""
        target = (
                torch.tensor(target_color).to(images.device) * 2 - 1
        )
        target = target[None, :, None, None]
        error = torch.abs(images - target).mean()
        return error

    def generate(self, color, guidance_loss_scale):
        """生成函数"""
        self.setup_models()

        target_color = ImageColor.getcolor(color, "RGB")  # 目标颜色为RGB
        target_color = [a / 255 for a in target_color]  # 从(0, 255)重新缩放到(0, 1)
        x = torch.randn(1, 3, 256, 256).to(self.device)

        for i, t in tqdm(enumerate(self.scheduler.timesteps)):
            model_input = self.scheduler.scale_model_input(x, t)
            with torch.no_grad():
                noise_pred = self.image_pipe.unet(model_input, t)["sample"]

            x = x.detach().requires_grad_()
            x0 = self.scheduler.step(noise_pred, t, x).pred_original_sample

            loss = self.color_loss(x0, target_color) * guidance_loss_scale
            cond_grad = -torch.autograd.grad(loss, x)[0]

            x = x.detach() + cond_grad
            x = self.scheduler.step(noise_pred, t, x).prev_sample

        grid = torchvision.utils.make_grid(x, nrow=4)
        im = grid.permute(1, 2, 0).cpu().clip(-1, 1) * 0.5 + 0.5
        im = Image.fromarray(np.array(im * 255).astype(np.uint8))
        im.save("test.jpeg")
        return im

    def launch_interface(self):
        """启动Gradio界面"""
        inputs = [
            gr.ColorPicker(label="color", value="55FFAA"),
            gr.Slider(label="guidance_scale", minimum=0, maximum=30, value=3),
        ]
        outputs = gr.Image(label="result")

        # 最小界面
        demo = gr.Interface(
            fn=self.generate,
            inputs=inputs,
            outputs=outputs,
            examples=[
                ["#BB2266", 3],
                ["#44CCAA", 5],
            ],
        )
        demo.launch(debug=True)


class ClassConditionedModel:
    def __init__(self, device):
        self.device = device
        self.net = None
        self.noise_scheduler = None

    def prepare_mnist_dataset(self, batch_size=8):
        """准备MNIST数据集"""
        dataset = torchvision.datasets.MNIST(
            root="mnist/",
            train=True,
            download=True,
            transform=torchvision.transforms.ToTensor()
        )

        train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # 查看一些示例
        x, y = next(iter(train_dataloader))
        print('Input shape:', x.shape)
        print('Labels:', y)
        plt.imshow(torchvision.utils.make_grid(x)[0], cmap='Greys')
        plt.show()

        return train_dataloader

    def create_class_conditioned_unet(self, num_classes=10, class_emb_size=4):
        """创建类别条件UNet"""

        class ClassConditionedUnet(nn.Module):
            def __init__(self, num_classes=10, class_emb_size=4):
                super().__init__()

                # 嵌入层将类别标签映射到大小为class_emb_size的向量
                self.class_emb = nn.Embedding(num_classes, class_emb_size)

                # 无条件UNet，具有额外的输入通道以接受条件信息（类别嵌入）
                self.model = UNet2DModel(
                    sample_size=28,
                    in_channels=1 + class_emb_size,
                    out_channels=1,
                    layers_per_block=2,
                    block_out_channels=(32, 64, 64),
                    down_block_types=(
                        "DownBlock2D",
                        "AttnDownBlock2D",
                        "AttnDownBlock2D",
                    ),
                    up_block_types=(
                        "AttnUpBlock2D",
                        "AttnUpBlock2D",
                        "UpBlock2D",
                    ),
                )

            def forward(self, x, t, class_labels):
                bs, ch, w, h = x.shape

                # 类别条件，形状正确，可以作为额外的输入通道添加
                class_cond = self.class_emb(class_labels)
                class_cond = class_cond.view(bs, class_cond.shape[1], 1, 1).expand(bs, class_cond.shape[1], w, h)

                # 网络输入现在是x和类别条件沿着维度1连接在一起
                net_input = torch.cat((x, class_cond), 1)

                # 将其与时间步一起馈送到unet并返回预测
                return self.model(net_input, t).sample

        self.net = ClassConditionedUnet(num_classes, class_emb_size).to(self.device)
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2')

        return self.net, self.noise_scheduler

    def train(self, n_epochs=10, batch_size=128, lr=1e-3):
        """训练模型"""
        if self.net is None:
            self.create_class_conditioned_unet()

        train_dataloader = self.prepare_mnist_dataset(batch_size)

        loss_fn = nn.MSELoss()
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        losses = []

        for epoch in range(n_epochs):
            for x, y in tqdm(train_dataloader):
                x = x.to(self.device) * 2 - 1
                y = y.to(self.device)
                noise = torch.randn_like(x)
                timesteps = torch.randint(0, 999, (x.shape[0],)).long().to(self.device)
                noisy_x = self.noise_scheduler.add_noise(x, noise, timesteps)

                pred = self.net(noisy_x, timesteps, y)
                loss = loss_fn(pred, noise)

                opt.zero_grad()
                loss.backward()
                opt.step()

                losses.append(loss.item())

            avg_loss = sum(losses[-100:]) / 100 if len(losses) >= 100 else sum(losses) / len(losses)
            print(f'Finished epoch {epoch}. Average of the last 100 loss values: {avg_loss:05f}')

        plt.plot(losses)
        plt.title('Training Loss')
        plt.show()

        return losses

    def sample_different_digits(self):
        """采样不同的数字"""
        if self.net is None or self.noise_scheduler is None:
            print("Please train or load a model first")
            return

        x = torch.randn(80, 1, 28, 28).to(self.device)
        y = torch.tensor([[i] * 8 for i in range(10)]).flatten().to(self.device)

        for i, t in tqdm(enumerate(self.noise_scheduler.timesteps)):
            with torch.no_grad():
                residual = self.net(x, t, y)

            x = self.noise_scheduler.step(residual, t, x).prev_sample

        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        ax.imshow(torchvision.utils.make_grid(x.detach().cpu().clip(-1, 1), nrow=8)[0], cmap='Greys')
        plt.show()


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='Diffusion Models Chapter 5')
    parser.add_argument('--mode', type=str, default='generate',
                        choices=['generate', 'finetune', 'guide', 'clip_guide', 'gradio', 'class_conditioned'],
                        help='Mode to run: generate, finetune, guide, clip_guide, gradio, class_conditioned')
    parser.add_argument('--model', type=str, default='google/ddpm-celebahq-256', help='Model name')
    parser.add_argument('--dataset', type=str, default='huggan/smithsonian_butterflies_subset', help='Dataset name')
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs for finetuning')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')

    args = parser.parse_args()

    # 设置设备
    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    if args.mode == 'generate':
        # 生成图像
        dm = DiffusionModel()
        dm.load_pretrained_model(args.model)
        images = dm.generate_images()
        images[0].show()

    elif args.mode == 'finetune':
        # 微调模型
        ft = FineTuner(device)
        ft.prepare_dataset(args.dataset, batch_size=args.batch_size)
        ft.load_model_for_finetuning(args.model)
        losses = ft.finetune(num_epochs=args.epochs, lr=args.lr)
        ft.save_model()

    elif args.mode == 'guide':
        # 引导采样
        gs = GuidedSampling(device)
        result = gs.guided_sampling_v1()
        result.show()

    elif args.mode == 'clip_guide':
        # CLIP引导采样
        cg = CLIPGuidance(device)
        result = cg.clip_guided_sampling()
        result.show()

    elif args.mode == 'gradio':
        # Gradio界面
        gi = GradioInterface(device)
        gi.launch_interface()

    elif args.mode == 'class_conditioned':
        # 类别条件模型
        ccm = ClassConditionedModel(device)
        ccm.create_class_conditioned_unet()
        losses = ccm.train(n_epochs=args.epochs, batch_size=args.batch_size)
        ccm.sample_different_digits()


if __name__ == "__main__":
    main()


