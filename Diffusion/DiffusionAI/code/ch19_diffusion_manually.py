# -*- coding: utf-8 -*-
# @project: LargeDemo
# @filename: ch19_diffusion_manually.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/8/26 10:27
# https://chat.deepseek.com/a/chat/s/0c9878b6-7349-4b86-8635-b2fb686e7ab5
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from transformers import CLIPTextModel, PreTrainedModel, PretrainedConfig
from diffusers import DiffusionPipeline, AutoencoderKL, UNet2DConditionModel
from datasets import load_dataset
from matplotlib import pyplot as plt
import argparse
import json
import os
from typing import List, Dict, Any, Optional


class Embed(nn.Module):
    """词编码层"""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(49408, 768)
        self.pos_embed = nn.Embedding(77, 768)
        self.register_buffer('pos_ids', torch.arange(77).unsqueeze(dim=0))

    def forward(self, input_ids):
        embed = self.embed(input_ids)
        pos_embed = self.pos_embed(self.pos_ids)
        return embed + pos_embed


class Atten(nn.Module):
    """注意力层"""

    def __init__(self):
        super().__init__()
        self.q = nn.Linear(768, 768)
        self.k = nn.Linear(768, 768)
        self.v = nn.Linear(768, 768)
        self.out = nn.Linear(768, 768)

    def forward(self, x):
        b = x.shape[0]
        q = self.q(x) * 0.125
        k = self.k(x)
        v = self.v(x)

        q = q.reshape(b, 77, 12, 64).transpose(1, 2).reshape(b * 12, 77, 64)
        k = k.reshape(b, 77, 12, 64).transpose(1, 2).reshape(b * 12, 77, 64)
        v = v.reshape(b, 77, 12, 64).transpose(1, 2).reshape(b * 12, 77, 64)

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn.reshape(b, 12, 77, 77)

        def get_mask(b):
            mask = torch.empty(b, 77, 77)
            mask.fill_(-float('inf'))
            mask.triu_(1)
            return mask.unsqueeze(1)

        attn = attn + get_mask(attn.shape[0]).to(attn.device)
        attn = attn.reshape(b * 12, 77, 77)
        attn = attn.softmax(dim=-1)
        attn = torch.bmm(attn, v)
        attn = attn.reshape(b, 12, 77, 64).transpose(1, 2).reshape(b, 77, 768)
        return self.out(attn)


class ClipEncoder(nn.Module):
    """编码器层"""

    def __init__(self):
        super().__init__()
        self.s1 = nn.Sequential(
            nn.LayerNorm(768),
            Atten(),
        )
        self.s2 = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 3072),
        )
        self.s3 = nn.Linear(3072, 768)

    def forward(self, x):
        x = x + self.s1(x)
        res = x
        x = self.s2(x)
        x = x * (x * 1.702).sigmoid()
        return res + self.s3(x)


class TextEncoder:
    """文本编码器模块"""

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.encoder = nn.Sequential(
            Embed(),
            ClipEncoder(), ClipEncoder(), ClipEncoder(), ClipEncoder(), ClipEncoder(),
            ClipEncoder(), ClipEncoder(), ClipEncoder(), ClipEncoder(), ClipEncoder(),
            ClipEncoder(), ClipEncoder(),
            nn.LayerNorm(768),
        ).to(device)

    def load_pretrained(self):
        """加载预训练参数"""
        params = CLIPTextModel.from_pretrained(
            'lansinuote/diffsion_from_scratch.params', subfolder='text_encoder')

        self.encoder[0].embed.load_state_dict(
            params.text_model.embeddings.token_embedding.state_dict())
        self.encoder[0].pos_embed.load_state_dict(
            params.text_model.embeddings.position_embedding.state_dict())

        for i in range(12):
            self.encoder[i + 1].s1[0].load_state_dict(
                params.text_model.encoder.layers[i].layer_norm1.state_dict())
            self.encoder[i + 1].s1[1].q.load_state_dict(
                params.text_model.encoder.layers[i].self_attn.q_proj.state_dict())
            self.encoder[i + 1].s1[1].k.load_state_dict(
                params.text_model.encoder.layers[i].self_attn.k_proj.state_dict())
            self.encoder[i + 1].s1[1].v.load_state_dict(
                params.text_model.encoder.layers[i].self_attn.v_proj.state_dict())
            self.encoder[i + 1].s1[1].out.load_state_dict(
                params.text_model.encoder.layers[i].self_attn.out_proj.state_dict())
            self.encoder[i + 1].s2[0].load_state_dict(
                params.text_model.encoder.layers[i].layer_norm2.state_dict())
            self.encoder[i + 1].s2[1].load_state_dict(
                params.text_model.encoder.layers[i].mlp.fc1.state_dict())
            self.encoder[i + 1].s3.load_state_dict(
                params.text_model.encoder.layers[i].mlp.fc2.state_dict())

        self.encoder[13].load_state_dict(params.text_model.final_layer_norm.state_dict())

    def encode(self, input_ids):
        """编码文本"""
        return self.encoder(input_ids)


class ResnetVAE(nn.Module):
    """VAE残差连接层"""

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.s = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=dim_in, eps=1e-6, affine=True),
            nn.SiLU(),
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=dim_out, eps=1e-6, affine=True),
            nn.SiLU(),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1),
        )
        self.res = None
        if dim_in != dim_out:
            self.res = nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        res = x
        if self.res:
            res = self.res(x)
        return res + self.s(x)


class AttenVAE(nn.Module):
    """VAE注意力层"""

    def __init__(self):
        super().__init__()
        self.norm = nn.GroupNorm(num_channels=512, num_groups=32, eps=1e-6, affine=True)
        self.q = nn.Linear(512, 512)
        self.k = nn.Linear(512, 512)
        self.v = nn.Linear(512, 512)
        self.out = nn.Linear(512, 512)

    def forward(self, x):
        res = x
        x = self.norm(x)
        x = x.flatten(start_dim=2).transpose(1, 2)
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        k = k.transpose(1, 2)
        atten = torch.baddbmm(torch.empty(1, 4096, 4096, device=q.device),
                              q, k, beta=0, alpha=0.044194173824159216)
        atten = torch.softmax(atten, dim=2)
        atten = atten.bmm(v)
        atten = self.out(atten)
        atten = atten.transpose(1, 2).reshape(-1, 512, 64, 64)
        return atten + res


class Pad(nn.Module):
    """PAD工具层"""

    def forward(self, x):
        return F.pad(x, (0, 1, 0, 1), mode='constant', value=0)


class VAE:
    """VAE模型"""

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.vae = self._build_vae().to(device)

    def _build_vae(self):
        return nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1),
            nn.Sequential(
                ResnetVAE(128, 128),
                ResnetVAE(128, 128),
                nn.Sequential(
                    Pad(),
                    nn.Conv2d(128, 128, 3, stride=2, padding=0),
                ),
            ),
            nn.Sequential(
                ResnetVAE(128, 256),
                ResnetVAE(256, 256),
                nn.Sequential(
                    Pad(),
                    nn.Conv2d(256, 256, 3, stride=2, padding=0),
                ),
            ),
            nn.Sequential(
                ResnetVAE(256, 512),
                ResnetVAE(512, 512),
                nn.Sequential(
                    Pad(),
                    nn.Conv2d(512, 512, 3, stride=2, padding=0),
                ),
            ),
            nn.Sequential(
                ResnetVAE(512, 512),
                ResnetVAE(512, 512),
            ),
            nn.Sequential(
                ResnetVAE(512, 512),
                AttenVAE(),
                ResnetVAE(512, 512),
            ),
            nn.Sequential(
                nn.GroupNorm(num_channels=512, num_groups=32, eps=1e-6),
                nn.SiLU(),
                nn.Conv2d(512, 8, 3, padding=1),
            ),
            nn.Conv2d(8, 8, 1),
        )

    def load_pretrained(self):
        """加载预训练参数"""
        params = AutoencoderKL.from_pretrained(
            'lansinuote/diffsion_from_scratch.params', subfolder='vae')

        def load_res(model, param):
            model.s[0].load_state_dict(param.norm1.state_dict())
            model.s[2].load_state_dict(param.conv1.state_dict())
            model.s[3].load_state_dict(param.norm2.state_dict())
            model.s[5].load_state_dict(param.conv2.state_dict())
            if isinstance(model.res, nn.Module):
                model.res.load_state_dict(param.conv_shortcut.state_dict())

        def load_atten(model, param):
            model.norm.load_state_dict(param.group_norm.state_dict())
            model.q.load_state_dict(param.query.state_dict())
            model.k.load_state_dict(param.key.state_dict())
            model.v.load_state_dict(param.value.state_dict())
            model.out.load_state_dict(param.proj_attn.state_dict())

        self.vae[0].load_state_dict(params.encoder.conv_in.state_dict())

        for i in range(4):
            load_res(self.vae[i + 1][0], params.encoder.down_blocks[i].resnets[0])
            load_res(self.vae[i + 1][1], params.encoder.down_blocks[i].resnets[1])
            if i != 3:
                self.vae[i + 1][2][1].load_state_dict(
                    params.encoder.down_blocks[i].downsamplers[0].conv.state_dict())

        load_res(self.vae[5][0], params.encoder.mid_block.resnets[0])
        load_res(self.vae[5][2], params.encoder.mid_block.resnets[1])
        load_atten(self.vae[5][1], params.encoder.mid_block.attentions[0])

        self.vae[6][0].load_state_dict(params.encoder.conv_norm_out.state_dict())
        self.vae[6][2].load_state_dict(params.encoder.conv_out.state_dict())
        self.vae[7].load_state_dict(params.quant_conv.state_dict())

    def encode(self, x):
        """编码图像"""
        return self.vae(x)

    def sample(self, h):
        """采样"""
        mean = h[:, :4]
        logvar = h[:, 4:]
        std = logvar.exp() ** 0.5
        h = torch.randn(mean.shape, device=mean.device)
        return mean + std * h

    def decode(self, h):
        """解码"""
        return self.vae.decoder(h)


class ResnetUNet(nn.Module):
    """UNet残差连接层"""

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.time = nn.Sequential(
            nn.SiLU(),
            nn.Linear(1280, dim_out),
            nn.Unflatten(dim=1, unflattened_size=(dim_out, 1, 1)),
        )
        self.s0 = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=dim_in, eps=1e-05, affine=True),
            nn.SiLU(),
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1),
        )
        self.s1 = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=dim_out, eps=1e-05, affine=True),
            nn.SiLU(),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1),
        )
        self.res = None
        if dim_in != dim_out:
            self.res = nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x, time):
        res = x
        time = self.time(time)
        x = self.s0(x) + time
        x = self.s1(x)
        if self.res:
            res = self.res(res)
        return res + x


class CrossAttention(nn.Module):
    """UNet交叉注意力层"""

    def __init__(self, dim_q, dim_kv):
        super().__init__()
        self.dim_q = dim_q
        self.q = nn.Linear(dim_q, dim_q, bias=False)
        self.k = nn.Linear(dim_kv, dim_q, bias=False)
        self.v = nn.Linear(dim_kv, dim_q, bias=False)
        self.out = nn.Linear(dim_q, dim_q)

    def forward(self, q, kv):
        q = self.q(q)
        k = self.k(kv)
        v = self.v(kv)

        def reshape(x):
            b, lens, dim = x.shape
            x = x.reshape(b, lens, 8, dim // 8)
            x = x.transpose(1, 2)
            x = x.reshape(b * 8, lens, dim // 8)
            return x

        q = reshape(q)
        k = reshape(k)
        v = reshape(v)

        atten = torch.baddbmm(
            torch.empty(q.shape[0], q.shape[1], k.shape[1], device=q.device),
            q, k.transpose(1, 2), beta=0, alpha=(self.dim_q // 8) ** -0.5)
        atten = atten.softmax(dim=-1)
        atten = atten.bmm(v)

        def reshape(x):
            b, lens, dim = x.shape
            x = x.reshape(b // 8, 8, lens, dim)
            x = x.transpose(1, 2)
            x = x.reshape(b // 8, lens, dim * 8)
            return x

        atten = reshape(atten)
        return self.out(atten)


class Transformer(nn.Module):
    """Transformer层"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.norm_in = nn.GroupNorm(num_groups=32, num_channels=dim, eps=1e-6, affine=True)
        self.cnn_in = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)
        self.norm_atten0 = nn.LayerNorm(dim, elementwise_affine=True)
        self.atten1 = CrossAttention(dim, dim)
        self.norm_atten1 = nn.LayerNorm(dim, elementwise_affine=True)
        self.atten2 = CrossAttention(dim, 768)
        self.norm_act = nn.LayerNorm(dim, elementwise_affine=True)
        self.fc0 = nn.Linear(dim, dim * 8)
        self.act = nn.GELU()
        self.fc1 = nn.Linear(dim * 4, dim)
        self.cnn_out = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)

    def forward(self, q, kv):
        b, _, h, w = q.shape
        res1 = q
        q = self.cnn_in(self.norm_in(q))
        q = q.permute(0, 2, 3, 1).reshape(b, h * w, self.dim)
        q = self.atten1(q=self.norm_atten0(q), kv=self.norm_atten0(q)) + q
        q = self.atten2(q=self.norm_atten1(q), kv=kv) + q
        res2 = q
        q = self.fc0(self.norm_act(q))
        d = q.shape[2] // 2
        q = q[:, :, :d] * self.act(q[:, :, d:])
        q = self.fc1(q) + res2
        q = q.reshape(b, h, w, self.dim).permute(0, 3, 1, 2).contiguous()
        return self.cnn_out(q) + res1


class DownBlock(nn.Module):
    """下采样块"""

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.tf0 = Transformer(dim_out)
        self.res0 = ResnetUNet(dim_in, dim_out)
        self.tf1 = Transformer(dim_out)
        self.res1 = ResnetUNet(dim_out, dim_out)
        self.out = nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=2, padding=1)

    def forward(self, out_vae, out_encoder, time):
        outs = []
        out_vae = self.res0(out_vae, time)
        out_vae = self.tf0(out_vae, out_encoder)
        outs.append(out_vae)
        out_vae = self.res1(out_vae, time)
        out_vae = self.tf1(out_vae, out_encoder)
        outs.append(out_vae)
        out_vae = self.out(out_vae)
        outs.append(out_vae)
        return out_vae, outs


class UpBlock(nn.Module):
    """上采样块"""

    def __init__(self, dim_in, dim_out, dim_prev, add_up):
        super().__init__()
        self.res0 = ResnetUNet(dim_out + dim_prev, dim_out)
        self.res1 = ResnetUNet(dim_out + dim_out, dim_out)
        self.res2 = ResnetUNet(dim_in + dim_out, dim_out)
        self.tf0 = Transformer(dim_out)
        self.tf1 = Transformer(dim_out)
        self.tf2 = Transformer(dim_out)
        self.out = None
        if add_up:
            self.out = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(dim_out, dim_out, kernel_size=3, padding=1),
            )

    def forward(self, out_vae, out_encoder, time, out_down):
        out_vae = self.res0(torch.cat([out_vae, out_down.pop()], dim=1), time)
        out_vae = self.tf0(out_vae, out_encoder)
        out_vae = self.res1(torch.cat([out_vae, out_down.pop()], dim=1), time)
        out_vae = self.tf1(out_vae, out_encoder)
        out_vae = self.res2(torch.cat([out_vae, out_down.pop()], dim=1), time)
        out_vae = self.tf2(out_vae, out_encoder)
        if self.out:
            out_vae = self.out(out_vae)
        return out_vae


class UNetModel:
    """UNet模型"""

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.unet = self._build_unet().to(device)

    def _build_unet(self):
        return nn.Sequential(
            nn.Conv2d(4, 320, kernel_size=3, padding=1),
            nn.Sequential(
                nn.Linear(320, 1280),
                nn.SiLU(),
                nn.Linear(1280, 1280),
            ),
            DownBlock(320, 320),
            DownBlock(320, 640),
            DownBlock(640, 1280),
            ResnetUNet(1280, 1280),
            ResnetUNet(1280, 1280),
            nn.Sequential(
                ResnetUNet(1280, 1280),
                Transformer(1280),
                ResnetUNet(1280, 1280),
            ),
            ResnetUNet(2560, 1280),
            ResnetUNet(2560, 1280),
            ResnetUNet(2560, 1280),
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(1280, 1280, kernel_size=3, padding=1),
            ),
            UpBlock(640, 1280, 1280, True),
            UpBlock(320, 640, 1280, True),
            UpBlock(320, 320, 640, False),
            nn.Sequential(
                nn.GroupNorm(num_channels=320, num_groups=32, eps=1e-5),
                nn.SiLU(),
                nn.Conv2d(320, 4, kernel_size=3, padding=1),
            )
        )

    def load_pretrained(self):
        """加载预训练参数"""
        params = UNet2DConditionModel.from_pretrained(
            'lansinuote/diffsion_from_scratch.params', subfolder='unet')

        def load_tf(model, param):
            model.norm_in.load_state_dict(param.norm.state_dict())
            model.cnn_in.load_state_dict(param.proj_in.state_dict())
            model.atten1.q.load_state_dict(param.transformer_blocks[0].attn1.to_q.state_dict())
            model.atten1.k.load_state_dict(param.transformer_blocks[0].attn1.to_k.state_dict())
            model.atten1.v.load_state_dict(param.transformer_blocks[0].attn1.to_v.state_dict())
            model.atten1.out.load_state_dict(param.transformer_blocks[0].attn1.to_out[0].state_dict())
            model.atten2.q.load_state_dict(param.transformer_blocks[0].attn2.to_q.state_dict())
            model.atten2.k.load_state_dict(param.transformer_blocks[0].attn2.to_k.state_dict())
            model.atten2.v.load_state_dict(param.transformer_blocks[0].attn2.to_v.state_dict())
            model.atten2.out.load_state_dict(param.transformer_blocks[0].attn2.to_out[0].state_dict())
            model.fc0.load_state_dict(param.transformer_blocks[0].ff.net[0].proj.state_dict())
            model.fc1.load_state_dict(param.transformer_blocks[0].ff.net[2].state_dict())
            model.norm_atten0.load_state_dict(param.transformer_blocks[0].norm1.state_dict())
            model.norm_atten1.load_state_dict(param.transformer_blocks[0].norm2.state_dict())
            model.norm_act.load_state_dict(param.transformer_blocks[0].norm3.state_dict())
            model.cnn_out.load_state_dict(param.proj_out.state_dict())

        def load_res(model, param):
            model.time[1].load_state_dict(param.time_emb_proj.state_dict())
            model.s0[0].load_state_dict(param.norm1.state_dict())
            model.s0[2].load_state_dict(param.conv1.state_dict())
            model.s1[0].load_state_dict(param.norm2.state_dict())
            model.s1[2].load_state_dict(param.conv2.state_dict())
            if isinstance(model.res, nn.Module):
                model.res.load_state_dict(param.conv_shortcut.state_dict())

        self.unet[0].load_state_dict(params.conv_in.state_dict())
        self.unet[1][0].load_state_dict(params.time_embedding.linear_1.state_dict())
        self.unet[1][2].load_state_dict(params.time_embedding.linear_2.state_dict())

    def forward(self, out_vae, out_encoder, time):
        """前向传播"""
        return self.unet(out_vae, out_encoder, time)


class DiffusionTrainer:
    """扩散模型训练器"""

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.text_encoder = TextEncoder(device)
        self.vae = VAE(device)
        self.unet = UNetModel(device)
        self.scheduler = None
        self.tokenizer = None

    def initialize_models(self):
        """初始化模型并加载预训练权重"""
        print("Initializing models...")
        pipeline = DiffusionPipeline.from_pretrained(
            'lansinuote/diffsion_from_scratch.params', safety_checker=None)
        self.scheduler = pipeline.scheduler
        self.tokenizer = pipeline.tokenizer
        del pipeline

        self.text_encoder.load_pretrained()
        self.vae.load_pretrained()
        self.unet.load_pretrained()

        # 准备训练
        self.text_encoder.encoder.requires_grad_(False)
        self.vae.vae.requires_grad_(False)
        self.unet.unet.requires_grad_(True)

        self.text_encoder.encoder.eval()
        self.vae.vae.eval()
        self.unet.unet.train()

        return self

    def prepare_dataset(self):
        """准备数据集"""
        print("Preparing dataset...")
        dataset = load_dataset(path='lansinuote/diffusion.4.text_to_image.book', split='train')

        compose = torchvision.transforms.Compose([
            torchvision.transforms.Resize(512, interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
            torchvision.transforms.CenterCrop(512),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5], [0.5]),
        ])

        def transform_data(data):
            pixel_values = [compose(i) for i in data['image']]
            input_ids = self.tokenizer.batch_encode_plus(
                data['text'], padding='max_length', truncation=True, max_length=77).input_ids
            return {'pixel_values': pixel_values, 'input_ids': input_ids}

        dataset = dataset.map(transform_data, batched=True, batch_size=100,
                              num_proc=1, remove_columns=['image', 'text'])
        dataset.set_format(type='torch')
        return dataset

    def create_dataloader(self, dataset, batch_size=1):
        """创建数据加载器"""

        def collate_fn(data):
            pixel_values = torch.stack([i['pixel_values'] for i in data]).to(self.device)
            input_ids = torch.stack([i['input_ids'] for i in data]).to(self.device)
            return {'pixel_values': pixel_values, 'input_ids': input_ids}

        return DataLoader(dataset, shuffle=True, collate_fn=collate_fn, batch_size=batch_size)

    def get_loss(self, data):
        """计算损失"""
        with torch.no_grad():
            out_encoder = self.text_encoder.encode(data['input_ids'])
            out_vae = self.vae.encode(data['pixel_values'])
            out_vae = self.vae.sample(out_vae)
            out_vae = out_vae * 0.18215  # vae.config.scaling_factor

        noise = torch.randn_like(out_vae)
        noise_step = torch.randint(0, 1000, (1,)).long().to(self.device)
        out_vae_noise = self.scheduler.add_noise(out_vae, noise, noise_step)

        out_unet = self.unet.forward(out_vae=out_vae_noise, out_encoder=out_encoder, time=noise_step)
        return nn.MSELoss()(out_unet, noise)

    def train(self, epochs=200, batch_size=1, save_path='save/unet.model'):
        """训练模型"""
        print("Starting training...")
        dataset = self.prepare_dataset()
        loader = self.create_dataloader(dataset, batch_size)

        optimizer = torch.optim.AdamW(
            self.unet.unet.parameters(), lr=1e-5, betas=(0.9, 0.999),
            weight_decay=0.01, eps=1e-8)
        criterion = nn.MSELoss()

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        loss_sum = 0
        for epoch in range(epochs):
            for i, data in enumerate(loader):
                loss = self.get_loss(data) / 4
                loss.backward()
                loss_sum += loss.item()

                if (epoch * len(loader) + i) % 4 == 0:
                    torch.nn.utils.clip_grad_norm_(self.unet.unet.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss_sum}")
                loss_sum = 0

        torch.save(self.unet.unet.to('cpu'), save_path)
        print(f"Training completed. Model saved to {save_path}")


class DiffusionGenerator:
    """扩散模型生成器"""

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.text_encoder = TextEncoder(device)
        self.vae = VAE(device)
        self.unet = UNetModel(device)
        self.scheduler = None
        self.tokenizer = None

    def initialize_models(self):
        """初始化模型"""
        print("Initializing models for generation...")
        pipeline = DiffusionPipeline.from_pretrained(
            'lansinuote/diffsion_from_scratch.params', safety_checker=None)
        self.scheduler = pipeline.scheduler
        self.tokenizer = pipeline.tokenizer
        del pipeline

        self.text_encoder.load_pretrained()
        self.vae.load_pretrained()
        self.unet.load_pretrained()

        self.text_encoder.encoder.eval()
        self.vae.vae.eval()
        self.unet.unet.eval()

        return self

    def load_custom_unet(self, model_path):
        """加载自定义UNet模型"""
        self.unet.unet = torch.load(model_path).to(self.device)
        self.unet.unet.eval()
        return self

    @torch.no_grad()
    def generate(self, text, num_inference_steps=50, guidance_scale=7.5):
        """生成图像"""
        # 词编码
        pos = self.tokenizer(
            text, padding='max_length', max_length=77,
            truncation=True, return_tensors='pt').input_ids.to(self.device)
        neg = self.tokenizer(
            '', padding='max_length', max_length=77,
            truncation=True, return_tensors='pt').input_ids.to(self.device)

        pos = self.text_encoder.encode(pos)
        neg = self.text_encoder.encode(neg)
        out_encoder = torch.cat((neg, pos), dim=0)

        # 从随机噪声开始
        out_vae = torch.randn(1, 4, 64, 64, device=self.device)

        # 生成时间步
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        for time in self.scheduler.timesteps:
            # 往图中加噪音
            noise = torch.cat((out_vae, out_vae), dim=0)
            noise = self.scheduler.scale_model_input(noise, time)

            # 计算噪音
            pred_noise = self.unet.forward(out_vae=noise, out_encoder=out_encoder, time=time)

            # 从正例图中减去反例图
            pred_noise = pred_noise[0] + guidance_scale * (pred_noise[1] - pred_noise[0])

            # 重新添加噪音，以进行下一步计算
            out_vae = self.scheduler.step(pred_noise, time, out_vae).prev_sample

        # 从压缩图恢复成图片
        out_vae = 1 / 0.18215 * out_vae
        image = self.vae.decode(out_vae)

        # 转换成图片数据
        image = image.cpu()
        image = (image + 1) / 2
        image = image.clamp(0, 1)
        image = image.permute(0, 2, 3, 1)
        return image.numpy()[0]

    def show_generated_images(self, texts):
        """显示生成的图像"""
        images = [self.generate(text) for text in texts]

        plt.figure(figsize=(20, 10))
        for i in range(len(texts)):
            plt.subplot(2, 3, i + 1)
            plt.imshow(images[i])
            plt.axis('off')
            plt.title(texts[i][:30] + '...' if len(texts[i]) > 30 else texts[i], fontsize=8)

        plt.tight_layout()
        plt.show()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Diffusion Model Training and Generation')
    parser.add_argument('--mode', type=str, choices=['train', 'generate', 'test'],
                        default='generate', help='运行模式: train, generate, test')
    parser.add_argument('--model_path', type=str, default='save/unet.model',
                        help='自定义模型路径')
    parser.add_argument('--epochs', type=int, default=200, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=1, help='批次大小')

    args = parser.parse_args()

    if args.mode == 'train':
        # 训练模式
        trainer = DiffusionTrainer()
        trainer.initialize_models()
        trainer.train(epochs=args.epochs, batch_size=args.batch_size)

    elif args.mode == 'generate':
        # 生成模式
        generator = DiffusionGenerator()
        generator.initialize_models()

        texts = [
            'a pixel art character with square orange glasses, a whale-shaped head and a teal-colored body on a warm background',
            'a pixel art character with square green and blue glasses, a hanger-shaped head and a red-colored body on a warm background',
            'a pixel art character with square red glasses, a square-shaped head and a green-colored body on a warm background',
            'a pixel art character with square blue glasses, a circle-shaped head and a yellow-colored body on a warm background',
            'a pixel art character with square yellow glasses, a triangle-shaped head and a purple-colored body on a warm background',
            'a pixel art character with square purple glasses, a star-shaped head and a orange-colored body on a warm background'
        ]

        generator.show_generated_images(texts)

    elif args.mode == 'test':
        # 测试模式 - 加载自定义模型并生成
        generator = DiffusionGenerator()
        generator.initialize_models()
        generator.load_custom_unet(args.model_path)

        test_texts = ['a beautiful landscape with mountains and lake']
        generator.show_generated_images(test_texts)


if __name__ == '__main__':
    main()
