# -*- coding: utf-8 -*-
# @project: LargeDemo
# @filename: ch13_controlnet.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/8/26 10:24
# https://chat.deepseek.com/a/chat/s/9ed71823-f88a-4508-b811-e326d9ab19e1
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from diffusers import (
    StableDiffusionControlNetPipeline,
    StableDiffusionPipeline,
    DDPMScheduler,
    UNet2DConditionModel,
    ControlNetModel,
    AutoencoderKL,
)
from diffusers.models.embeddings import TimestepEmbedding, get_timestep_embedding
from diffusers.models.unet_2d_blocks import CrossAttnDownBlock2D, DownBlock2D, UNetMidBlock2DCrossAttn
from transformers import (
    CLIPTokenizer,
    CLIPTextModel,
    PretrainedConfig,
    PreTrainedModel,
)
from datasets import load_dataset
from matplotlib import pyplot as plt
import PIL.Image
import os

# Global constants
REPO_ID = 'lansinuote/diffusion.7.control_net'
CHECKPOINT = 'runwayml/stable-diffusion-v1-5'


class ControlNet(PreTrainedModel):
    config_class = PretrainedConfig

    def __init__(self, config):
        super().__init__(config)

        # Input embedding part
        self.out_vae_noise_embed = nn.Conv2d(4, 320, kernel_size=3, padding=1)

        self.time_embed = TimestepEmbedding(320, 1280, act_fn='silu')

        self.condition_embed = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 96, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(96, 256, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(256, 320, kernel_size=3, stride=1, padding=1),
        )

        # UNet down part
        self.unet_down = nn.ModuleList([])
        for i in range(3):
            self.unet_down.append(
                CrossAttnDownBlock2D(
                    num_layers=2,
                    in_channels=[320, 320, 640][i],
                    out_channels=[320, 640, 1280][i],
                    temb_channels=1280,
                    add_downsample=True,
                    resnet_eps=1e-5,
                    resnet_act_fn='silu',
                    resnet_groups=32,
                    downsample_padding=1,
                    cross_attention_dim=768,
                    attn_num_head_channels=8,
                    dual_cross_attention=False,
                    use_linear_projection=False,
                    only_cross_attention=False,
                    upcast_attention=False,
                    resnet_time_scale_shift='default'
                )
            )
        self.unet_down.append(
            DownBlock2D(
                num_layers=2,
                in_channels=1280,
                out_channels=1280,
                temb_channels=1280,
                add_downsample=False,
                resnet_eps=1e-5,
                resnet_act_fn='silu',
                resnet_groups=32,
                downsample_padding=1,
                resnet_time_scale_shift='default'
            )
        )

        # UNet mid part
        self.unet_mid = UNetMidBlock2DCrossAttn(
            in_channels=1280,
            temb_channels=1280,
            resnet_eps=1e-5,
            resnet_act_fn='silu',
            output_scale_factor=1,
            resnet_time_scale_shift='default',
            cross_attention_dim=768,
            attn_num_head_channels=8,
            resnet_groups=32,
            use_linear_projection=False,
            upcast_attention=False
        )

        # Control down part
        self.control_down = nn.ModuleList([
            nn.Conv2d(320, 320, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(320, 320, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(320, 320, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(320, 320, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(640, 640, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(640, 640, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(640, 640, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(1280, 1280, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(1280, 1280, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(1280, 1280, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(1280, 1280, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(1280, 1280, kernel_size=1, stride=1, padding=0),
        ])

        # Control mid part
        self.control_mid = nn.Conv2d(1280, 1280, kernel_size=1)

    def forward(self, out_vae_noise, noise_step, out_encoder, condition):
        # Encode noise_step
        noise_step = get_timestep_embedding(
            noise_step, 320, flip_sin_to_cos=True, downscale_freq_shift=0
        )
        noise_step = self.time_embed(noise_step, None)

        # Project out_vae_noise to higher dimension
        out_vae_noise = self.out_vae_noise_embed(out_vae_noise)

        # Project condition to same dimension space as out_vae_noise
        condition = self.condition_embed(condition)

        # Add condition information to out_vae_noise
        out_vae_noise += condition

        # UNet down part calculation
        out_unet_down = [out_vae_noise]
        for i in range(4):
            if i < 3:
                out_vae_noise, out = self.unet_down[i](
                    hidden_states=out_vae_noise,
                    temb=noise_step,
                    encoder_hidden_states=out_encoder,
                    attention_mask=None,
                    cross_attention_kwargs=None
                )
            else:
                out_vae_noise, out = self.unet_down[i](
                    hidden_states=out_vae_noise, temb=noise_step
                )
            out_unet_down.extend(out)

        # UNet mid calculation
        out_vae_noise = self.unet_mid(
            out_vae_noise,
            noise_step,
            encoder_hidden_states=out_encoder,
            attention_mask=None,
            cross_attention_kwargs=None
        )

        # Control down part calculation
        out_control_down = [
            self.control_down[i](out_unet_down[i]) for i in range(12)
        ]

        # Control mid part calculation
        out_control_mid = self.control_mid(out_vae_noise)

        return out_control_down, out_control_mid


def load_params(controlnet, unet):
    """Load parameters from UNet to ControlNet"""
    controlnet.out_vae_noise_embed.load_state_dict(unet.conv_in.state_dict())
    controlnet.time_embed.load_state_dict(unet.time_embedding.state_dict())
    controlnet.unet_down.load_state_dict(unet.down_blocks.state_dict())
    controlnet.unet_mid.load_state_dict(unet.mid_block.state_dict())


def print_model_size(name, model):
    """Print model parameter count"""
    print(name, sum(i.numel() for i in model.parameters()) / 10000)


def setup_data_transforms():
    """Setup data transforms for preprocessing"""
    tokenizer = CLIPTokenizer.from_pretrained(CHECKPOINT, subfolder='tokenizer')

    compose = torchvision.transforms.Compose([
        torchvision.transforms.Resize(
            512, interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        ),
        torchvision.transforms.CenterCrop(512),
        torchvision.transforms.ToTensor(),
    ])

    norm = torchvision.transforms.Normalize([0.5], [0.5])

    return tokenizer, compose, norm


def preprocess_data(data, tokenizer, compose, norm):
    """Preprocess data for training"""
    # Text encoding
    input_ids = tokenizer.batch_encode_plus(
        data['text'],
        max_length=77,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    ).input_ids

    # Image encoding
    pixel_values = norm(compose(data['image'][0])).unsqueeze(dim=0)
    conditioning_pixel_values = compose(data['conditioning_image'][0]).unsqueeze(dim=0)

    return {
        'input_ids': input_ids,
        'pixel_values': pixel_values,
        'conditioning_pixel_values': conditioning_pixel_values
    }


def load_and_preprocess_dataset(use_custom_dataset=True):
    """Load and preprocess the dataset"""
    if use_custom_dataset:
        dataset = load_dataset(path=REPO_ID, split='train')
    else:
        dataset = load_dataset('fusing/fill50k', split='train')

    tokenizer, compose, norm = setup_data_transforms()

    def transform_function(data):
        return preprocess_data(data, tokenizer, compose, norm)

    dataset = dataset.with_transform(transform_function)
    return dataset


def create_data_loader(dataset, batch_size=1):
    """Create data loader from dataset"""
    loader = DataLoader(
        dataset,
        shuffle=True,
        collate_fn=None,
        batch_size=batch_size
    )
    return loader


def setup_models():
    """Setup all models for training"""
    encoder = CLIPTextModel.from_pretrained(CHECKPOINT, subfolder='text_encoder')
    vae = AutoencoderKL.from_pretrained(CHECKPOINT, subfolder='vae')
    unet = UNet2DConditionModel.from_pretrained(CHECKPOINT, subfolder='unet')

    # Define ControlNet
    controlnet = ControlNet(PretrainedConfig())
    load_params(controlnet, unet)

    # Freeze parameters
    vae.requires_grad_(False)
    encoder.requires_grad_(False)
    unet.requires_grad_(False)

    return encoder, vae, unet, controlnet


def setup_training_tools():
    """Setup scheduler, optimizer and criterion for training"""
    scheduler = DDPMScheduler.from_pretrained(CHECKPOINT, subfolder='scheduler')

    optimizer = torch.optim.AdamW(
        controlnet.parameters(),
        lr=1e-5,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        eps=1e-8
    )

    criterion = nn.MSELoss()

    return scheduler, optimizer, criterion


def get_loss(data, encoder, vae, unet, controlnet, scheduler, device='cuda'):
    """Calculate loss for training"""
    # Text encoding
    out_encoder = encoder(data['input_ids'])[0]

    # Extract image feature map
    out_vae = vae.encode(data['pixel_values']).latent_dist.sample()
    out_vae = out_vae * 0.18215  # vae.config.scaling_factor

    # Random noise
    noise = torch.randn_like(out_vae)

    # Add noise to feature map
    noise_step = torch.randint(0, 1000, (1,)).long().to(device)
    out_vae_noise = scheduler.add_noise(out_vae, noise, noise_step)

    # Use ControlNet to compute UNet down and mid parts
    out_control_down, out_control_mid = controlnet(
        out_vae_noise,
        noise_step,
        out_encoder,
        data['conditioning_pixel_values']
    )

    # Compute noise from feature map based on text information
    out_unet = unet(
        out_vae_noise,
        noise_step,
        encoder_hidden_states=out_encoder,
        down_block_additional_residuals=out_control_down,
        mid_block_additional_residual=out_control_mid
    ).sample

    # Compute MSELoss
    return criterion(out_unet, noise)


def train_model(encoder, vae, unet, controlnet, loader, scheduler, optimizer, criterion):
    """Train the ControlNet model"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    unet.to(device)
    encoder.to(device)
    vae.to(device)
    controlnet.to(device)
    controlnet.train()

    loss_sum = 0
    for i, data in enumerate(loader):
        for k in data.keys():
            data[k] = data[k].to(device)

        loss = get_loss(data, encoder, vae, unet, controlnet, scheduler, device) / 4
        loss.backward()
        loss_sum += loss.item()

        if i % 4 == 0:
            torch.nn.utils.clip_grad_norm_(controlnet.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        if i % 2000 == 0:
            print(i, loss_sum)
            loss_sum = 0

    # Save to local
    os.makedirs('./save', exist_ok=True)
    torch.save(controlnet.cpu(), './save/controlnet.model')


def test_pipeline(pipeline):
    """Test the pipeline with sample images"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pipeline = pipeline.to(device)

    texts = [
        'red circle with blue background',
        'cyan circle with brown floral background',
    ]

    images = [
        PIL.Image.open('images/conditioning_image_1.png').convert('RGB'),
        PIL.Image.open('images/conditioning_image_2.png').convert('RGB'),
    ]

    plt.figure(figsize=(20, 10))
    for i in range(2):
        image = pipeline(texts[i], images[i], num_inference_steps=20).images[0]

        plt.subplot(1, 2, i + 1)
        plt.imshow(image)
        plt.axis('off')

    plt.show()


class MyControlNetModel(ControlNetModel):
    """Wrapper class for ControlNet model"""

    def __init__(self, unet):
        super().__init__(cross_attention_dim=768)
        controlnet = ControlNet(PretrainedConfig())
        load_params(controlnet, unet)
        self.controlnet = controlnet

    def forward(self,
                sample,
                timestep,
                encoder_hidden_states,
                controlnet_cond,
                conditioning_scale=1.0,
                class_labels=None,
                timestep_cond=None,
                attention_mask=None,
                cross_attention_kwargs=None,
                return_dict=True):
        timestep = timestep.reshape(1)
        return self.controlnet(
            out_vae_noise=sample,
            noise_step=timestep,
            out_encoder=encoder_hidden_states,
            condition=controlnet_cond
        )


def create_test_pipeline():
    """Create pipeline for testing"""
    unet = UNet2DConditionModel.from_pretrained(
        CHECKPOINT, subfolder='unet'
    )

    controlnet = MyControlNetModel(unet)

    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        CHECKPOINT,
        unet=unet,
        controlnet=controlnet,
        safety_checker=None
    )

    return pipeline


def test_pretrained_model():
    """Test the pretrained model"""
    pipeline = create_test_pipeline()
    test_pipeline(pipeline)


def test_trained_model_local():
    """Test locally trained model"""
    pipeline = create_test_pipeline()
    pipeline.controlnet.controlnet = torch.load('./save/controlnet.model')
    test_pipeline(pipeline)


def test_online_pretrained_model():
    """Test online pretrained model"""
    pipeline = create_test_pipeline()
    pipeline.controlnet.controlnet = ControlNet.from_pretrained(REPO_ID)
    test_pipeline(pipeline)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='ControlNet Training and Testing')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['train', 'test_pretrained', 'test_trained', 'test_online'],
                        help='Mode to run: train, test_pretrained, test_trained, or test_online')
    parser.add_argument('--use_custom_dataset', action='store_true',
                        help='Use custom dataset from Hugging Face')

    args = parser.parse_args()

    if args.mode == 'train':
        # Load and preprocess dataset
        dataset = load_and_preprocess_dataset(args.use_custom_dataset)
        loader = create_data_loader(dataset)

        # Setup models
        encoder, vae, unet, controlnet = setup_models()

        # Setup training tools
        scheduler, optimizer, criterion = setup_training_tools()

        # Print model sizes
        print_model_size('encoder', encoder)
        print_model_size('vae', vae)
        print_model_size('unet', unet)
        print_model_size('controlnet', controlnet)

        # Train model
        train_model(encoder, vae, unet, controlnet, loader, scheduler, optimizer, criterion)

    elif args.mode == 'test_pretrained':
        test_pretrained_model()

    elif args.mode == 'test_trained':
        test_trained_model_local()

    elif args.mode == 'test_online':
        test_online_pretrained_model()
