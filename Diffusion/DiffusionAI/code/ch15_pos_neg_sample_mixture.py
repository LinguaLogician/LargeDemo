# -*- coding: utf-8 -*-
# @project: LargeDemo
# @filename: ch15_pos_neg_sample_mixture.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/8/26 10:25
# https://chat.deepseek.com/a/chat/s/3652fdb3-842e-41b3-a635-0f9e267abaf1

import torch
import torch.nn as nn
import torchvision
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import (
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    AutoencoderKL,
    UNet2DConditionModel,
    DDPMScheduler
)
from diffusers.models.attention_processor import CustomDiffusionAttnProcessor
from diffusers.loaders import AttnProcsLayers
from datasets import load_dataset
from matplotlib import pyplot as plt
import PIL.Image
import random
import numpy as np
import os
from typing import Dict, List, Tuple, Optional


class CustomDiffusionTrainer:
    def __init__(self, checkpoint: str = 'CompVis/stable-diffusion-v1-4'):
        self.checkpoint = checkpoint
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Initialize components
        self.tokenizer = None
        self.encoder = None
        self.vae = None
        self.unet = None
        self.scheduler = None
        self.attn_processor = None
        self.optimizer = None
        self.criterion = None

    def setup_components(self):
        """Initialize all model components"""
        # Text encoder
        self.encoder = CLIPTextModel.from_pretrained(self.checkpoint, subfolder='text_encoder')

        # VAE
        self.vae = AutoencoderKL.from_pretrained(self.checkpoint, subfolder='vae')

        # UNet
        self.unet = UNet2DConditionModel.from_pretrained(self.checkpoint, subfolder='unet')

        # Tokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained(self.checkpoint, subfolder='tokenizer')

        # Scheduler
        self.scheduler = DDPMScheduler.from_pretrained(self.checkpoint, subfolder='scheduler')

        # Move to device
        self.encoder.to(self.device)
        self.vae.to(self.device)
        self.unet.to(self.device)

    def add_custom_token(self, token: str, init_token: str = 'ktn'):
        """Add a custom token to the tokenizer and initialize its embedding"""
        self.tokenizer.add_tokens(token)
        self.encoder.resize_token_embeddings(len(self.tokenizer))

        # Initialize new token embedding with an existing token
        token_id = self.tokenizer.convert_tokens_to_ids(token)
        init_token_id = self.tokenizer.convert_tokens_to_ids(init_token)

        token_embeds = self.encoder.get_input_embeddings().weight.data
        token_embeds[token_id] = token_embeds[init_token_id]

        return token_id

    def freeze_models(self):
        """Freeze all models except token embeddings"""
        self.encoder.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.encoder.text_model.embeddings.token_embedding.requires_grad_(True)

    def create_lora_layers(self):
        """Create and setup LoRA attention processors"""
        attn_processor = {}

        for name, _ in self.unet.attn_processors.items():
            # Determine hidden size based on block type
            if name.startswith('mid_block'):
                hidden_size = 1280  # unet.config.block_out_channels[-1]
            elif name.startswith('up_blocks'):
                hidden_size = [1280, 1280, 640, 320][int(name[10])]
            elif name.startswith('down_blocks'):
                hidden_size = [320, 640, 1280, 1280][int(name[12])]
            else:
                continue

            # Only train key-value for cross-attention layers
            train_kv = not name.endswith('attn1.processor')

            # Create custom attention processor
            attn_processor[name] = CustomDiffusionAttnProcessor(
                train_kv=train_kv,
                train_q_out=False,
                hidden_size=hidden_size,
                cross_attention_dim=768 if train_kv else None,
            )

            # Load pretrained weights for KV layers
            if train_kv:
                param = {
                    'to_k_custom_diffusion.weight': self.unet.state_dict()[
                        name.split('.processor')[0] + '.to_k.weight'],
                    'to_v_custom_diffusion.weight': self.unet.state_dict()[
                        name.split('.processor')[0] + '.to_v.weight'],
                }
                attn_processor[name].load_state_dict(param)

        # Set attention processors
        self.unet.set_attn_processor(attn_processor)

        # Create trainable layers
        self.attn_processor = AttnProcsLayers(self.unet.attn_processors)

        return self.attn_processor

    def setup_optimizer(self, lr: float = 2e-5):
        """Setup optimizer for training"""
        parameters = list(self.encoder.get_input_embeddings().parameters()) + list(self.attn_processor.parameters())

        self.optimizer = torch.optim.AdamW(
            parameters,
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=0.01,
            eps=1e-8
        )

        self.criterion = nn.MSELoss(reduction='none')

        return self.optimizer

    def preprocess_instance_image(self, image_path: str, resize: int) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess instance image with augmentation"""
        image = PIL.Image.open(image_path)

        # Calculate offsets
        offset_y = random.randint(0, abs(resize - 512))
        offset_x = random.randint(0, abs(resize - 512))

        # Resize
        image = image.resize((resize, resize), resample=PIL.Image.Resampling.BILINEAR)

        # Normalize to [-1, 1]
        image = (np.array(image) / 127.5 - 1.0).astype(np.float32)

        out_image = np.zeros((512, 512, 3), dtype=np.float32)
        mask = np.zeros((512, 512))

        if resize > 512:
            # Crop if resized larger than 512
            out_image = image[offset_y:offset_y + 512, offset_x:offset_x + 512]
            mask[:, :] = 1.0
        else:
            # Paste if resized smaller than 512
            out_image[offset_y:offset_y + resize, offset_x:offset_x + resize] = image
            mask[offset_y:offset_y + resize, offset_x:offset_x + resize] = 1.0

        # Resize mask to 64x64
        mask = torch.FloatTensor(mask).unsqueeze(dim=0)
        mask = torchvision.transforms.Resize([64, 64])(mask).squeeze(dim=0).numpy()

        # Binarize mask
        mask[mask > 0.5] = 1.0
        mask[mask <= 0.5] = 0.0

        return out_image, mask

    def get_instance_data(self, instance_dir: str, token: str = 'maorongrong') -> Dict[str, torch.Tensor]:
        """Get a single instance data sample with augmentation"""
        # Randomly select an instance image
        image_files = [f for f in os.listdir(instance_dir) if f.endswith(('.jpg', '.png'))]
        image_path = os.path.join(instance_dir, random.choice(image_files))

        # Apply horizontal flip
        image = PIL.Image.open(image_path)
        image = torchvision.transforms.RandomHorizontalFlip(0.5)(image)

        # Random scale
        scale = random.randint(614, 715)
        if random.random() < 0.66:
            scale = random.randint(170, 512)

        # Preprocess image
        image, mask = self.preprocess_instance_image(image_path, scale)

        # Create prompt based on scale
        prompt = f'photo of a {token} cat'
        if scale < 307.2:
            prompt = random.choice(['a far away ', 'very small ']) + prompt
        if scale > 512:
            prompt = random.choice(['zoomed in ', 'close up ']) + prompt

        return {
            'pixel_values': torch.FloatTensor(image).permute(2, 0, 1),
            'mask': torch.FloatTensor(mask),
            'input_ids': self.tokenizer(
                prompt,
                truncation=True,
                padding='max_length',
                max_length=77,
                return_tensors='pt'
            ).input_ids.squeeze(dim=0)
        }

    def load_negative_dataset(self, repo_id: str = 'lansinuote/diffusion.9.custom_diffusion'):
        """Load and preprocess negative dataset"""
        dataset = load_dataset(repo_id, split='train')

        # Define transforms
        compose = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(0.5),
            torchvision.transforms.Resize(
                512, interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
            torchvision.transforms.RandomCrop(512),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5], [0.5]),
        ])

        def transform_fn(data):
            # Apply image augmentation
            pixel_values = [compose(i) for i in data['image']]

            # Tokenize text
            input_ids = self.tokenizer(
                data['prompt'],
                truncation=True,
                padding='max_length',
                max_length=77,
                return_tensors='pt'
            ).input_ids

            return {'pixel_values': pixel_values, 'input_ids': input_ids}

        # Apply transformation
        dataset = dataset.with_transform(transform_fn)

        return dataset

    def create_data_loader(self, dataset, instance_dir: str, batch_size: int = 1):
        """Create data loader with custom collate function"""

        def collate_fn(data):
            # Get base data
            input_ids = data[0]['input_ids']
            pixel_values = data[0]['pixel_values']

            # Get instance data
            instance = self.get_instance_data(instance_dir)

            # Stack data
            input_ids = torch.stack([input_ids, instance['input_ids']])
            pixel_values = torch.stack([pixel_values, instance['pixel_values']])

            # Reshape mask
            mask = instance['mask'].reshape(1, 1, 64, 64)

            return {'input_ids': input_ids, 'pixel_values': pixel_values, 'mask': mask}

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )

    def compute_loss(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute training loss"""
        # Text encoding
        out_encoder = self.encoder(data['input_ids'])[0]

        # Extract image features
        out_vae = self.vae.encode(data['pixel_values']).latent_dist.sample()
        out_vae = out_vae * 0.18215  # vae.config.scaling_factor

        # Create noise
        noise = torch.randn_like(out_vae)

        # Add noise to features
        noise_step = torch.randint(0, 1000, (2,)).long().to(self.device)
        out_vae_noise = self.scheduler.add_noise(out_vae, noise, noise_step)

        # Predict noise
        out_unet = self.unet(out_vae_noise, noise_step, out_encoder).sample

        # Split into base and instance parts
        out_unet, out_unet_instance = torch.chunk(out_unet, 2, dim=0)
        noise, noise_instance = torch.chunk(noise, 2, dim=0)

        # Base loss
        loss = self.criterion(out_unet, noise).mean()

        # Instance loss with mask
        loss_instance = self.criterion(out_unet_instance, noise_instance)
        loss_instance = loss_instance * data['mask']
        loss_instance = loss_instance.sum([1, 2, 3]) / data['mask'].sum([1, 2, 3])
        loss_instance = loss_instance.mean()

        return loss_instance + loss

    def train(self, data_loader, epochs: int = 5, save_dir: str = './save'):
        """Train the model"""
        self.unet.train()

        os.makedirs(save_dir, exist_ok=True)

        loss_sum = 0
        for epoch in range(epochs):
            for i, data in enumerate(data_loader):
                # Move data to device
                for k in data.keys():
                    data[k] = data[k].to(self.device)

                # Compute loss and backpropagate
                loss = self.compute_loss(data)
                loss.backward()
                loss_sum += loss.item()

                # Gradient clipping and optimization step
                torch.nn.utils.clip_grad_norm_(
                    list(self.encoder.get_input_embeddings().parameters()) +
                    list(self.attn_processor.parameters()),
                    1.0
                )
                self.optimizer.step()
                self.optimizer.zero_grad()

            # Print loss
            if epoch % 1 == 0:
                print(f"Epoch {epoch}: Loss = {loss_sum}")
                loss_sum = 0

        # Save embeddings
        token_id = self.tokenizer.convert_tokens_to_ids('maorongrong')
        torch.save(
            {'maorongrong': self.encoder.get_input_embeddings().weight[token_id]},
            os.path.join(save_dir, 'embed_weight.bin')
        )

        # Save attention processors
        self.unet.save_attn_procs(save_dir)

        print("Training completed and models saved")

    def test_model(self, prompt: str = 'maorongrong cat sitting in a bucket',
                   num_images: int = 4, num_steps: int = 100):
        """Test the trained model"""
        # Create pipeline
        pipeline = DiffusionPipeline.from_pretrained(
            self.checkpoint,
            safety_checker=None
        )

        # Load trained components
        pipeline.load_textual_inversion('./save', weight_name='embed_weight.bin')
        pipeline.unet.load_attn_procs('./save', weight_name='pytorch_custom_diffusion_weights.bin')

        # Move to device
        pipeline = pipeline.to(self.device)

        # Generate images
        images = pipeline(
            prompt,
            num_images_per_prompt=num_images,
            num_inference_steps=num_steps,
            guidance_scale=6.0,
            eta=1.0
        ).images

        # Display images
        plt.figure(figsize=(20, 10))
        for i in range(num_images):
            plt.subplot(1, num_images, i + 1)
            plt.imshow(images[i])
            plt.axis('off')

        plt.show()


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Custom Diffusion Training')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'test', 'setup'],
                        help='Mode: train, test, or setup')
    parser.add_argument('--instance_dir', type=str, default='instance_images',
                        help='Directory containing instance images')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('--save_dir', type=str, default='./save',
                        help='Directory to save trained models')
    parser.add_argument('--checkpoint', type=str, default='CompVis/stable-diffusion-v1-4',
                        help='Pretrained model checkpoint')

    args = parser.parse_args()

    trainer = CustomDiffusionTrainer(checkpoint=args.checkpoint)

    if args.mode == 'setup':
        # Setup components only
        trainer.setup_components()
        trainer.add_custom_token('maorongrong')
        trainer.freeze_models()
        trainer.create_lora_layers()
        trainer.setup_optimizer()
        print("Components setup completed")

    elif args.mode == 'train':
        # Full training pipeline
        trainer.setup_components()
        token_id = trainer.add_custom_token('maorongrong')
        trainer.freeze_models()
        trainer.create_lora_layers()
        trainer.setup_optimizer()

        # Load data
        dataset = trainer.load_negative_dataset()
        data_loader = trainer.create_data_loader(dataset, args.instance_dir)

        # Train
        trainer.train(data_loader, epochs=args.epochs, save_dir=args.save_dir)

    elif args.mode == 'test':
        # Test the trained model
        trainer.test_model()


if __name__ == '__main__':
    main()