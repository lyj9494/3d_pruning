import warnings
warnings.filterwarnings("ignore")  # ignore all warnings
import diffusers.utils.logging as diffusion_logging
diffusion_logging.set_verbosity_error()  # ignore diffusers warnings

import os
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from TripoSG.triposg.utils.typing_utils import *

import argparse
import logging
from tqdm import tqdm

import torch
import torch.nn.functional as tF
import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger as get_accelerate_logger
from accelerate.utils import set_seed

from transformers import (
    BitImageProcessor,
    Dinov2Model,
)
from TripoSG.triposg.schedulers import RectifiedFlowScheduler
from TripoSG.triposg.models.autoencoders import TripoSGVAEModel
from TripoSG.triposg.models.transformers.triposg_transformer_dynamic import TripoSGDiTModel
from TripoSG.triposg.pipelines.pipeline_triposg import retrieve_timesteps

from TripoSG.triposg.datasets import (
    ObjaverseMaskDataset,
    MultiEpochsDataLoader, 
    yield_forever
)

from TripoSG.triposg.utils.train_utils import (
    get_configs,
)

def main():
    parser = argparse.ArgumentParser(
        description="Test a diffusion model for 3D object generation with dynamic layer skipping",
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the config file"
    )
    parser.add_argument(
        "--router_ckpt", type=str, required=True,
        help="Optional path to a router checkpoint"
    )
    parser.add_argument(
        "--thres", type=float, default=0.5,
        help="Threshold for STE function"
    )
    parser.add_argument(
        "--num_sampling_steps", type=int, default=50,
        help="Number of sampling steps"
    )
    parser.add_argument(
        "--guidance_scale", type=float, default=7.0,
        help="Classifier-free guidance scale"
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Seed for the PRNG"
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help="Type of mixed precision training"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="The number of processed spawned by the batch provider"
    )

    parser.add_argument(
        "--mlp_mode",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Enable MLP mode."
    )
    parser.add_argument(
        "--self_attn_mode",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Enable self-attention mode."
    )
    parser.add_argument(
        "--cross_attn_mode",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Enable cross-attention mode."
    )

    args, extras = parser.parse_known_args()
    configs = get_configs(args.config, extras)

    # Initialize accelerator
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    logger = get_accelerate_logger(__name__, log_level="INFO")
    
    if args.seed is not None:
        set_seed(args.seed)

    # Load models
    logger.info("Initializing models...")
    vae = TripoSGVAEModel.from_pretrained(
        configs["model"]["pretrained_model_name_or_path"],
        subfolder="vae"
    )
    feature_extractor_dinov2 = BitImageProcessor.from_pretrained(
        configs["model"]["pretrained_model_name_or_path"],
        subfolder="feature_extractor_dinov2"
    )
    image_encoder_dinov2 = Dinov2Model.from_pretrained(
        configs["model"]["pretrained_model_name_or_path"],
        subfolder="image_encoder_dinov2"
    )
    transformer = TripoSGDiTModel.from_pretrained(
        configs["model"]["pretrained_model_name_or_path"],
        subfolder="transformer",
    )
    noise_scheduler = RectifiedFlowScheduler.from_pretrained(
        configs["model"]["pretrained_model_name_or_path"],
        subfolder="scheduler"
    )

    # Prepare models
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype).eval()
    image_encoder_dinov2.to(accelerator.device, dtype=weight_dtype).eval()
    transformer.to(accelerator.device).eval() # Keep transformer in fp32 for stability

    # Load dataset
    val_dataset = ObjaverseMaskDataset(configs=configs, training=False)
    val_loader = MultiEpochsDataLoader(
        val_dataset,
        batch_size=1, # One object at a time for testing
        num_workers=args.num_workers,
        drop_last=True,
    )
    val_loader = accelerator.prepare(val_loader)
    
    # Load ranking from router checkpoint
    timesteps, num_inference_steps = retrieve_timesteps(
        noise_scheduler, num_inference_steps=args.num_sampling_steps, device=accelerator.device, timesteps=None
    )
    timestep_map = noise_scheduler.timesteps.tolist()
    transformer.load_ranking(
        args.router_ckpt, 
        args.num_sampling_steps, 
        timestep_map, 
        args.thres,
        mlp_mode=args.mlp_mode,
        self_attn_mode=args.self_attn_mode,
        cross_attn_mode=args.cross_attn_mode,
    )

    total_mse = 0
    num_batches = 0

    val_iterator = iter(val_loader)

    for i in range(len(val_loader)):
        batch = next(val_iterator)
        
        # Prepare inputs
        images = batch["images"]
        with torch.no_grad():
            processed_images = feature_extractor_dinov2(images=images, return_tensors="pt").pixel_values
        processed_images = processed_images.to(device=accelerator.device, dtype=weight_dtype)
        with torch.no_grad():
            image_embeds = image_encoder_dinov2(processed_images).last_hidden_state
        
        # Classifier-free guidance setup
        negative_image_embeds = torch.zeros_like(image_embeds)
        image_embeds_cfg = torch.cat([image_embeds, negative_image_embeds], 0)

        # Initial latents
        num_channels_latents = transformer.config.in_channels
        num_tokens = configs["model"]["vae"]["num_tokens"]
        shape = (images.shape[0], num_tokens, num_channels_latents)
        
        latents = torch.randn(shape, device=accelerator.device, generator=torch.manual_seed(args.seed) if args.seed is not None else None)
        ori_latents = latents.clone()

        noise_scheduler.set_timesteps(args.num_sampling_steps)
        
        logger.info(f"Processing batch {i+1}/{len(val_loader)}")
        batch_mse = 0
        
        transformer.reset()
        for t in tqdm(noise_scheduler.timesteps, desc=f"Batch {i+1} Sampling"):
            # CFG input duplication
            latent_model_input = torch.cat([latents] * 2)
            ori_latent_model_input = torch.cat([ori_latents] * 2)

            t_in = t.expand(latent_model_input.shape[0])

            with torch.no_grad():
                # Original model prediction
                ori_noise_pred = transformer(
                    ori_latent_model_input, timestep=t_in, encoder_hidden_states=image_embeds_cfg, ori=True
                ).sample
                
                # Dynamic model prediction
                noise_pred = transformer(
                    latent_model_input, timestep=t_in, encoder_hidden_states=image_embeds_cfg, ori=False
                ).sample

            # CFG guidance
            ori_noise_uncond, ori_noise_cond = ori_noise_pred.chunk(2)
            ori_noise_pred = ori_noise_uncond + args.guidance_scale * (ori_noise_cond - ori_noise_uncond)

            noise_uncond, noise_cond = noise_pred.chunk(2)
            noise_pred = noise_uncond + args.guidance_scale * (noise_cond - noise_uncond)

            # Scheduler step
            ori_latents = noise_scheduler.step(ori_noise_pred, t, ori_latents).prev_sample
            latents = noise_scheduler.step(noise_pred, t, latents).prev_sample
            
            # MSE calculation
            mse = tF.mse_loss(ori_latents, latents).item()
            batch_mse += mse
        
        avg_batch_mse = batch_mse / len(noise_scheduler.timesteps)
        logger.info(f"Batch {i+1} Average MSE: {avg_batch_mse:.6f}")
        total_mse += avg_batch_mse
        num_batches += 1

    final_avg_mse = total_mse / num_batches
    logger.info(f"Final Average MSE across all batches: {final_avg_mse:.6f}")

if __name__ == "__main__":
    main()

'''
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_SOCKET_NTHREADS=1 CUDA_VISIBLE_DEVICES=5 \
python TripoSG/scripts/eval/eval_triposg_mask.py \
    --config configs/mp8_nt2048.yaml \
    --router_ckpt /data1/luoyajing/3d_pruning/output_mask/scaleup_mp8_nt512_mask_mlp_sa_ca/checkpoints/000002.pt \
    --thres 0.5 \
    --num_sampling_steps 50 \
    --guidance_scale 7.0 \
    --seed 0 \
    --mixed_precision fp16 \
    --num_workers 8
'''