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

import os
import argparse
import logging
import time
import math
import gc
from packaging import version

import trimesh
from PIL import Image
import numpy as np
import wandb
from tqdm import tqdm

import torch
import torch.nn.functional as tF
from torch.nn.parallel import DistributedDataParallel
import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger as get_accelerate_logger
from accelerate import DataLoaderConfiguration, DeepSpeedPlugin
from accelerate.optimizer import AcceleratedOptimizer
from accelerate.data_loader import DataLoaderShard
from accelerate.utils import DistributedDataParallelKwargs, GradientAccumulationPlugin
from accelerate.scheduler import AcceleratedScheduler
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3
)


from transformers import (
    BitImageProcessor,
    Dinov2Model,
)
from TripoSG.triposg.schedulers import RectifiedFlowScheduler
from TripoSG.triposg.models.autoencoders import TripoSGVAEModel
from TripoSG.triposg.models.transformers.triposg_transformer_mask import (
    TripoSGDiTModel,
)
from TripoSG.triposg.pipelines.pipeline_triposg import TripoSGPipeline, retrieve_timesteps

from TripoSG.triposg.datasets import (
    ObjaverseMaskDataset,
    BatchedObjaverseMaskDataset,
    MultiEpochsDataLoader, 
    yield_forever
)

from TripoSG.triposg.utils.train_utils import (
    MyEMAModel, 
    get_configs,
    get_optimizer,
    get_lr_scheduler,
    save_experiment_params,
    save_model_architecture,
)
from TripoSG.triposg.utils.render_utils import (
    render_views_around_mesh, 
    render_normal_views_around_mesh, 
    make_grid_for_images_or_videos,
    export_renderings
)
from TripoSG.triposg.utils.metric_utils import compute_cd_and_f_score_in_training
from collections import defaultdict

def main():
    PROJECT_NAME = "TripoSG_4090"

    parser = argparse.ArgumentParser(
        description="Train a diffusion model for 3D object generation",
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the config file"
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Tag that refers to the current experiment"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Path to the output directory"
    )
    parser.add_argument(
        "--resume_from_iter",
        type=int,
        default=None,
        help="The iteration to load the checkpoint from"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the PRNG"
    )
    parser.add_argument(
        "--offline_wandb",
        action="store_true",
        help="Use offline WandB for experiment tracking"
    )

    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="The max iteration step for training"
    )
    parser.add_argument(
        "--max_val_steps",
        type=int,
        default=2,
        help="The max iteration step for validation"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=32,
        help="The number of processed spawned by the batch provider"
    )
    parser.add_argument(
        "--pin_memory",
        action="store_true",
        help="Pin memory for the data loader"
    )

    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Use EMA model for training"
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        help="Scale lr with total batch size (base batch size: 256)"
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.,
        help="Max gradient norm for gradient clipping"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass"
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help="Type of mixed precision training"
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help="Enable TF32 for faster training on Ampere GPUs"
    )

    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.0,
        help="CFG scale used for training"
    )

    parser.add_argument(
        "--val_guidance_scales",
        type=list,
        nargs="+",
        default=[7.0],
        help="CFG scale used for validation"
    )

    parser.add_argument(
        "--use_deepspeed",
        action="store_true",
        help="Use DeepSpeed for training"
    )
    parser.add_argument(
        "--zero_stage",
        type=int,
        default=1,
        choices=[1, 2, 3],  # https://huggingface.co/docs/accelerate/usage_guides/deepspeed
        help="ZeRO stage type for DeepSpeed"
    )

    parser.add_argument(
        "--from_scratch",
        action="store_true",
        help="Train from scratch"
    )
    parser.add_argument(
        "--load_pretrained_model",
        type=str,
        default=None,
        help="Tag of a pretrained TripoSGDiTModel in this project"
    )
    parser.add_argument(
        "--load_pretrained_model_ckpt",
        type=int,
        default=-1,
        help="Iteration of the pretrained TripoSGDiTModel checkpoint"
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="The number of inference steps. This is used to generate the timestep map for the router.",
    )
    
    parser.add_argument(
        "--mlp_mode",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Reuse MLP features in the router."
    )
    parser.add_argument(
        "--self_attn_mode",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Reuse self-attention features in the router."
    )
    parser.add_argument(
        "--cross_attn_mode",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Reuse cross-attention features in the router."
    )

    # Parse the arguments
    args, extras = parser.parse_known_args()
    # Parse the config file
    configs = get_configs(args.config, extras)  # change yaml configs by `extras`
    
    print(f"task mode: mlp: {args.mlp_mode}, self_attn: {args.self_attn_mode}, cross_attn: {args.cross_attn_mode}")
    
    args.val_guidance_scales = [float(x[0]) if isinstance(x, list) else float(x) for x in args.val_guidance_scales]
    if args.max_val_steps > 0: 
        # If enable validation, the max_val_steps must be a multiple of nrow
        # Always keep validation batchsize 1
        divider = configs["val"]["nrow"]
        args.max_val_steps = max(args.max_val_steps, divider)
        if args.max_val_steps % divider != 0:
            args.max_val_steps = (args.max_val_steps // divider + 1) * divider

    # Create an experiment directory using the `tag`
    mode_str = ""
    if args.mlp_mode:
        mode_str += "_mlp"
    if args.self_attn_mode:
        mode_str += "_sa"
    if args.cross_attn_mode:
        mode_str += "_ca"
        
    if args.tag is None:
        args.tag = time.strftime("%Y%m%d_%H_%M_%S")
    if mode_str:
        args.tag += mode_str
    exp_dir = os.path.join(args.output_dir, args.tag)
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    eval_dir = os.path.join(exp_dir, "evaluations")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    # Initialize the logger
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        level=logging.INFO
    )
    logger = get_accelerate_logger(__name__, log_level="INFO")
    file_handler = logging.FileHandler(os.path.join(exp_dir, "log.txt"))  # output to file
    file_handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s - %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S"
    ))
    logger.logger.addHandler(file_handler)
    logger.logger.propagate = True  # propagate to the root logger (console)

    # Set DeepSpeed config
    if args.use_deepspeed:
        deepspeed_plugin = DeepSpeedPlugin(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            gradient_clipping=args.max_grad_norm,
            zero_stage=int(args.zero_stage),
            offload_optimizer_device="cpu",  # hard-coded here, TODO: make it configurable
        )
    else:
        deepspeed_plugin = None

    # Initialize the accelerator
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        project_dir=exp_dir,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        split_batches=False,  # batch size per GPU
        dataloader_config=DataLoaderConfiguration(non_blocking=args.pin_memory),
        deepspeed_plugin=deepspeed_plugin,
        kwargs_handlers=[ddp_kwargs],
    )
    logger.info(f"Accelerator state:\n{accelerator.state}\n")

    # Set the random seed
    if args.seed >= 0:
        accelerate.utils.set_seed(args.seed)
        logger.info(f"You have chosen to seed([{args.seed}]) the experiment [{args.tag}]\n")

    # Enable TF32 for faster training on Ampere GPUs
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    train_dataset = BatchedObjaverseMaskDataset(
        configs=configs,
        batch_size=configs["train"]["batch_size_per_gpu"],
        is_main_process=accelerator.is_main_process,
        shuffle=True,
        training=True,
    )
    val_dataset = ObjaverseMaskDataset(
        configs=configs,
        training=False,
    )
    train_loader = MultiEpochsDataLoader(
        train_dataset,
        batch_size=configs["train"]["batch_size_per_gpu"],
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=args.pin_memory,
        collate_fn=train_dataset.collate_fn,
    )
    val_loader = MultiEpochsDataLoader(
        val_dataset,
        batch_size=configs["val"]["batch_size_per_gpu"],
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=args.pin_memory,
    )
    random_val_loader = MultiEpochsDataLoader(
        val_dataset,
        batch_size=configs["val"]["batch_size_per_gpu"],
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=args.pin_memory,
    )

    logger.info(f"Loaded [{len(train_dataset)}] training samples and [{len(val_dataset)}] validation samples\n")

    # Compute the effective batch size and scale learning rate
    total_batch_size = configs["train"]["batch_size_per_gpu"] * \
        accelerator.num_processes * args.gradient_accumulation_steps
    configs["train"]["total_batch_size"] = total_batch_size
    if args.scale_lr:
        configs["optimizer"]["lr"] *= (total_batch_size / 256)
        configs["lr_scheduler"]["max_lr"] = configs["optimizer"]["lr"]
    
    # Initialize the model
    logger.info("Initializing the model...")
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

    logger.info(f"Load pretrained TripoSGDiTModel to initialize TripoSGDiTModel from [{configs['model']['pretrained_model_name_or_path']}]\n")
    transformer, loading_info = TripoSGDiTModel.from_pretrained(
        configs["model"]["pretrained_model_name_or_path"],
        subfolder="transformer",
        low_cpu_mem_usage=False, 
        output_loading_info=True, )

    for v in loading_info.values():
        if v and len(v) > 0:
            logger.info(f"Loading info of TripoSGDiTModel: {loading_info}\n")
            break

    noise_scheduler = RectifiedFlowScheduler.from_pretrained(
        configs["model"]["pretrained_model_name_or_path"],
        subfolder="scheduler"
    )
    
    timesteps, num_inference_steps = retrieve_timesteps(
            noise_scheduler, num_inference_steps=args.num_inference_steps, device=accelerator.device, timesteps=None
        )
    timestep_map = noise_scheduler.timesteps.tolist()

    # TODO: for debug
    
    transformer.requires_grad_(False)
    
    transformer.eval()
    transformer.add_router(
        args.num_inference_steps, 
        timestep_map,
        mlp_mode=args.mlp_mode,
        self_attn_mode=args.self_attn_mode,
        cross_attn_mode=args.cross_attn_mode,
    )

    if args.use_ema:
        ema_transformer = MyEMAModel(
            transformer.parameters(),
            model_cls=TripoSGDiTModel,
            model_config=transformer.config,
            **configs["train"]["ema_kwargs"]
        )

    # Freeze VAE and image encoder
    vae.requires_grad_(False)
    image_encoder_dinov2.requires_grad_(False)
    vae.eval()
    image_encoder_dinov2.eval()

    # transformer.enable_xformers_memory_efficient_attention()  # use `tF.scaled_dot_product_attention` instead

    if configs["train"]["grad_checkpoint"]:
        transformer.enable_gradient_checkpointing()

    # Initialize the optimizer and learning rate scheduler
    logger.info("Initializing the optimizer and learning rate scheduler...\n")
    
    # Get trainable parameters
    params = [param for name, param in transformer.named_parameters() if "routers" in name]
    # compute the number of parameters that need to be trained
    num_trainable_parameters = sum(param.numel() for param in params)
    num_grad_parameters = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    logger.info(f"Number of trainable parameters: {num_trainable_parameters}\n")
    logger.info(f"Number of grad parameters: {num_grad_parameters}\n")
    
    optimizer = get_optimizer(
        params=[
            {"params": params, "lr": configs["optimizer"]["lr"]}
        ],
        **configs["optimizer"]
    )

    # updated_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    # configs["lr_scheduler"]["total_steps"] = configs["train"]["epochs"] * updated_steps_per_epoch
    # lr_scheduler = get_lr_scheduler(optimizer=optimizer, **configs["lr_scheduler"])

    # Prepare everything with `accelerator`
    transformer, optimizer, train_loader, val_loader, random_val_loader = accelerator.prepare(
        transformer, optimizer, train_loader, val_loader, random_val_loader
    )
    # Set classes explicitly for everything
    transformer: DistributedDataParallel
    optimizer: AcceleratedOptimizer
    # lr_scheduler: AcceleratedScheduler
    train_loader: DataLoaderShard
    val_loader: DataLoaderShard
    random_val_loader: DataLoaderShard

    if args.use_ema:
        ema_transformer.to(accelerator.device)

    # For mixed precision training we cast all non-trainable weigths to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move `vae` and `image_encoder_dinov2` to gpu and cast to `weight_dtype`
    vae.to(accelerator.device, dtype=weight_dtype)
    image_encoder_dinov2.to(accelerator.device, dtype=weight_dtype)

    # Training configs after distribution and accumulation setup
    updated_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    total_updated_steps = configs["train"]["epochs"] * updated_steps_per_epoch
    if args.max_train_steps is None:
        args.max_train_steps = total_updated_steps
    # assert configs["train"]["epochs"] * updated_steps_per_epoch == total_updated_steps
    if accelerator.num_processes > 1 and accelerator.is_main_process:
        print()
    accelerator.wait_for_everyone()
    logger.info(f"Total batch size: [{total_batch_size}]")
    logger.info(f"Learning rate: [{configs['optimizer']['lr']}]")
    logger.info(f"Gradient Accumulation steps: [{args.gradient_accumulation_steps}]")
    logger.info(f"Total epochs: [{configs['train']['epochs']}]")
    logger.info(f"Total steps: [{total_updated_steps}]")
    logger.info(f"Steps for updating per epoch: [{updated_steps_per_epoch}]")
    logger.info(f"Steps for validation: [{len(val_loader)}]\n")

    # (Optional) Load checkpoint
    global_update_step = 0
    if args.resume_from_iter is not None:
        if args.resume_from_iter < 0:
            # find the latest checkpoint
            ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
            if len(ckpts) == 0:
                logger.info("No checkpoint found. Training from scratch.")
                args.resume_from_iter = None
            else:
                latest_ckpt = sorted(ckpts, key=lambda x: int(x.split('.')[0]))[-1]
                args.resume_from_iter = int(latest_ckpt.split('.')[0])
        
        if args.resume_from_iter is not None:
            checkpoint_path = os.path.join(ckpt_dir, f"{args.resume_from_iter:06d}.pt")
            if os.path.exists(checkpoint_path):
                logger.info(f"Load checkpoint from [{checkpoint_path}]\n")
                checkpoint = torch.load(checkpoint_path, map_location="cpu")

                unwrapped_transformer = accelerator.unwrap_model(transformer)
                unwrapped_transformer.routers.load_state_dict(checkpoint["routers"])

                optimizer.load_state_dict(checkpoint["opt"])

                global_update_step = checkpoint.get("global_step", args.resume_from_iter)
            else:
                logger.info(f"Checkpoint [{checkpoint_path}] not found. Training from scratch.")

    # Save all experimental parameters and model architecture of this run to a file (args and configs)
    if accelerator.is_main_process:
        exp_params = save_experiment_params(args, configs, exp_dir)
        save_model_architecture(accelerator.unwrap_model(transformer), exp_dir)

    # WandB logger
    if accelerator.is_main_process:
        if args.offline_wandb:
            os.environ["WANDB_MODE"] = "offline"
        wandb.init(
            project=PROJECT_NAME, name=args.tag,
            config=exp_params, dir=exp_dir,
            resume=True
        )
        # Wandb artifact for logging experiment information
        arti_exp_info = wandb.Artifact(args.tag, type="exp_info")
        arti_exp_info.add_file(os.path.join(exp_dir, "params.yaml"))
        arti_exp_info.add_file(os.path.join(exp_dir, "model.txt"))
        arti_exp_info.add_file(os.path.join(exp_dir, "log.txt"))  # only save the log before training
        wandb.log_artifact(arti_exp_info)

    def get_sigmas(timesteps: Tensor, n_dim: int, dtype=torch.float32):
        sigmas = noise_scheduler.sigmas.to(dtype=dtype, device=accelerator.device)
        schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)

        step_indices = [(schedule_timesteps == t).nonzero()[0].item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    # Start training
    if accelerator.is_main_process:
        print()
    logger.info(f"Start training into {exp_dir}\n")
    logger.logger.propagate = False  # not propagate to the root logger (console)
    progress_bar = tqdm(
        range(total_updated_steps),
        initial=global_update_step,
        desc="Training",
        ncols=125,
        disable=not accelerator.is_main_process
    )
    for batch in yield_forever(train_loader):
        batch_size = batch["images"].shape[0]

        if global_update_step == args.max_train_steps:
            progress_bar.close()
            logger.logger.propagate = True  # propagate to the root logger (console)
            if accelerator.is_main_process:
                wandb.finish()
            logger.info("Training finished!\n")
            return

        transformer.train()
        accelerator.unwrap_model(transformer).reset()

        # prepare image embeddings
        images = batch["images"] # [N, H, W, 3]
        with torch.no_grad():
            images = feature_extractor_dinov2(images=images, return_tensors="pt").pixel_values
        images = images.to(device=accelerator.device, dtype=weight_dtype)
        with torch.no_grad():
            image_embeds = image_encoder_dinov2(images).last_hidden_state

        negative_image_embeds = torch.zeros_like(image_embeds)
        image_embeds = torch.cat([image_embeds, negative_image_embeds], 0)
        
        # set noise_scheduler
        noise_scheduler._step_index = None
        timesteps_for_loop = noise_scheduler.timesteps

        # Prepare latent variables
        num_channels_latents = accelerator.unwrap_model(transformer).config.in_channels
        num_tokens = configs["model"]["vae"]["num_tokens"]
        shape = (batch_size, num_tokens, num_channels_latents)
        # print("shape:", shape) # shape: (n, 2048, 64)
        
        latents_t = torch.randn(shape, device=accelerator.device)
        ori_latents_t = latents_t.clone()
        
        # prepare training loss
        running_data_loss, running_l1_loss = 0.0, 0.0
        log_steps = 0
        
        for i, t_step in enumerate(timesteps_for_loop):
            # setup classifier-free guidance
            latents_t_model_input = torch.cat([latents_t] * 2, 0)
            ori_latents_t_model_input = torch.cat([ori_latents_t] * 2, 0)
        
            with accelerator.accumulate(transformer):
                t = t_step.expand(ori_latents_t_model_input.shape[0])

                with torch.no_grad():
                    noise_pred = transformer(
                        hidden_states=ori_latents_t_model_input,
                        timestep=t,
                        encoder_hidden_states=image_embeds,
                    ).sample
                    
                    # cfg
                    noise_pred_uncond, noise_pred_image = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + args.guidance_scale * (
                        noise_pred_image - noise_pred_uncond
                    )
                    # compute the previous noisy sample x_t -> x_{t-1}
                    # print('noise_scheduler._step_index for ori_latents_t:', noise_scheduler._step_index)
                    current_step_index = noise_scheduler._step_index
                    ori_latents_t = noise_scheduler.step(noise_pred, t_step, ori_latents_t, return_dict=False)[0]
                    noise_scheduler._step_index = current_step_index
                    
            if i % 3 != 0:
                transformer.train() # use the drop out
                noise_output_router_ = transformer(
                    hidden_states=latents_t_model_input,
                    timestep=t,
                    encoder_hidden_states=image_embeds,
                    model_train=True,
                    activate_router = True,
                    # thres=0.5 # A placeholder, might need to be an arg

                )
                noise_pred_router, l1_loss = noise_output_router_
                # print("l1_loss:", l1_loss)
                noise_pred_router = noise_pred_router.sample

            else:
                with torch.no_grad():
                    noise_pred_router = transformer(
                        hidden_states=latents_t_model_input,
                        timestep=t,
                        encoder_hidden_states=image_embeds,
                        model_train=False,
                        activate_router = False,
                        # thres=0.5 # A placeholder, might need to be an arg
                    ).sample
            
            # cfg
            noise_pred_router_uncond, noise_pred_router_image = noise_pred_router.chunk(2)
            noise_pred_router = noise_pred_router_uncond + args.guidance_scale * (
                noise_pred_router_image - noise_pred_router_uncond
            )

            # Now, for the original latents, we do one step with the router
            latents_t = noise_scheduler.step(noise_pred_router, t_step, latents_t, return_dict=False)[0]
                  
            if i % 3 != 0:
                # print router, 大于0.5为1，小于0.5为0
                if accelerator.is_main_process:
                    router_sum = 0
                    # print("router:", accelerator.unwrap_model(transformer).routers)
                    for router in accelerator.unwrap_model(transformer).routers:
                        router_sum += (router.prob > 0.5).sum().item()
                    print("router_sum:", router_sum)
                
                data_loss = tF.mse_loss(ori_latents_t, latents_t)
                loss = configs["train"]["data_loss_weight"] * data_loss + configs["train"]["l1_loss_weight"] * l1_loss

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)
                
                optimizer.step()
                optimizer.zero_grad()
                
                with torch.no_grad():
                    for name, param in transformer.named_parameters():
                        if "routers" in name:
                            param.clamp_(-5, 5)

                    # Detach latents to break the computation graph for the next iteration
                    latents_t = latents_t.detach()
                    ori_latents_t = ori_latents_t.detach()

                    running_data_loss += (configs["train"]["data_loss_weight"] * data_loss).item()
                    running_l1_loss += (configs["train"]["l1_loss_weight"] * l1_loss).item()
                log_steps += 1

        accelerator.unwrap_model(transformer).reset()
        
        # Checks if the accelerator has performed an optimization step behind the scenes
        if accelerator.sync_gradients:
            # Gather the losses across all processes for logging (if we use distributed training)
            
            avg_data_loss = running_data_loss / log_steps
            avg_l1_loss = running_l1_loss / log_steps
            total_loss = avg_data_loss + avg_l1_loss

            logs = {
                "loss": total_loss,
                "data_loss": avg_data_loss,
                "l1_loss": avg_l1_loss,
                "lr": optimizer.param_groups[0]["lr"]
            }
            if args.use_ema:
                ema_transformer.step(transformer.parameters())
                logs.update({"ema": ema_transformer.cur_decay_value})

            progress_bar.set_postfix(**logs)
            progress_bar.update(1)
            global_update_step += 1

            logger.info(
                f"[{global_update_step:06d} / {total_updated_steps:06d}] " +
                f"loss: {logs['loss']:.4f}, data_loss: {logs['data_loss']:.4f}, l1_loss: {logs['l1_loss']:.4f}, lr: {logs['lr']:.2e}" +
                f", ema: {logs['ema']:.4f}" if args.use_ema else ""
            )

            # Log the training progress
            if (
                global_update_step % configs["train"]["log_freq"] == 0 
                or global_update_step == 1
                or global_update_step % updated_steps_per_epoch == 0 # last step of an epoch
            ):  
                if accelerator.is_main_process:
                    wandb.log({
                        "training/loss": logs["loss"],
                        "training/data_loss": logs["data_loss"],
                        "training/l1_loss": logs["l1_loss"],
                        "training/lr": logs["lr"],
                    }, step=global_update_step)
                    if args.use_ema:
                        wandb.log({
                            "training/ema": logs["ema"]
                        }, step=global_update_step)
            
            # Save checkpoint
            if (
                global_update_step % configs["train"]["save_freq"] == 0  # 1. every `save_freq` steps
                or global_update_step % (configs["train"]["save_freq_epoch"] * updated_steps_per_epoch) == 0  # 2. every `save_freq_epoch` epochs
                or global_update_step == total_updated_steps # 3. last step of an epoch
                # or global_update_step == 1 # 4. first step
            ): 

                if accelerator.is_main_process:
                    unwrapped_transformer = accelerator.unwrap_model(transformer)
                    checkpoint = {
                        "routers": unwrapped_transformer.routers.state_dict(),
                        "opt": optimizer.state_dict(),
                        "args": args,
                        "global_step": global_update_step
                    }
                    checkpoint_path = os.path.join(ckpt_dir, f"{global_update_step:06d}.pt")
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

                accelerator.wait_for_everyone()
                gc.collect()

            # Evaluate on the validation set
            if args.max_val_steps > 0 and (
                global_update_step % configs["train"]["eval_freq"] == 0
                or global_update_step == total_updated_steps
                or global_update_step == 1
            ):  

                # Use EMA parameters for evaluation
                if args.use_ema:
                    # Store the Transformer parameters temporarily and load the EMA parameters to perform inference
                    ema_transformer.store(transformer.parameters())
                    ema_transformer.copy_to(transformer.parameters())

                if args.use_ema:
                    # Switch back to the original Transformer parameters
                    ema_transformer.restore(transformer.parameters())

                torch.cuda.empty_cache()
                gc.collect()



if __name__ == "__main__":
    main()
    
    
'''
export NCCL_P2P_DISABLE=1 # for 4090
export NCCL_IB_DISABLE=1 # for 4090
export NCCL_SOCKET_NTHREADS=1 # for 4090

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python TripoSG/scripts/train/train_triposg_mask.py \
    --config configs/mp8_nt2048.yaml --use_ema --gradient_accumulation_steps 4 \
        --output_dir output_mask --tag scaleup_mp8_nt512_mask 
          
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 CUDA_VISIBLE_DEVICES=4,5,6,7\
    accelerate launch --multi_gpu --num_processes 4 \
    TripoSG/scripts/train/train_triposg_mask.py \
    --config configs/mp8_nt2048.yaml \
    --use_ema \
    --gradient_accumulation_steps 4 \
    --output_dir output_mask \
    --tag scaleup_mp8_nt512_mask
'''