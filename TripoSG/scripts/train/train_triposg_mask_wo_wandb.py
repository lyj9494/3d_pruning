import warnings
warnings.filterwarnings("ignore")  # ignore all warnings
import diffusers.utils.logging as diffusion_logging
diffusion_logging.set_verbosity_error()  # ignore diffusers warnings

import os
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))
print(str(project_root))

from TripoSG.triposg.utils.typing_utils import *


import argparse
import logging
import time
import math
import gc
from packaging import version

import trimesh
from PIL import Image
import numpy as np
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

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_SOCKET_NTHREADS"] = "1"
os.environ["LD_PRELOAD"] = "/usr/lib/x86_64-linux-gnu/libstdc++.so.6"

def main():
    PROJECT_NAME = "TripoSG"

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

    # Parse the arguments
    args, extras = parser.parse_known_args()
    # Parse the config file
    configs = get_configs(args.config, extras)  # change yaml configs by `extras`

    args.val_guidance_scales = [float(x[0]) if isinstance(x, list) else float(x) for x in args.val_guidance_scales]
    if args.max_val_steps > 0: 
        # If enable validation, the max_val_steps must be a multiple of nrow
        # Always keep validation batchsize 1
        divider = configs["val"]["nrow"]
        args.max_val_steps = max(args.max_val_steps, divider)
        if args.max_val_steps % divider != 0:
            args.max_val_steps = (args.max_val_steps // divider + 1) * divider

    # Create an experiment directory using the `tag`
    if args.tag is None:
        args.tag = time.strftime("%Y%m%d_%H_%M_%S")
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

    # print(f"timestep_map: {timestep_map}")
    # Freeze all parameters except routers
    transformer.requires_grad_(False)
    transformer.eval()
    transformer.add_router(args.num_inference_steps, timestep_map)

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

    
    for name, param in transformer.named_parameters():
        if "routers" in name:
            param.requires_grad = True

    trainable_modules = configs["train"].get("trainable_modules", None)
    if trainable_modules is None:
        pass
        # transformer.requires_grad_(True)
    else:
        trainable_module_names = []
        transformer.requires_grad_(False)
        for name, module in transformer.named_modules():
            for module_name in tuple(trainable_modules.split(",")):
                if module_name in name:
                    for params in module.parameters():
                        params.requires_grad = True
                    trainable_module_names.append(name)
        logger.info(f"Trainable parameter names: {trainable_module_names}\n")
    # the number of parameters that need to be trained
    num_trainable_parameters = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    logger.info(f"Number of trainable parameters: {num_trainable_parameters}\n")
    
    # transformer.enable_xformers_memory_efficient_attention()  # use `tF.scaled_dot_product_attention` instead

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # Create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_transformer.save_pretrained(os.path.join(output_dir, "transformer_ema"))

                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "transformer"))

                    # Make sure to pop weight so that corresponding model is not saved again
                    if weights:
                        weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = MyEMAModel.from_pretrained(os.path.join(input_dir, "transformer_ema"), TripoSGDiTModel)
                ema_transformer.load_state_dict(load_model.state_dict())
                ema_transformer.to(accelerator.device)
                del load_model

            for _ in range(len(models)):
                # Pop models so that they are not loaded again
                model = models.pop()

                # Load diffusers style into model
                load_model = TripoSGDiTModel.from_pretrained(input_dir, subfolder="transformer")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if configs["train"]["grad_checkpoint"]:
        transformer.enable_gradient_checkpointing()

    # Initialize the optimizer and learning rate scheduler
    logger.info("Initializing the optimizer and learning rate scheduler...\n")
    
    # Get trainable parameters
    params = [p for p in transformer.parameters() if p.requires_grad]
    # compute the number of parameters that need to be trained
    num_trainable_parameters = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    logger.info(f"Number of trainable parameters: {num_trainable_parameters}\n")
    
    optimizer = get_optimizer(
        params=[
            {"params": params, "lr": configs["optimizer"]["lr"]}
        ],
        **configs["optimizer"]
    )
    print("optimizer:", optimizer)
    
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
            args.resume_from_iter = int(sorted(os.listdir(ckpt_dir))[-1])
        logger.info(f"Load checkpoint from iteration [{args.resume_from_iter}]\n")
        # Load everything
        if version.parse(torch.__version__) >= version.parse("2.4.0"):
            torch.serialization.add_safe_globals([
                int, list, dict, 
                defaultdict,
                Any,
                DictConfig, ListConfig, Metadata, ContainerMetadata, AnyNode
            ]) # avoid deserialization error when loading optimizer state
        accelerator.load_state(os.path.join(ckpt_dir, f"{args.resume_from_iter:06d}"))  # torch < 2.4.0 here for `weights_only=False`
        global_update_step = int(args.resume_from_iter)

    # Save all experimental parameters and model architecture of this run to a file (args and configs)
    if accelerator.is_main_process:
        exp_params = save_experiment_params(args, configs, exp_dir)
        save_model_architecture(accelerator.unwrap_model(transformer), exp_dir)

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
                print("l1_loss:", l1_loss)
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
            
            # Save checkpoint
            if (
                global_update_step % configs["train"]["save_freq"] == 0  # 1. every `save_freq` steps
                or global_update_step % (configs["train"]["save_freq_epoch"] * updated_steps_per_epoch) == 0  # 2. every `save_freq_epoch` epochs
                or global_update_step == total_updated_steps # 3. last step of an epoch
                # or global_update_step == 1 # 4. first step
            ): 

                gc.collect()
                if accelerator.distributed_type == accelerate.utils.DistributedType.DEEPSPEED:
                    # DeepSpeed requires saving weights on every device; saving weights only on the main process would cause issues
                    accelerator.save_state(os.path.join(ckpt_dir, f"{global_update_step:06d}"))
                elif accelerator.is_main_process:
                    accelerator.save_state(os.path.join(ckpt_dir, f"{global_update_step:06d}"))
                accelerator.wait_for_everyone()  # ensure all processes have finished saving
                gc.collect()

            # Evaluate on the validation set
            if args.max_val_steps > 0 and (
                (global_update_step % configs["train"]["early_eval_freq"] == 0 and global_update_step < configs["train"]["early_eval"])  # 1. more frequently at the beginning
                or global_update_step % configs["train"]["eval_freq"] == 0  # 2. every `eval_freq` steps
                or global_update_step % (configs["train"]["eval_freq_epoch"] * updated_steps_per_epoch) == 0  # 3. every `eval_freq_epoch` epochs
                or global_update_step == total_updated_steps # 4. last step of an epoch
                or global_update_step == 1 # 5. first step
            ):  

                # Use EMA parameters for evaluation
                if args.use_ema:
                    # Store the Transformer parameters temporarily and load the EMA parameters to perform inference
                    ema_transformer.store(transformer.parameters())
                    ema_transformer.copy_to(transformer.parameters())

                transformer.eval()

                log_validation(
                    val_loader, random_val_loader,
                    feature_extractor_dinov2, image_encoder_dinov2,
                    vae, transformer,
                    global_update_step, eval_dir,
                    accelerator, logger,
                    args, configs
                )

                if args.use_ema:
                    # Switch back to the original Transformer parameters
                    ema_transformer.restore(transformer.parameters())

                torch.cuda.empty_cache()
                gc.collect()

@torch.no_grad()
def log_validation(
    dataloader, random_dataloader,
    feature_extractor_dinov2, image_encoder_dinov2,
    vae, transformer, 
    global_step, eval_dir,
    accelerator, logger,  
    args, configs
):  

    val_noise_scheduler = RectifiedFlowScheduler.from_pretrained(
        configs["model"]["pretrained_model_name_or_path"],
        subfolder="scheduler"
    )

    pipeline = TripoSGPipeline(
        vae=vae,
        transformer=accelerator.unwrap_model(transformer),
        scheduler=val_noise_scheduler,
        feature_extractor_dinov2=feature_extractor_dinov2,
        image_encoder_dinov2=image_encoder_dinov2,
    )

    pipeline.set_progress_bar_config(disable=True)
    # pipeline.enable_xformers_memory_efficient_attention()

    if args.seed >= 0:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
    else:
        generator = None
        

    val_progress_bar = tqdm(
        range(len(dataloader)) if args.max_val_steps is None else range(args.max_val_steps),
        desc=f"Validation [{global_step:06d}]",
        ncols=125,
        disable=not accelerator.is_main_process
    )

    medias_dictlist, metrics_dictlist = defaultdict(list), defaultdict(list)

    val_dataloder, random_val_dataloader = yield_forever(dataloader), yield_forever(random_dataloader)
    val_step = 0
    while val_step < args.max_val_steps:

        if val_step < args.max_val_steps // 2:
            # fix the first half
            batch = next(val_dataloder)
        else:
            # randomly sample the next batch
            batch = next(random_val_dataloader)

        images = batch["images"]
        if len(images.shape) == 5:
            images = images[0] # (1, N, H, W, 3) -> (N, H, W, 3)
        images = [Image.fromarray(image) for image in images.cpu().numpy()]
        surfaces = batch["surfaces"].cpu().numpy()
        if len(surfaces.shape) == 4:
            surfaces = surfaces[0] # (1, N, P, 6) -> (N, P, 6)

        N = len(images)

        val_progress_bar.set_postfix(
            {"num_objects": N}
        )

        with torch.autocast("cuda", torch.float16):
            for guidance_scale in sorted(args.val_guidance_scales):
                pred_meshes = pipeline(
                    images, 
                    num_inference_steps=configs['val']['num_inference_steps'],
                    num_tokens=configs['model']['vae']['num_tokens'],
                    guidance_scale=guidance_scale, 
                    generator=generator,
                    # max_num_expanded_coords=configs['val']['max_num_expanded_coords'],
                    use_flash_decoder=configs['val']['use_flash_decoder'],
                ).meshes

                # Save the generated meshes
                if accelerator.is_main_process:
                    local_eval_dir = os.path.join(eval_dir, f"{global_step:06d}", f"guidance_scale_{guidance_scale:.1f}")
                    os.makedirs(local_eval_dir, exist_ok=True)
                    rendered_images_list, rendered_normals_list = [], []
                    # 1. save the gt image
                    images[0].save(os.path.join(local_eval_dir, f"{val_step:04d}.png"))
                    # 2. randomly save the generated meshes
                    import random
                    n = random.randint(0, N - 1)
                    
                    if pred_meshes[n] is None:
                        # If the generated mesh is None (decoing error), use a dummy mesh
                        pred_meshes[n] = trimesh.Trimesh(vertices=[[0, 0, 0]], faces=[[0, 0, 0]])
                    pred_meshes[n].export(os.path.join(local_eval_dir, f"{val_step:04d}_{n:02d}.glb"))
                    # 3. render the generated mesh and save the rendered images
                    
                    rendered_images: List[Image.Image] = render_views_around_mesh(
                        pred_meshes[n], 
                        num_views=configs['val']['rendering']['num_views'],
                        radius=configs['val']['rendering']['radius'],
                    )
                    rendered_normals: List[Image.Image] = render_normal_views_around_mesh(
                        pred_meshes[n],
                        num_views=configs['val']['rendering']['num_views'],
                        radius=configs['val']['rendering']['radius'],
                    )
                    export_renderings(
                        rendered_images,
                        os.path.join(local_eval_dir, f"{val_step:04d}.gif"),
                        fps=configs['val']['rendering']['fps']
                    )
                    export_renderings(
                        rendered_normals,
                        os.path.join(local_eval_dir, f"{val_step:04d}_normals.gif"),
                        fps=configs['val']['rendering']['fps']
                    )
                    rendered_images_list.append(rendered_images)
                    rendered_normals_list.append(rendered_normals)

                    medias_dictlist[f"guidance_scale_{guidance_scale:.1f}/gt_image"] += [images[0]] # List[Image.Image] TODO: support batch size > 1
                    medias_dictlist[f"guidance_scale_{guidance_scale:.1f}/pred_rendered_images"] += rendered_images_list # List[List[Image.Image]]
                    medias_dictlist[f"guidance_scale_{guidance_scale:.1f}/pred_rendered_normals"] += rendered_normals_list # List[List[Image.Image]]

                ################################ Compute generation metrics ################################

                chamfer_distances, f_scores = [], []

                for n in range(N):
                    gt_surface = surfaces[n]
                    pred_mesh = pred_meshes[n]
                    if pred_mesh is None:
                        # If the generated mesh is None (decoing error), use a dummy mesh
                        pred_mesh = trimesh.Trimesh(vertices=[[0, 0, 0]], faces=[[0, 0, 0]])
                    cd, f_score = compute_cd_and_f_score_in_training(
                        gt_surface, pred_mesh,
                        num_samples=configs['val']['metric']['cd_num_samples'],
                        threshold=configs['val']['metric']['f1_score_threshold'],
                        metric=configs['val']['metric']['cd_metric']
                    )
                    # avoid nan
                    cd = configs['val']['metric']['default_cd'] if np.isnan(cd) else cd
                    f_score = configs['val']['metric']['default_f1'] if np.isnan(f_score) else f_score
                    chamfer_distances.append(cd)
                    f_scores.append(f_score)


                chamfer_distances = torch.tensor(chamfer_distances, device=accelerator.device)
                f_scores = torch.tensor(f_scores, device=accelerator.device)

                metrics_dictlist[f"chamfer_distance_cfg{guidance_scale:.1f}"].append(chamfer_distances.mean())
                metrics_dictlist[f"f_score_cfg{guidance_scale:.1f}"].append(f_scores.mean())
            
        # Only log the last (biggest) cfg metrics in the progress bar
        val_logs = {
            "chamfer_distance": chamfer_distances.mean().item(),
            "f_score": f_scores.mean().item(),
        }
        val_progress_bar.set_postfix(**val_logs)
        logger.info(
            f"Validation [{val_step:02d}/{args.max_val_steps:02d}] " +
            f"chamfer_distance: {val_logs['chamfer_distance']:.4f}, f_score: {val_logs['f_score']:.4f}"
        )
        logger.info(
            f"chamfer_distances: {[f'{x:.4f}' for x in chamfer_distances.tolist()]}"
        )
        logger.info(
            f"f_scores: {[f'{x:.4f}' for x in f_scores.tolist()]}"
        )
        val_step += 1
        val_progress_bar.update(1)

    val_progress_bar.close()

    if accelerator.is_main_process:
        for key, value in medias_dictlist.items():
            if isinstance(value[0], Image.Image): # assuming gt_image
                image_grid = make_grid_for_images_or_videos(
                    value, 
                    nrow=configs['val']['nrow'],
                    return_type='pil', 
                )
                image_grid.save(os.path.join(eval_dir, f"{global_step:06d}", f"{key.replace('/', '_')}.png"))
            else: # assuming pred_rendered_images or pred_rendered_normals
                image_grids = make_grid_for_images_or_videos(
                    value, 
                    nrow=configs['val']['nrow'],
                    return_type='ndarray',
                )
                image_grids = [Image.fromarray(image_grid.transpose(1, 2, 0)) for image_grid in image_grids]
                export_renderings(
                    image_grids, 
                    os.path.join(eval_dir, f"{global_step:06d}", f"{key.replace('/', '_')}.gif"), 
                    fps=configs['val']['rendering']['fps']
                )

if __name__ == "__main__":
    main()
    
    
'''
export NCCL_P2P_DISABLE=1 # for 4090
export NCCL_SOCKET_NTHREADS=1 # for 4090

LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 CUDA_VISIBLE_DEVICES=2\
    python TripoSG/scripts/train/train_triposg_mask.py \
    --config configs/mp8_nt2048.yaml --use_ema --gradient_accumulation_steps 4 \
        --output_dir output_partcrafter --tag scaleup_mp8_nt512_test 
        
        
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 CUDA_VISIBLE_DEVICES=1,2\
    accelerate launch --multi_gpu --num_processes 2 \
    TripoSG/scripts/train/train_triposg_mask.py \
    --config configs/mp8_nt2048.yaml \
    --use_ema \
    --gradient_accumulation_steps 4 \
    --output_dir output_partcrafter \
    --tag scaleup_mp8_nt512_test 
'''