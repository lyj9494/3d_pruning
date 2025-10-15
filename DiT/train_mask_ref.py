# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from download import find_model
from glob import glob

import time
import math
import argparse
import numpy as np
import logging
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from models.router_models import DiT_models, STE

def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


def main(args):
    # Setup an experiment folder:
    os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
    experiment_index = len(glob(f"{args.results_dir}/*"))
    model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
    experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
    checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger = create_logger(experiment_dir)
    logger.info(f"Experiment directory created at {experiment_dir}")
    # Log important run configuration including ste-threshold so it's recorded in the logfile
    logger.info(f"--ste-threshold: {args.ste_threshold}")
    # Also log full args for reproducibility
    try:
        logger.info(f"Command-line args: {args}")
    except Exception:
        # Fallback in case args contains non-serializable objects
        logger.info("Command-line args logged (unable to stringify).")
    
    # Setup PyTorch:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    # initialize diffusin process
    diffusion = create_diffusion(str(args.num_sampling_steps))  

    # Load model:
    latent_size = args.image_size // 8

    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)
    
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!

    model.add_router(args.num_sampling_steps, diffusion.timestep_map)

    # torch.manual_seed(args.seed)
    opts = torch.optim.AdamW(
        [param for name, param in model.named_parameters() if "routers" in name], 
        lr=args.lr, weight_decay=0
    )

    if args.wandb:
        try:
            import importlib
            wandb = importlib.import_module("wandb")
        except Exception:
            logger.warning("wandb not available -- continuing without wandb logging")
        else:
            wandb.init(
                project="DiT-Mask",
                name=f"{experiment_index:03d}-{model_string_name}",
                config=args.__dict__,
            )
            wandb.define_metric("step")
            wandb.define_metric("loss", step_metric="step")

    # train mask
    model.train()

    min_loss1 = 1
    min_batch = 0

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    running_data_loss, running_l1_loss = 0, 0

    for step in range(1000):
        # Create sampling noise:

        n = 1 # batch size 1 4 8
        x_t = torch.randn(n, 4, latent_size, latent_size, device=device)

        ori_xt = x_t.clone()

        y = torch.randint(0,1000,(n,)).to(device) # y is randomly sampled

        # Setup classifier-free guidance:
        x_t = torch.cat([x_t, x_t], 0)
        ori_xt = torch.cat([ori_xt, ori_xt], 0)

        y_null = torch.tensor([1000] * n, device=device)
        y = torch.cat([y, y_null], 0)
        model_kwargs = dict(y=y, thres=args.ste_threshold)

        indices = list(range(args.num_sampling_steps))[::-1]
        for i in indices:

            t = torch.tensor([i] * x_t.shape[0], device=device)
            dict1 = diffusion.mask_training_losses(model, x_t, ori_xt, t, model_kwargs)
            
            x_t = dict1["x_next"]
            ori_xt = dict1["ori_x_next"]
            
            if i % 3 != 0: # i % 2 == 0
                data_loss = dict1["mse"].mean()
                l1_loss = dict1["l1_loss"].mean()

                # loss_scale = math.sqrt(args.num_sampling_steps - i)
                #loss_scale = (args.num_sampling_steps - i) / args.num_sampling_steps
                # if i >= 16:
                #     loss_scale = 1.5 * loss_scale
                # elif i >= 12:
                #     loss_scale = 1.1 * loss_scale

                # if i >= args.num_sampling_steps*2/5:
                #     loss_scale = 2
                # else:
                #     loss_scale = 0.5
                # if i >= args.num_sampling_steps*1/2:
                #     loss_scale = 2
                # else:
                #     loss_scale = 0.5
                loss_scale = 1
                loss = data_loss + args.l1 * l1_loss * loss_scale
                opts.zero_grad()
                model.zero_grad()

                loss.backward()
                # loss.backward(retain_graph=True)
                opts.step()

                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if "routers" in name:
                            param.clamp_(-5, 5)

                # Log loss values:
                running_loss += loss.item()
                running_data_loss += data_loss.item()
                running_l1_loss += args.l1 * l1_loss.item() * loss_scale

                log_steps += 1
                train_steps += 1

                # # 遍历这些参数并打印它们的梯度
                # p = opts.param_groups[0]['params'][i]
                # print(p, p.grad)

                if train_steps % args.log_every == 0: 
                    scores = [model.routers[idx]() for idx in range(0, args.num_sampling_steps, 2)]
                    final_score = [sum(score) for score in scores]
                    logger.info(f"train_steps: {train_steps}, data_loss: {running_data_loss / log_steps}, l1_loss: {running_l1_loss / log_steps}, non_zero: {sum(final_score)}")
                    # if args.wandb:
                    #     #print(scores)
                    #     if args.ste_threshold is not None:
                    #         final_score = [sum(STE.apply(score, args.ste_threshold)) for score in scores]
                    #     else:
                    #         final_score = [sum(score) for score in scores]
                    #     wandb.log({
                    #         "train_steps": train_steps,
                    #         "data_loss": running_data_loss / log_steps,
                    #         "l1_loss": running_l1_loss / log_steps,
                    #         "non_zero": sum(final_score),
                    #     })

                    # Reset monitoring variables:
                    running_loss = 0
                    running_data_loss, running_l1_loss = 0, 0
                    log_steps = 0

        # raise ValueError
        model.reset()

        # Save DiT checkpoint:
        if (step+1) % 100 == 0:
            # include ste_threshold explicitly in the checkpoint dict and filename
            ste_val = args.ste_threshold
            ste_str = "None" if ste_val is None else str(ste_val).replace('.', 'p')
            checkpoint = {
                "routers": model.routers.state_dict(),
                "opt": opts.state,
                "args": args,
                "ste_threshold": ste_val,
            }
            checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}_ste{ste_str}.pt"
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path} (ste_threshold={ste_val})")

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")

            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="DiT-XL/2")
    parser.add_argument("--results-dir", type=str, default="test_results")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=512) #
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--num-sampling-steps", type=int, default=50) #
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default="/workspace/luoyajing/DiT/pretrained_models/DiT-XL-2-512x512.pt",
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--wandb", default=False) #action="store_true"
    
    parser.add_argument("--ddim-sample", action="store_true", default=False,)
    parser.add_argument("--p-sample", action="store_true", default=False,)
    parser.add_argument("--l1", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1.0)

    parser.add_argument("--ste-threshold", type=float, default=None)

    args = parser.parse_args()
    main(args)
