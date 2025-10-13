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
from diffusers.models import AutoencoderKL
from download import find_model

import argparse
import numpy as np


def main(args):
    # Setup PyTorch:
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    # initialize diffusin process
    diffusion = create_diffusion(str(args.num_sampling_steps))  

    # Load model:
    latent_size = args.image_size // 8
    if args.accelerate_method is not None and args.accelerate_method == "dynamiclayer":
        from models.dynamic_models import DiT_models
    else:
        from models.models import DiT_models

    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)

    if args.accelerate_method is not None and 'dynamiclayer' in args.accelerate_method:
        model.load_ranking(args.path, args.num_sampling_steps, diffusion.timestep_map, args.thres)
    
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!

    torch.manual_seed(args.seed)
    # Labels to condition the model with (feel free to change):
    class_labels = [207, 992, 387, 974, 142, 979, 417, 279]
    # 207, 992, 387, 974, 142, 979, 417, 279

    # Create sampling noise:
    n = len(class_labels)
    x_t = torch.randn(n, 4, latent_size, latent_size, device=device)
    ori_xt = x_t.clone()

    y = torch.tensor(class_labels, device=device)

    # Setup classifier-free guidance:
    x_t = torch.cat([x_t, x_t], 0)
    ori_xt = torch.cat([ori_xt, ori_xt], 0)

    y_null = torch.tensor([1000] * n, device=device)
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y)

    indices = list(range(args.num_sampling_steps))[::-1]
    for i in indices:
        t = torch.tensor([i] * x_t.shape[0], device=device)

        with torch.no_grad():
            model_kwargs['ori'] = True
            ori_x_next = diffusion.ddim_sample(
                model, ori_xt, t, 
                clip_denoised=False, denoised_fn=None, cond_fn=None, 
                model_kwargs=model_kwargs, eta=0.0
            )['sample']

            model_kwargs['ori'] = False
            x_next = diffusion.ddim_sample(
                model, x_t, t, 
                clip_denoised=False, denoised_fn=None, cond_fn=None, 
                model_kwargs=model_kwargs, eta=0.0
            )['sample']

        if i % 2 == 0:
            data_loss = (((ori_x_next - x_next) ** 2).mean(dim=list(range(1, len(((ori_x_next - x_next) ** 2).shape))))).mean()
            print(data_loss)

        x_t = x_next
        ori_xt = ori_x_next

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=512)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default="/workspace/luoyajing/DiT/pretrained_models/DiT-XL-2-512x512.pt",
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--accelerate-method", type=str, default="dynamiclayer",
                        help="Use the accelerated version of the model.")
    
    parser.add_argument("--ddim-sample", action="store_true", default=False,)
    parser.add_argument("--p-sample", action="store_true", default=False,)
    
    parser.add_argument("--path", type=str, default=None,
                        help="Optional path to a router checkpoint")
    parser.add_argument("--thres", type=float, default=0.5)

    args = parser.parse_args()
    main(args)
