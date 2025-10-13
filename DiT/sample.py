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
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    torch.manual_seed(args.seed)
    # Labels to condition the model with (feel free to change):
    import random

    # 设置随机种子以确保结果的可重复性（可选）
    random.seed(10)

    # 在0到1000（包含0，不包含1001）的范围内随机生成32个不重复的索引
    random_indices = random.sample(range(1000), 16)

    # 将生成的随机索引转换成列表
    class_labels = list(random_indices)
    # class_labels = [635,155,283,325,575,628,579,950] # [207, 992, 387, 974, 142, 979, 417, 279] 十几的鸟不能选 企鹅不行
    # 950 635
    # [635,155,283,325,575,628,579,950]
    # Create sampling noise:
    n = len(class_labels)
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    y = torch.tensor(class_labels, device=device)

    # Setup classifier-free guidance:
    z = torch.cat([z, z], 0)
    y_null = torch.tensor([1000] * n, device=device)
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

    # Sample images:
    import time
    times = []
    for _ in range(1):
        start_time = time.time()
        if args.p_sample:
            samples = diffusion.p_sample_loop(
                model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
            )
        elif args.ddim_sample:
            samples = diffusion.ddim_sample_loop(
                model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
            )
        times.append(time.time() - start_time)
        model.reset()

    if len(times) > 1:
        times = np.array(times[1:])
        print("Sampling time: {:.3f}±{:.3f}".format(np.mean(times), np.std(times)))

    
    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    # vae.enable_vae_slicing() #
    samples = vae.decode(samples / 0.18215).sample
    # save_image(samples, f"Sample_NFE{args.num_sampling_steps}_Method_{args.accelerate_method}.png", nrow=8, normalize=True, value_range=(-1, 1))
    # 遍历samples张量中的每个图像，并分别保存
    for i, sample in enumerate(samples):
        # 构建每个图像的文件名
        filename = f"Sample_NFE{args.num_sampling_steps}_Method_{args.accelerate_method}/{i}.png"
        
        # 保存每个图像
        save_image(sample.unsqueeze(0), filename, normalize=True, value_range=(-1, 1))

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
    parser.add_argument("--accelerate-method", type=str, default=None,
                        help="Use the accelerated version of the model.")
    
    parser.add_argument("--ddim-sample", action="store_true", default=False,)
    parser.add_argument("--p-sample", action="store_true", default=False,)
    
    parser.add_argument("--path", type=str, default=None,
                        help="Optional path to a router checkpoint")
    parser.add_argument("--thres", type=float, default=0.5)

    args = parser.parse_args()
    main(args)
