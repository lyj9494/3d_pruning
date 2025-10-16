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
import gc
from collections import defaultdict

import trimesh
from PIL import Image
import numpy as np
from tqdm import tqdm

import torch
import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger as get_accelerate_logger
from accelerate.data_loader import DataLoaderShard
from safetensors.torch import load_file

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
    MultiEpochsDataLoader, 
    yield_forever
)

from TripoSG.triposg.utils.train_utils import (
    get_configs,
)
from TripoSG.triposg.utils.render_utils import (
    render_views_around_mesh, 
    render_normal_views_around_mesh, 
    make_grid_for_images_or_videos,
    export_renderings
)
from TripoSG.triposg.utils.metric_utils import compute_cd_and_f_score_in_training


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained TripoSG model.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the config file used during training."
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path to the saved router checkpoint file (e.g., 'output/tag/checkpoints/010000.pt')."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="eval_output",
        help="Path to the output directory to save evaluation results."
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="eval",
        help="A tag for this evaluation run."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for the PRNG."
    )
    parser.add_argument(
        "--max_eval_steps",
        type=int,
        default=10,
        help="The max number of batches to evaluate."
    )
    parser.add_argument(
        "--guidance_scales",
        type=float,
        nargs="+",
        default=[7.0],
        help="A list of CFG scales to use for validation."
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="The number of inference steps for the pipeline."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="The number of processes for the data loader."
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help="Type of mixed precision evaluation."
    )

    parser.add_argument(
        '--mlp_mode',
        default=True,
        action=argparse.BooleanOptionalAction
    )
    parser.add_argument(
        '--self_attn_mode',
        default=True,
        action=argparse.BooleanOptionalAction
    )
    parser.add_argument(
        '--cross_attn_mode',
        default=True,
        action=argparse.BooleanOptionalAction
    )
    
    args, extras = parser.parse_known_args()
    configs = get_configs(args.config, extras)

    return args, configs


@torch.no_grad()
def run_evaluation(
    dataloader,
    feature_extractor_dinov2, image_encoder_dinov2,
    vae, transformer, 
    eval_dir,
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

    if args.seed >= 0:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
    else:
        generator = None
        
    for activate_router_mode in [False, True]:
        run_prefix = "router_on" if activate_router_mode else "router_off"
        logger.info(f"Running validation with router mode: {run_prefix}")

        val_progress_bar = tqdm(
            range(len(dataloader)) if args.max_eval_steps is None else range(args.max_eval_steps),
            desc=f"Evaluation [{run_prefix}]",
            ncols=125,
            disable=not accelerator.is_main_process
        )

        medias_dictlist, metrics_dictlist = defaultdict(list), defaultdict(list)

        val_dataloder = yield_forever(dataloader)
        val_step = 0
        while val_step < args.max_eval_steps:
            batch = next(val_dataloder)

            images = batch["images"]
            if len(images.shape) == 5:
                images = images[0]
            images = [Image.fromarray(image) for image in images.cpu().numpy()]
            surfaces = batch["surfaces"].cpu().numpy()
            if len(surfaces.shape) == 4:
                surfaces = surfaces[0]

            N = len(images)
            val_progress_bar.set_postfix({"num_objects": N})

            with torch.autocast("cuda", torch.float16):
                for guidance_scale in sorted(args.guidance_scales):
                    pred_meshes = pipeline(
                        images, 
                        num_inference_steps=args.num_inference_steps,
                        num_tokens=configs['model']['vae']['num_tokens'],
                        guidance_scale=guidance_scale, 
                        generator=generator,
                        use_flash_decoder=configs['val']['use_flash_decoder'],
                        activate_router=activate_router_mode,
                        # mlp_mode=args.mlp_mode,
                        # self_attn_mode=args.self_attn_mode,
                        # cross_attn_mode=args.cross_attn_mode,
                    ).meshes

                    if accelerator.is_main_process:
                        local_eval_dir = os.path.join(eval_dir, run_prefix, f"guidance_scale_{guidance_scale:.1f}")
                        os.makedirs(local_eval_dir, exist_ok=True)
                        rendered_images_list, rendered_normals_list = [], []
                        
                        images[0].save(os.path.join(local_eval_dir, f"{val_step:04d}.png"))
                        
                        import random
                        n = random.randint(0, N - 1)
                        
                        if pred_meshes[n] is None:
                            pred_meshes[n] = trimesh.Trimesh(vertices=[[0, 0, 0]], faces=[[0, 0, 0]])
                        pred_meshes[n].export(os.path.join(local_eval_dir, f"{val_step:04d}_{n:02d}.glb"))
                        
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

                        medias_dictlist[f"guidance_scale_{guidance_scale:.1f}/gt_image"] += [images[0]]
                        medias_dictlist[f"guidance_scale_{guidance_scale:.1f}/pred_rendered_images"] += rendered_images_list
                        medias_dictlist[f"guidance_scale_{guidance_scale:.1f}/pred_rendered_normals"] += rendered_normals_list

                    chamfer_distances, f_scores = [], []
                    for n in range(N):
                        gt_surface = surfaces[n]
                        pred_mesh = pred_meshes[n]
                        if pred_mesh is None:
                            pred_mesh = trimesh.Trimesh(vertices=[[0, 0, 0]], faces=[[0, 0, 0]])
                        cd, f_score = compute_cd_and_f_score_in_training(
                            gt_surface, pred_mesh,
                            num_samples=configs['val']['metric']['cd_num_samples'],
                            threshold=configs['val']['metric']['f1_score_threshold'],
                            metric=configs['val']['metric']['cd_metric']
                        )
                        cd = configs['val']['metric']['default_cd'] if np.isnan(cd) else cd
                        f_score = configs['val']['metric']['default_f1'] if np.isnan(f_score) else f_score
                        chamfer_distances.append(cd)
                        f_scores.append(f_score)

                    chamfer_distances = torch.tensor(chamfer_distances, device=accelerator.device)
                    f_scores = torch.tensor(f_scores, device=accelerator.device)

                    metrics_dictlist[f"chamfer_distance_cfg{guidance_scale:.1f}"].append(chamfer_distances.mean())
                    metrics_dictlist[f"f_score_cfg{guidance_scale:.1f}"].append(f_scores.mean())
                
            val_logs = {
                "chamfer_distance": chamfer_distances.mean().item(),
                "f_score": f_scores.mean().item(),
            }
            val_progress_bar.set_postfix(**val_logs)
            logger.info(
                f"Validation [{val_step:02d}/{args.max_eval_steps:02d}] " +
                f"chamfer_distance: {val_logs['chamfer_distance']:.4f}, f_score: {val_logs['f_score']:.4f}"
            )
            val_step += 1
            val_progress_bar.update(1)

        val_progress_bar.close()

        if accelerator.is_main_process:
            for key, value in medias_dictlist.items():
                if isinstance(value[0], Image.Image):
                    image_grid = make_grid_for_images_or_videos(
                        value, nrow=configs['val']['nrow'], return_type='pil'
                    )
                    image_grid.save(os.path.join(eval_dir, f"{run_prefix}_{key.replace('/', '_')}.png"))
                else:
                    image_grids = make_grid_for_images_or_videos(
                        value, nrow=configs['val']['nrow'], return_type='ndarray'
                    )
                    image_grids_pil = [Image.fromarray(grid.transpose(1, 2, 0)) for grid in image_grids]
                    export_renderings(
                        image_grids_pil, 
                        os.path.join(eval_dir, f"{run_prefix}_{key.replace('/', '_')}.gif"), 
                        fps=configs['val']['rendering']['fps']
                    )

            final_metrics = {}
            for k, v in metrics_dictlist.items():
                mean_metric = torch.tensor(v).mean().item()
                final_metrics[k] = mean_metric
                logger.info(f"Final metric for {run_prefix}/{k}: {mean_metric:.4f}")
            
            # save metrics to a file
            with open(os.path.join(eval_dir, f"metrics_{run_prefix}.txt"), "w") as f:
                for k, v in final_metrics.items():
                    f.write(f"{k}: {v:.4f}\n")


def main():
    args, configs = parse_args()

    # === Accelerator Setup ===
    accelerator = Accelerator(mixed_precision=args.mixed_precision)

    # === Logging Setup ===
    exp_dir = os.path.join(args.output_dir, args.tag)
    os.makedirs(exp_dir, exist_ok=True)
    
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        level=logging.INFO
    )
    logger = get_accelerate_logger(__name__, log_level="INFO")
    file_handler = logging.FileHandler(os.path.join(exp_dir, "log_eval.txt"))
    file_handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s - %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S"
    ))
    logger.logger.addHandler(file_handler)
    logger.logger.propagate = True

    # === Seed Setup ===
    if args.seed >= 0:
        accelerate.utils.set_seed(args.seed)
        logger.info(f"Seeding experiment with seed: {args.seed}")

    # === Model Loading ===
    logger.info("Loading models...")
    vae = TripoSGVAEModel.from_pretrained(
        configs["model"]["pretrained_model_name_or_path"], subfolder="vae"
    )
    feature_extractor_dinov2 = BitImageProcessor.from_pretrained(
        configs["model"]["pretrained_model_name_or_path"], subfolder="feature_extractor_dinov2"
    )
    image_encoder_dinov2 = Dinov2Model.from_pretrained(
        configs["model"]["pretrained_model_name_or_path"], subfolder="image_encoder_dinov2"
    )

    logger.info("Initializing transformer from base model config...")
    transformer = TripoSGDiTModel.from_pretrained(
        configs["model"]["pretrained_model_name_or_path"], subfolder="transformer"
    )

    logger.info("Adding router to the transformer...")
    noise_scheduler = RectifiedFlowScheduler.from_pretrained(
        configs["model"]["pretrained_model_name_or_path"], subfolder="scheduler"
    )
    timesteps, num_inference_steps = retrieve_timesteps(
        noise_scheduler, num_inference_steps=args.num_inference_steps, device=accelerator.device, timesteps=None
    )
    timestep_map = noise_scheduler.timesteps.tolist()
    transformer.add_router(
        num_inference_steps, 
        timestep_map, 
        mlp_mode=args.mlp_mode, 
        self_attn_mode=args.self_attn_mode, 
        cross_attn_mode=args.cross_attn_mode
    )

    logger.info(f"Loading router weights from checkpoint: {args.ckpt_path}")
    if os.path.exists(args.ckpt_path):
        checkpoint = torch.load(args.ckpt_path, map_location="cpu")
        transformer.routers.load_state_dict(checkpoint["routers"])
        logger.info("Router weights loaded successfully.")
    else:
        logger.warning(f"Checkpoint file not found at {args.ckpt_path}. Using randomly initialized router.")

    # === Freeze Models ===
    vae.requires_grad_(False)
    image_encoder_dinov2.requires_grad_(False)
    transformer.requires_grad_(False)
    vae.eval()
    image_encoder_dinov2.eval()
    transformer.eval()

    # === Dataloader Setup ===
    logger.info("Setting up dataset...")
    val_dataset = ObjaverseMaskDataset(configs=configs, training=False)
    val_loader = MultiEpochsDataLoader(
        val_dataset,
        batch_size=configs["val"]["batch_size_per_gpu"],
        num_workers=args.num_workers,
        drop_last=True,
    )
    logger.info(f"Loaded {len(val_dataset)} validation samples.")

    # === Prepare with Accelerator ===
    transformer, val_loader = accelerator.prepare(transformer, val_loader)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)
    image_encoder_dinov2.to(accelerator.device, dtype=weight_dtype)

    # === Run Evaluation ===
    logger.info("Starting evaluation...")
    run_evaluation(
        val_loader,
        feature_extractor_dinov2, image_encoder_dinov2,
        vae, transformer,
        exp_dir,
        accelerator, logger,
        args, configs
    )
    
    logger.info("Evaluation finished!")
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
    
'''
export NCCL_P2P_DISABLE=1 # for 4090
export NCCL_IB_DISABLE=1 # for 4090
export NCCL_SOCKET_NTHREADS=1 # for 4090

LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 CUDA_VISIBLE_DEVICES=6,7 \
CUDA_VISIBLE_DEVICES=6,7 python TripoSG/scripts/eval/eval_triposg.py \
    --config configs/mp8_nt2048.yaml\
    --ckpt_path output_mask/scaleup_mp8_nt512_mask/checkpoints/000060.pt \
    --output_dir evaluation_results \
    --tag "eval_run_mask_000060" \
    --guidance_scales 7.0 \
    --max_eval_steps 50 
    
accelerate launch --multi_gpu --num_processes 2 \
    TripoSG/scripts/eval/eval_triposg.py \
    --config configs/mp8_nt2048.yaml \
    --ckpt_path output_mask/scaleup_mp8_nt512_mask/checkpoints/000060.pt \
    --output_dir evaluation_results \
    --tag "eval_run_mask_000060_multi_gpu" \
    --guidance_scales 7.0 \
    --max_eval_steps 50
'''