import argparse
import os
import sys
import logging
from glob import glob
from typing import Any, Union

import numpy as np
import torch
import trimesh
from huggingface_hub import snapshot_download
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('/data1/luoyajing/3d_pruning')
sys.path.append('/data1/luoyajing/3d_pruning/TripoSG')


from TripoSG.triposg.pipelines.pipeline_triposg_dynamic import TripoSGPipeline, retrieve_timesteps
from TripoSG.triposg.models.autoencoders import TripoSGVAEModel
from TripoSG.triposg.models.transformers.triposg_transformer_dynamic import TripoSGDiTModelDyn
from TripoSG.triposg.schedulers import RectifiedFlowScheduler
from TripoSG.triposg.datasets import ObjaverseMaskDataset, BatchedObjaverseMaskDataset, MultiEpochsDataLoader
from TripoSG.triposg.utils.metric_utils import compute_cd_and_f_score_in_training
from TripoSG.triposg.utils.train_utils import get_configs
from TripoSG.triposg.utils.render_utils import (
    render_views_around_mesh, 
    render_normal_views_around_mesh, 
    make_grid_for_images_or_videos,
    export_renderings
)
from transformers import (
    BitImageProcessor,
    Dinov2Model,
)
from image_process import prepare_image
from briarmbg import BriaRMBG

import pymeshlab


def mesh_to_pymesh(vertices, faces):
    mesh = pymeshlab.Mesh(vertex_matrix=vertices, face_matrix=faces)
    ms = pymeshlab.MeshSet()
    ms.add_mesh(mesh)
    return ms

def pymesh_to_trimesh(mesh):
    verts = mesh.vertex_matrix()#.tolist()
    faces = mesh.face_matrix()#.tolist()
    return trimesh.Trimesh(vertices=verts, faces=faces)  #, vID, fID

def simplify_mesh(mesh: trimesh.Trimesh, n_faces):
    if mesh.faces.shape[0] > n_faces:
        ms = mesh_to_pymesh(mesh.vertices, mesh.faces)
        ms.meshing_merge_close_vertices()
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum = n_faces)
        return pymesh_to_trimesh(ms.current_mesh())
    else:
        return mesh

@torch.no_grad()
def run_evaluation(
    pipe: Any,
    dataloader: Any,
    rmbg_net: Any,
    args: argparse.Namespace,
    configs: dict,
    logger: logging.Logger,
    output_dir: str,
    ori: bool = False,
) -> None:
    if args.seed >= 0:
        generator = torch.Generator(device=pipe.device).manual_seed(args.seed)
    else:
        generator = None

    metrics_dictlist = defaultdict(list)
    
    for i, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
        images = batch["images"]
        surfaces = batch["surfaces"]
        
        # In ObjaverseMaskDataset, the batch size is always 1, so we unpack it.
        # But after collation, it might be a list of lists.
        if len(images.shape) == 5:
            images = images[0] # (1, N, H, W, 3) -> (N, H, W, 3)
        img_pils = [Image.fromarray(image) for image in images.cpu().numpy()]

        surfaces = batch["surfaces"].cpu().numpy()
        if len(surfaces.shape) == 4:
            surfaces = surfaces[0] # (1, N, P, 6) -> (N, P, 6)
            
        outputs = pipe(
            image=img_pils,
            generator=generator,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            ori=ori,
        ).samples
        
        pred_meshes = [trimesh.Trimesh(o[0].astype(np.float32), np.ascontiguousarray(o[1])) for o in outputs]

        if args.faces > 0:
            pred_meshes = [simplify_mesh(mesh, args.faces) for mesh in pred_meshes]

        local_eval_dir = os.path.join(output_dir, f"{i:04d}")
        os.makedirs(local_eval_dir, exist_ok=True)
        
        chamfer_distances, f_scores = [], []
        
        for n in range(len(pred_meshes)):
            gt_surface = surfaces[n]
            pred_mesh = pred_meshes[n]

            if pred_mesh is None or len(pred_mesh.vertices) == 0 or len(pred_mesh.faces) == 0:
                pred_mesh = trimesh.Trimesh(vertices=[[0, 0, 0]], faces=[[0, 0, 0]])

            cd, f_score = compute_cd_and_f_score_in_training(
                gt_surface, pred_mesh,
                num_samples=configs["val"]["metric"]["cd_num_samples"],
                threshold=configs["val"]["metric"]["f1_score_threshold"],
                metric=configs["val"]["metric"]["cd_metric"],
            )
            cd = configs["val"]["metric"]["default_cd"] if np.isnan(cd) else cd
            f_score = configs["val"]["metric"]["default_f1"] if np.isnan(f_score) else f_score
            chamfer_distances.append(cd)
            f_scores.append(f_score)

            # Save results
            img_pils[n].save(os.path.join(local_eval_dir, f"{n:02d}_gt.png"))
            pred_mesh.export(os.path.join(local_eval_dir, f"{n:02d}_pred.glb"))
            
            rendered_images = render_views_around_mesh(
                pred_mesh, 
                num_views=configs["val"]["rendering"]["num_views"],
                radius=configs["val"]["rendering"]["radius"],
            )
            export_renderings(
                rendered_images,
                os.path.join(local_eval_dir, f"{n:02d}_rendered.gif"),
                fps=configs["val"]["rendering"]["fps"]
            )
            
            rendered_normals = render_normal_views_around_mesh(
                pred_mesh,
                num_views=configs["val"]["rendering"]["num_views"],
                radius=configs["val"]["rendering"]["radius"],
            )
            export_renderings(
                rendered_normals,
                os.path.join(local_eval_dir, f"{n:02d}_normals.gif"),
                fps=configs["val"]["rendering"]["fps"]
            )

        logger.info(f"[{i:04d}/{len(dataloader):04d}] Chamfer Distance: {[f'{x:.4f}' for x in chamfer_distances]}")
        logger.info(f"[{i:04d}/{len(dataloader):04d}] F-score: {[f'{x:.4f}' for x in f_scores]}")
        metrics_dictlist[f"chamfer_distance"].extend(chamfer_distances)
        metrics_dictlist[f"f_score"].extend(f_scores)

    # Calculate and print average metrics
    avg_cd = np.mean(metrics_dictlist["chamfer_distance"])
    avg_f1 = np.mean(metrics_dictlist["f_score"])
    mode_str = "original" if ori else "pruned"
    logger.info(f"[{mode_str.upper()}] Average Chamfer Distance: {avg_cd:.4f}")
    logger.info(f"[{mode_str.upper()}] Average F-score: {avg_f1:.4f}")
    
    with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
        f.write(f"Mode: {mode_str}\n")
        f.write(f"Average Chamfer Distance: {avg_cd:.4f}\n")
        f.write(f"Average F-score: {avg_f1:.4f}\n")


if __name__ == "__main__":
    device = "cuda"
    dtype = torch.float16

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/mp8_nt2048.yaml", help="Path to the config file")
    parser.add_argument("--output-path", type=str, default="./evaluation_results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument("--faces", type=int, default=-1)
    
    parser.add_argument("--router-ckpt", type=str, default=None)
    parser.add_argument("--thres", type=float, default=0.5)
    parser.add_argument("--mlp-mode", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--self-attn-mode", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--cross-attn-mode", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", action="store_true")
    
    args, extras = parser.parse_known_args()
    configs = get_configs(args.config, extras)
    
    os.makedirs(args.output_path, exist_ok=True)

    # Initialize the logger
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        level=logging.INFO
    )
    logger = logging.getLogger(__name__)
    log_file = os.path.join(args.output_path, "eval.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s - %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S"
    ))
    logger.addHandler(file_handler)

    logger.info(f"Start evaluation...")
    logger.info(f"Arguments: {args}")

    # download pretrained weights
    triposg_weights_dir = "/data1/luoyajing/3d_pruning/TripoSG/pretrained_weights/TripoSG"
    rmbg_weights_dir = "/data1/luoyajing/3d_pruning/TripoSG/pretrained_weights/RMBG-1.4"

    # init rmbg model for background removal
    rmbg_net = BriaRMBG.from_pretrained(rmbg_weights_dir).to(device)
    rmbg_net.eval() 

    # init tripoSG pipeline
    vae = TripoSGVAEModel.from_pretrained(triposg_weights_dir, subfolder="vae").to(device, dtype)
    transformer = TripoSGDiTModelDyn.from_pretrained(triposg_weights_dir, subfolder="transformer").to(device, dtype)
    scheduler = RectifiedFlowScheduler.from_pretrained(triposg_weights_dir, subfolder="scheduler")
    feature_extractor_dinov2 = BitImageProcessor.from_pretrained(triposg_weights_dir, subfolder="feature_extractor_dinov2")
    image_encoder_dinov2 = Dinov2Model.from_pretrained(triposg_weights_dir, subfolder="image_encoder_dinov2").to(device, dtype)

    pipe = TripoSGPipeline(vae=vae, 
                           transformer=transformer, 
                           scheduler=scheduler, 
                           feature_extractor_dinov2=feature_extractor_dinov2, 
                           image_encoder_dinov2=image_encoder_dinov2).to(device, dtype)

    timesteps, num_inference_steps = retrieve_timesteps(
        pipe.scheduler, num_inference_steps=args.num_inference_steps, device=device, timesteps=None
    )
    timestep_map = pipe.scheduler.timesteps.tolist()

    pipe.transformer.load_ranking(
            args.router_ckpt, 
            args.num_inference_steps, 
            timestep_map, 
            args.thres,
            mlp_mode=args.mlp_mode,
            self_attn_mode=args.self_attn_mode,
            cross_attn_mode=args.cross_attn_mode,
            log_file= args.output_path + "/eval.log",
        )
    
    # Dataset and Dataloader
    val_dataset = ObjaverseMaskDataset(
        configs=configs,
        training=False,
    )
    
    val_loader = MultiEpochsDataLoader(
        val_dataset,
        batch_size=configs["val"]["batch_size_per_gpu"],
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=args.pin_memory,
    )

    # Run evaluation for ori=False (pruned)
    output_path_pruned = os.path.join(args.output_path, "pruned")
    os.makedirs(output_path_pruned, exist_ok=True)
    logger.info("="*20 + " Running evaluation for pruned model (ori=False) " + "="*20)
    run_evaluation(
        pipe=pipe,
        dataloader=val_loader,
        rmbg_net=rmbg_net,
        args=args,
        configs=configs,
        logger=logger,
        output_dir=output_path_pruned,
        ori=False,
    )

    # Run evaluation for ori=True (original)
    output_path_ori = os.path.join(args.output_path, "ori")
    os.makedirs(output_path_ori, exist_ok=True)
    logger.info("="*20 + " Running evaluation for original model (ori=True) " + "="*20)
    run_evaluation(
        pipe=pipe,
        dataloader=val_loader,
        rmbg_net=rmbg_net,
        args=args,
        configs=configs,
        logger=logger,
        output_dir=output_path_ori,
        ori=True,
    )

    logger.info(f"Evaluation finished. Results saved to {args.output_path}")

'''
CUDA_VISIBLE_DEVICES=2 python TripoSG/scripts/eval/eval_triposg_mask.py \
    --output-path out_eval
    --router-ckpt  /data1/luoyajing/3d_pruning/output_mask/scaleup_mp8_nt512_mask_data_1000_lr_1e-3__mlp_sa_ca/checkpoints/000063.pt
    --thres "0.5"
   
'''