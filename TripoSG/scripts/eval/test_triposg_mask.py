import argparse
import os
import sys
from glob import glob
from typing import Any, Union

import numpy as np
import torch
import trimesh
from huggingface_hub import snapshot_download
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('/data1/luoyajing/3d_pruning')
sys.path.append('/data1/luoyajing/3d_pruning/TripoSG')


from TripoSG.triposg.pipelines.pipeline_triposg_dynamic import TripoSGPipeline, retrieve_timesteps
from TripoSG.triposg.models.autoencoders import TripoSGVAEModel
from TripoSG.triposg.models.transformers.triposg_transformer_dynamic import TripoSGDiTModelDyn
from TripoSG.triposg.schedulers import RectifiedFlowScheduler
from transformers import (
    BitImageProcessor,
    Dinov2Model,
)
from image_process import prepare_image
from briarmbg import BriaRMBG

import pymeshlab


@torch.no_grad()
def run_triposg(
    pipe: Any,
    image_input: Union[str, Image.Image],
    rmbg_net: Any,
    seed: int,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.0,
    faces: int = -1,
    ori: bool = False,
) -> trimesh.Scene:

    img_pil = prepare_image(image_input, bg_color=np.array([1.0, 1.0, 1.0]), rmbg_net=rmbg_net)

    outputs = pipe(
        image=img_pil,
        generator=torch.Generator(device=pipe.device).manual_seed(seed),
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,  
        ori=ori,
    ).samples[0]
    mesh = trimesh.Trimesh(outputs[0].astype(np.float32), np.ascontiguousarray(outputs[1]))

    if faces > 0:
        mesh = simplify_mesh(mesh, faces)

    return mesh

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

if __name__ == "__main__":
    device = "cuda"
    dtype = torch.float16

    parser = argparse.ArgumentParser()
    parser.add_argument("--image-input", type=str, required=True)
    parser.add_argument("--output-path", type=str, default="./output.glb")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument("--faces", type=int, default=-1)
    
    parser.add_argument("--router-ckpt", type=str, default=None)
    parser.add_argument("--thres", type=float, default=0.5)
    parser.add_argument("--mlp-mode", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--self-attn-mode", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--cross-attn-mode", default=True, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    
    os.makedirs(args.output_path, exist_ok=True)

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
    
    # run inference
    run_triposg(
        pipe,
        image_input=args.image_input,
        rmbg_net=rmbg_net,
        seed=args.seed,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        faces=args.faces,
        ori = False,
    ).export(args.output_path + "/output.glb")
    print(f"Mesh saved to {args.output_path + '/output.glb'}")

'''
CUDA_VISIBLE_DEVICES=7 python TripoSG/scripts/eval/test_triposg_mask.py \
    --image-input /data1/luoyajing/PartCrafter/preprocessed_data_objaverse_demo/0a7a71cb4c064e729c3b7c2742ff9f72/rendering_rmbg.png \
    --output-path out_test\
    --router-ckpt /data1/luoyajing/3d_pruning/output_mask/scaleup_mp8_nt512_mask_data_1000_lr_1e-3__mlp_sa_ca/checkpoints/000063.pt

'''