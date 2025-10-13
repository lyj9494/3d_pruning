import os,sys
sys.path.append('.')
import json
import argparse
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

import trimesh
import numpy as np
import torch
from huggingface_hub import snapshot_download
from TripoSG.triposg.utils.data_utils import scene_to_parts, mesh_to_surface, normalize_mesh
from TripoSG.triposg.utils.render_utils import render_single_view
from TripoSG.triposg.utils.metric_utils import compute_IoU_for_scene
from TripoSG.triposg.utils.image_utils import prepare_image
from TripoSG.scripts.briarmbg import BriaRMBG
import pyrender

RADIUS = 4
IMAGE_SIZE = (2048, 2048)
LIGHT_INTENSITY = 2.5
NUM_ENV_LIGHTS = 36

rmbg_weights_dir = "TripoSG/pretrained_weights/RMBG-1.4"
device = "cuda" if torch.cuda.is_available() else "cpu"
rmbg_net = BriaRMBG.from_pretrained(rmbg_weights_dir).to(device)
    
def mesh_to_point(input_path, output_path):
    assert os.path.exists(input_path), f'{input_path} does not exist'

    mesh_name = os.path.basename(input_path).split('.')[0]
    output_path = os.path.join(output_path, mesh_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    points_path = os.path.join(output_path, 'points.npy')
    if os.path.exists(points_path):
        return

    mesh = trimesh.load(input_path, process=False)
    mesh = normalize_mesh(mesh)
    if isinstance(mesh, trimesh.Scene):
        mesh = sum(mesh.dump())
    
    object = mesh_to_surface(mesh, return_dict=True)
    datas = {
        "object": object,
    }
    np.save(points_path, datas)


def render_mesh(input_path, output_path, renderer=None):
    assert os.path.exists(input_path), f'{input_path} does not exist'

    mesh_name = os.path.basename(input_path).split('.')[0]
    output_path = os.path.join(output_path, mesh_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    rendering_path = os.path.join(output_path, 'rendering.png')
    if os.path.exists(rendering_path):
        return

    mesh = normalize_mesh(trimesh.load(input_path, process=False))
    mesh = mesh.to_geometry()
    image = render_single_view(
        mesh,
        radius=RADIUS,
        image_size=IMAGE_SIZE,
        light_intensity=LIGHT_INTENSITY,
        num_env_lights=NUM_ENV_LIGHTS,
        return_type='pil',
        renderer=renderer
    )
    image.save(rendering_path)
    

def remove_background(input_path, output_path):
    assert os.path.exists(input_path), f'{input_path} does not exist'

    mesh_name = os.path.basename(os.path.dirname(input_path))
    output_path = os.path.join(output_path, mesh_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    rendering_rmbg_path = os.path.join(output_path, 'rendering_rmbg.png')
    if os.path.exists(rendering_rmbg_path):
        return

    rendering_rmbg = prepare_image(input_path, bg_color=np.array([1.0, 1.0, 1.0]), rmbg_net=rmbg_net, device=device)
    rendering_rmbg.save(rendering_rmbg_path)

def calculate_iou(input_path, output_path):
    assert os.path.exists(input_path), f'{input_path} does not exist'

    mesh_name = os.path.basename(input_path).split('.')[0]
    output_path = os.path.join(output_path, mesh_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    iou_path = os.path.join(output_path, 'iou.json')
    if os.path.exists(iou_path):
        return

    config = {
        'iou_mean': 0.0,
        'iou_max': 0.0,
        'iou_list': [],
    }
    mesh = normalize_mesh(trimesh.load(input_path, process=False))
    try:
        iou_list = compute_IoU_for_scene(mesh, return_type='iou_list')
        config['iou_list'] = iou_list
        config['iou_mean'] = np.mean(iou_list)
        config['iou_max'] = np.max(iou_list)
    except:
        config['iou_list'] = []
        config['iou_mean'] = 0.0
        config['iou_max'] = 0.0

    json.dump(config, open(iou_path, 'w'), indent=4)
    
def process_mesh(mesh_name, input_path, output_path, renderer=None):
    mesh_path = os.path.join(input_path, mesh_name)
    # 1. Sample points from mesh surface
    
    mesh_to_point(mesh_path, output_path)
    # 2. Render images
    render_mesh(mesh_path, output_path, renderer=renderer)
    
    # 3. Remove background for rendered images and resize to 90%
    export_mesh_folder = os.path.join(output_path, mesh_name.replace('.glb', ''))
    export_rendering_path = os.path.join(export_mesh_folder, 'rendering.png')
    remove_background(export_rendering_path, output_path)

    time.sleep(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='/workspace/luoyajing/objaverse/hf-objaverse-v1/glbs')
    parser.add_argument('--output', type=str, default='preprocessed_data_objaverse')
    parser.add_argument('--num_workers', type=int, default=0, help='并行处理的线程数, 0表示在主线程中顺序执行')
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output
    num_workers = args.num_workers

    assert os.path.exists(input_path), f'{input_path} does not exist'

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    mesh_names = os.listdir(input_path)

    renderer = pyrender.OffscreenRenderer(*IMAGE_SIZE)
    for mesh_name in tqdm(mesh_names, desc='Processing meshes'):
        process_mesh(mesh_name, input_path, output_path, renderer=renderer)
    renderer.delete()

    # generate configs
    configs = []
    for mesh_name in tqdm(mesh_names, desc='Generating configs'):
        mesh_path = os.path.join(output_path, mesh_name.replace('.glb', ''))
        surface_path = os.path.join(mesh_path, 'points.npy')
        image_path = os.path.join(mesh_path, 'rendering_rmbg.png')
        iou_path = os.path.join(mesh_path, 'iou.json')
        config = {
            "file": mesh_name,
            "valid": False,
            "mesh_path": os.path.join(input_path, mesh_name),
            "surface_path": None,
            "image_path": None,
            "iou_mean": 0.0,
            "iou_max": 0.0
        }
        try:
            iou_config = json.load(open(iou_path))
            config['iou_mean'] = iou_config['iou_mean']
            config['iou_max'] = iou_config['iou_max']
            assert os.path.exists(surface_path)
            config['surface_path'] = surface_path
            assert os.path.exists(image_path)
            config['image_path'] = image_path
            config['valid'] = True
            configs.append(config)
        except:
            continue

    configs_path = os.path.join(output_path, 'object_part_configs.json')
    json.dump(configs, open(configs_path, 'w'), indent=4)
    
    # export MESA_DRIVERS_PATH=/usr/lib/x86_64-linux-gnu/dri/
    # export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

    
    # CUDA_VISIBLE_DEVICES=4,5 python TripoSG/triposg/datasets/preprocess/preprocess.py --num_workers 4 --multi_gpu
    # CUDA_VISIBLE_DEVICES=4 python TripoSG/triposg/datasets/preprocess/preprocess.py --num_workers 0