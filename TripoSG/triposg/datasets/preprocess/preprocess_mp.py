import os, sys
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

# --- 全局常量定义 ---
RADIUS = 4
IMAGE_SIZE = (2048, 2048)
LIGHT_INTENSITY = 2.5
NUM_ENV_LIGHTS = 36
RMBG_WEIGHTS_DIR = "TripoSG/pretrained_weights/RMBG-1.4"
    
def mesh_to_point(input_path, output_path):
    """从网格表面采样点并保存"""
    assert os.path.exists(input_path), f'{input_path} does not exist'

    mesh_name = os.path.basename(input_path).split('.')[0]
    output_path = os.path.join(output_path, mesh_name)
    os.makedirs(output_path, exist_ok=True)

    points_path = os.path.join(output_path, 'points.npy')
    if os.path.exists(points_path):
        return

    mesh = trimesh.load(input_path, process=False)
    mesh = normalize_mesh(mesh)
    if isinstance(mesh, trimesh.Scene):
        mesh = sum(mesh.dump())
    
    object_data = mesh_to_surface(mesh, return_dict=True)
    datas = {"object": object_data}
    np.save(points_path, datas)


def render_mesh(input_path, output_path, renderer):
    """使用指定的渲染器渲染网格"""
    assert os.path.exists(input_path), f'{input_path} does not exist'

    mesh_name = os.path.basename(input_path).split('.')[0]
    output_path = os.path.join(output_path, mesh_name)
    os.makedirs(output_path, exist_ok=True)

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
    

def remove_background(input_path, output_path, rmbg_net, device):
    """使用指定的模型和设备为图像移除背景"""
    assert os.path.exists(input_path), f'{input_path} does not exist'

    mesh_name = os.path.basename(os.path.dirname(input_path))
    output_path = os.path.join(output_path, mesh_name)
    os.makedirs(output_path, exist_ok=True)

    rendering_rmbg_path = os.path.join(output_path, 'rendering_rmbg.png')
    if os.path.exists(rendering_rmbg_path):
        return

    rendering_rmbg = prepare_image(input_path, bg_color=np.array([1.0, 1.0, 1.0]), rmbg_net=rmbg_net, device=device)
    rendering_rmbg.save(rendering_rmbg_path)


def calculate_iou(input_path, output_path):
    """计算网格的IoU指标"""
    assert os.path.exists(input_path), f'{input_path} does not exist'

    mesh_name = os.path.basename(input_path).split('.')[0]
    output_path = os.path.join(output_path, mesh_name)
    os.makedirs(output_path, exist_ok=True)

    iou_path = os.path.join(output_path, 'iou.json')
    if os.path.exists(iou_path):
        return

    config = {'iou_mean': 0.0, 'iou_max': 0.0, 'iou_list': []}
    mesh = normalize_mesh(trimesh.load(input_path, process=False))
    try:
        iou_list = compute_IoU_for_scene(mesh, return_type='iou_list')
        config['iou_list'] = iou_list
        config['iou_mean'] = np.mean(iou_list) if iou_list else 0.0
        config['iou_max'] = np.max(iou_list) if iou_list else 0.0
    except Exception as e:
        print(f"Could not compute IoU for {mesh_name}: {e}")
        config['iou_list'] = []
        config['iou_mean'] = 0.0
        config['iou_max'] = 0.0

    with open(iou_path, 'w') as f:
        json.dump(config, f, indent=4)

    
def process_mesh(mesh_name, input_path, output_path, renderer, rmbg_net, device):
    """
    单个网格的完整处理流程：采样、渲染、去背景。
    所有依赖项（renderer, rmbg_net, device）都作为参数传入。
    """
    mesh_path = os.path.join(input_path, mesh_name)
    mesh_basename = mesh_name.split('.')[0]
    
    # 1. 从网格表面采样点
    mesh_to_point(mesh_path, output_path)
    
    # 2. 渲染图像
    render_mesh(mesh_path, output_path, renderer=renderer)
    
    # 3. 为渲染图像移除背景
    export_mesh_folder = os.path.join(output_path, mesh_basename)
    export_rendering_path = os.path.join(export_mesh_folder, 'rendering.png')
    remove_background(export_rendering_path, output_path, rmbg_net, device)


def process_mesh_worker(mesh_names, worker_id, num_gpus, input_path, output_path):
    """
    为单个线程设计的工作函数。
    它会独立初始化模型和渲染器，确保线程安全。
    """
    # 1. 根据 worker_id 分配 GPU
    gpu_id = worker_id % num_gpus
    device = f'cuda:{gpu_id}'
    torch.cuda.set_device(device)
    
    # 2. 在当前线程和指定设备上初始化资源
    print(f"Worker {worker_id}: Initializing on device {device}...")
    renderer = pyrender.OffscreenRenderer(*IMAGE_SIZE)
    rmbg_net = BriaRMBG.from_pretrained(RMBG_WEIGHTS_DIR).to(device)
    
    # 3. 循环处理分配到的所有网格
    for mesh_name in tqdm(mesh_names, desc=f'Worker {worker_id} on GPU {gpu_id}', position=worker_id):
        try:
            process_mesh(mesh_name, input_path, output_path, renderer, rmbg_net, device)
        except Exception as e:
            print(f"Error processing {mesh_name} on worker {worker_id}: {e}")

    # 4. 清理资源
    renderer.delete()
    print(f"Worker {worker_id}: Processing finished and resources cleaned up.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess 3D mesh data for TripoSG.")
    parser.add_argument('--input', type=str, default='/workspace/luoyajing/objaverse/hf-objaverse-v1/glbs', help='Input directory containing .glb files.')
    parser.add_argument('--output', type=str, default='preprocessed_data_objaverse', help='Output directory to save processed data.')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of parallel threads. Set to 0 for single-threaded execution.')
    parser.add_argument('--multi_gpu', action='store_true', help='Enable multi-GPU processing. Requires num_workers > 0.')
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output
    num_workers = args.num_workers

    assert os.path.exists(input_path), f'{input_path} does not exist'
    os.makedirs(output_path, exist_ok=True)

    mesh_names = [f for f in os.listdir(input_path) if f.endswith('.glb')]

    # --- 并行或串行处理 ---
    if num_workers > 0:
        # 多线程/多GPU模式
        num_gpus = torch.cuda.device_count()
        if not args.multi_gpu:
            num_gpus = 1  # 如果未指定 multi_gpu，则所有 worker 只使用第一个 GPU
        
        assert num_gpus > 0, "No CUDA devices found for parallel processing."
        print(f"Starting parallel processing with {num_workers} workers on {num_gpus} GPU(s).")
        
        # 将任务列表分割成 num_workers 份
        mesh_chunks = np.array_split(mesh_names, num_workers)
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(process_mesh_worker, chunk, worker_id, num_gpus, input_path, output_path)
                for worker_id, chunk in enumerate(mesh_chunks) if chunk.size > 0
            ]
            
            # 等待所有任务完成
            for future in as_completed(futures):
                try:
                    future.result()  # 获取结果或捕获异常
                except Exception as e:
                    print(f"A worker process failed: {e}")
    else:
        # 单线程模式
        print("Starting single-threaded processing.")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        renderer = pyrender.OffscreenRenderer(*IMAGE_SIZE)
        rmbg_net = BriaRMBG.from_pretrained(RMBG_WEIGHTS_DIR).to(device)
        
        for mesh_name in tqdm(mesh_names, desc='Processing meshes'):
            process_mesh(mesh_name, input_path, output_path, renderer, rmbg_net, device)
            
        renderer.delete()

    print("\n--- All mesh processing completed. Generating final config file. ---")

    # --- 生成最终的配置文件 ---
    configs = []
    for mesh_name in tqdm(mesh_names, desc='Generating configs'):
        mesh_basename = mesh_name.split('.')[0]
        mesh_folder = os.path.join(output_path, mesh_basename)
        
        surface_path = os.path.join(mesh_folder, 'points.npy')
        image_path = os.path.join(mesh_folder, 'rendering_rmbg.png')
        iou_path = os.path.join(mesh_folder, 'iou.json')
        
        config = {
            "file": mesh_name,
            "valid": False,
            "mesh_path": os.path.join(input_path, mesh_name),
            "surface_path": None,
            "image_path": None,
            "iou_mean": 0.0,
            "iou_max": 0.0
        }
        
        if os.path.exists(surface_path) and os.path.exists(image_path):
            config['surface_path'] = surface_path
            config['image_path'] = image_path
            config['valid'] = True
            if os.path.exists(iou_path):
                try:
                    with open(iou_path, 'r') as f:
                        iou_config = json.load(f)
                    config['iou_mean'] = iou_config.get('iou_mean', 0.0)
                    config['iou_max'] = iou_config.get('iou_max', 0.0)
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode IoU JSON for {mesh_name}. Skipping.")
            
            configs.append(config)
        else:
            print(f"Warning: Missing processed files for {mesh_name}. Skipping in config.")

    configs_path = os.path.join(output_path, 'object_part_configs.json')
    with open(configs_path, 'w') as f:
        json.dump(configs, f, indent=4)
        
    print(f"\nConfiguration file saved to {configs_path}")
    print("Preprocessing finished successfully. ✨")