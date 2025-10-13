from TripoSG.triposg.utils.typing_utils import *

import json
import os
import random

import accelerate
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from tqdm import tqdm

from TripoSG.triposg.utils.data_utils import load_surface

class ObjaverseDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        configs: DictConfig, 
        training: bool = True, 
    ):
        super().__init__()
        self.configs = configs
        self.training = training

        self.training_ratio = configs['dataset']['training_ratio']

        self.rotating_ratio = configs['dataset'].get('rotating_ratio', 0.0)
        self.rotating_degree = configs['dataset'].get('rotating_degree', 10.0)
        self.transform = transforms.Compose([
            transforms.RandomRotation(degrees=(-self.rotating_degree, self.rotating_degree), fill=(255, 255, 255)),
        ])

        if isinstance(configs['dataset']['config'], ListConfig):
            data_configs = []
            for config in configs['dataset']['config']:
                local_data_configs = json.load(open(config))
                data_configs += local_data_configs
        else:
            data_configs = json.load(open(configs['dataset']['config']))
            
        data_configs = [config for config in data_configs]
        
        if self.training:
            data_configs = data_configs[:int(len(data_configs) * self.training_ratio)]
        else:
            data_configs = data_configs[int(len(data_configs) * self.training_ratio):]
        
        self.data_configs = data_configs
        self.image_size = (512, 512)

    def __len__(self) -> int:
        return len(self.data_configs)
    
    def _get_data_by_config(self, data_config):
        surface_path = data_config['surface_path']
        surface_data = np.load(surface_path, allow_pickle=True).item()
        surface = load_surface(surface_data['object']) # [P, 6]
            
        image_path = data_config['image_path']
        image = Image.open(image_path).resize(self.image_size)
        if random.random() < self.rotating_ratio:
            image = self.transform(image)
        image = np.array(image)
        image = torch.from_numpy(image).to(torch.uint8) # [H, W, 3]
        
        return {
            "images": image.unsqueeze(0), # [1, H, W, 3]
            "surfaces": surface.unsqueeze(0), # [1, P, 6]
        }
    
    def __getitem__(self, idx: int):
        data_config = self.data_configs[idx]
        data = self._get_data_by_config(data_config)
        return data
        
class BatchedObjaverseDataset(ObjaverseDataset):
    def __init__(
        self,
        configs: DictConfig,
        batch_size: int,
        is_main_process: bool = False,
        shuffle: bool = True,
        training: bool = True,
    ):
        assert training
        super().__init__(configs, training)
        self.batch_size = batch_size
        self.is_main_process = is_main_process
        
        if shuffle:
            random.shuffle(self.data_configs)

    def __len__(self) -> int:
        return len(self.data_configs) // self.batch_size

    def __getitem__(self, idx: int):
        # This is not used, see collate_fn
        return super().__getitem__(idx)

    def collate_fn(self, batch):
        images = torch.cat([data['images'] for data in batch], dim=0) # [B, H, W, 3]
        surfaces = torch.cat([data['surfaces'] for data in batch], dim=0) # [B, P, 6]
        assert images.shape[0] == surfaces.shape[0] == self.batch_size
        batch = {
            "images": images,
            "surfaces": surfaces,
        }
        return batch