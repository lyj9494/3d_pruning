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

class ObjaverseMaskDataset(torch.utils.data.Dataset):
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

        self.train_data_dir = configs['dataset']['train_data_dir']
        self.eval_data_dir = configs['dataset']['eval_data_dir']
        
        train_uids = sorted([d for d in os.listdir(self.train_data_dir) if os.path.isdir(os.path.join(self.train_data_dir, d))])
        
        eval_uids = []
        if os.path.exists(self.eval_data_dir):
            eval_uids = sorted([d for d in os.listdir(self.eval_data_dir) if os.path.isdir(os.path.join(self.eval_data_dir, d))])
        
        if self.training:
            eval_uids_set = set(eval_uids)
            uids = [uid for uid in train_uids if uid not in eval_uids_set]
        else:
            uids = eval_uids
            
        self.image_paths = []
        if self.training:
            for uid in uids:
                image_path = os.path.join(self.train_data_dir, uid, 'rgb_000_front.png')
                if os.path.exists(image_path):
                    self.image_paths.append(image_path)
        else:
            for uid in uids:
                image_path = os.path.join(self.eval_data_dir, uid, 'rendering_rmbg.png')
                if os.path.exists(image_path):
                    self.image_paths.append(image_path)
                
        
        # TODO: a debug here
        if self.training:
            self.image_paths = self.image_paths[:100]
        else:
            self.image_paths = self.image_paths[:1]
        '''
        '''
        
        self.image_size = (512, 512)

    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB").resize(self.image_size)
        if random.random() < self.rotating_ratio:
            image = self.transform(image)
        image = np.array(image)
        image = torch.from_numpy(image).to(torch.uint8) # [H, W, 3]
        
        if self.training:
            return {
                "images": image.unsqueeze(0), # [1, H, W, 3]
            }
        else:
            # surface_path
            surface_path = image_path.replace('rendering_rmbg.png', 'points.npy')
            surface_data = np.load(surface_path, allow_pickle=True).item()
            surface = load_surface(surface_data['object']) # [P, 6]
        
            return {
                "images": image.unsqueeze(0), # [1, H, W, 3]
                "surfaces": surface.unsqueeze(0), # [1, P, 6]
            }
        
class BatchedObjaverseMaskDataset(ObjaverseMaskDataset):
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
            random.shuffle(self.image_paths)

    def __len__(self) -> int:
        return len(self.image_paths) // self.batch_size

    def __getitem__(self, idx: int):
        # This is not used, see collate_fn
        return super().__getitem__(idx)

    def collate_fn(self, batch):
        images = torch.cat([data['images'] for data in batch], dim=0) # [B, H, W, 3]
        assert images.shape[0] == self.batch_size
        if self.training:
            batch = {
                "images": images,
            }
        else:
            surfaces = torch.cat([data['surfaces'] for data in batch], dim=0) # [B, P, 6]
            assert images.shape[0] == surfaces.shape[0] == self.batch_size
            batch = {
                "images": images,
                "surfaces": surfaces,
            }
        return batch