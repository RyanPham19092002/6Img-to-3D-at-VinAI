# Copyright (C) 2024 co-pace GmbH (subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Théo Gieruc and Marius Kästingschäfer
# ==============================================================================

import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from tqdm import tqdm
from triplane_decoder.ray_utils import *
from triplane_decoder.intrinsics import Intrinsics

class RaysDataset(Dataset):
    def __init__(self, config_path, config, dataset_config, mode="val", factor=1):
        super().__init__()
        if mode != "full":
            self.config_path = os.path.join(config_path, f"transforms/transforms_ego_{mode}.json")
            print("In raydataset class")
            print(self.config_path)
            #if os.path.exists(self.config_path):
            #    print("Đường dẫn self.config_path tồn tại.")
            #else:
            #    print("Đường dẫn self.config_path không tồn tại.")
        else:
            self.config_path = os.path.join(config_path, f"transforms/transforms_ego.json")
        self.config_dir = os.path.dirname(self.config_path)
        self.config = config
        self.dataset_config = dataset_config
        self.mode = mode
        self.factor = factor

        if config.decoder.whiteout:
            # print("Có config.decoder.whiteout")
            self.N_z, self.N_h, self.N_w = config.N_z_, config.N_h_, config.N_w_
            self.scale_z, self.scale_h, self.scale_w = config.scale_z, config.scale_h, config.scale_w
            self.offset_z, self.offset_h, self.offset_w = config.offset_z, config.offset_h, config.offset_w
            self.compute_bounds()


        self.intrinsics = None
        self.define_transforms()
        self.read_meta()

    def compute_bounds(self):
        self.x_bounds = [-self.N_z * self.scale_z  - self.offset_z, self.N_z * self.scale_z  - self.offset_z]
        self.y_bounds = [-self.N_h * self.scale_h  - self.offset_h, self.N_h * self.scale_h  - self.offset_h]
        self.z_bounds = [-self.N_w * self.scale_w  - self.offset_w, self.N_w * self.scale_w  - self.offset_w]
 

    def read_meta(self):
        # print("In read meta")
        # append path of this file to config path 
        root_path = os.path.dirname(os.path.dirname(__file__))    
        self.config_path = os.path.join(root_path, self.config_path)
        try:
            with open(self.config_path, 'r') as f:
                print("self.config_path------", self.config_path)
                self.meta = json.load(f)
        except FileNotFoundError:
            print("File not found")
        except Exception as e:
            print("An error occurred: ", str(e))

        fl_x = self.meta['img_size'][0] / (2*np.tan(self.meta['fov'] * np.pi / 360))
        fl_y = self.meta['img_size'][0] / (2*np.tan(self.meta['fov'] * np.pi / 360))
        self.intrinsics = Intrinsics(self.meta['img_size'][0], self.meta['img_size'][1], fl_x, fl_y, self.meta['img_size'][0]/2, self.meta['img_size'][1]/2)
        if self.factor != 1.0:
            self.intrinsics.scale(self.factor)
     
        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = \
            get_ray_directions(self.intrinsics) # (h, w, 3)
        #print("self.directions[[:,:,2]]------------------", self.directions[:,:,:2].shape)
        self.poses = []
        self.dataset = []
        self.depth_maps = []
        pbar = tqdm(self.meta['transform'], desc='Loading Dataset', leave=False, disable=True)
        print("pbar: ", pbar)
        for i, frame in enumerate(pbar):
            pose = np.array(self.meta['transform'][frame])[:3, :4]
            self.poses += [pose]
            c2w = torch.Tensor(pose)
            rays_o, rays_d = get_rays(self.directions.clone(), c2w) # both (h*w, 3)
            image_path = os.path.join(root_path, self.config_dir, "sphere_dataset", frame+".png")
            img = Image.open(image_path)
            img = img.resize((self.intrinsics.width, self.intrinsics.height), Image.LANCZOS)
            img = self.transform(img) # (4, h, w)
            img = img.view(img.size(0), -1).permute(1, 0) # (h*w, 4) RGBA
            #print("img---------------------", img.shape)
            if img.size(0) == 4:
                img = img[:, :3]*img[:, -1:] + (1-img[:, -1:]) # blend A to RGB
                #print("img blend A---------------------", img.shape)

            #print("img shape--------------------------------------------------------------", img.shape)
            if (self.config.decoder.whiteout or self.dataset_config.depth):
                
                #print("Having depth map")
                depth_path = os.path.join(root_path, self.config_dir, "sphere_dataset_log_depth", frame+".png")
                #print("depth_path-----------", depth_path)
                #depth_path = os.path.join(root_path, self.config_dir, f"{frame['depth_file_path']}")
                depth_map = Image.open(depth_path)
                #convert to grayscale
                depth_map = depth_map.convert("L")
                depth_map = depth_map.resize((self.intrinsics.width, self.intrinsics.height), Image.LANCZOS)
                depth_map = self.transform(depth_map).float() * 25.0
                #print("depth map shape before-------", depth_map.shape)
                depth_map = depth_map.view(-1,1)
                #print("depth map shape after-------", depth_map.shape)
                depth_offset = torch.linalg.norm( self.directions[:,:,:2],dim=2) 
                #print("depth_offset shape after-------", depth_offset.shape)
                
                depth_map = depth_map / torch.cos(torch.arctan(depth_offset).view(-1,1))
                #print("depth map shape after 2-------", depth_map.shape)
                #exit(0)
                if self.config.decoder.whiteout:
                # compute depth map in world coordinate
                    points = rays_o + rays_d * depth_map
                    

                    # filter out points outside the bounding box
                    mask = (points[:, 0] > self.x_bounds[0]) & (points[:, 0] < self.x_bounds[1]) & \
                            (points[:, 1] > self.y_bounds[0]) & (points[:, 1] < self.y_bounds[1]) & \
                            (points[:, 2] > self.z_bounds[0]) & (points[:, 2] < self.z_bounds[1])
                else:
                    mask = torch.ones(rays_o.size(0)).bool()
                    
            else: 
                mask = torch.ones(rays_o.size(0)).bool()
                depth_map = None

          

            if self.dataset_config.depth and depth_map is not None:
                self.dataset += [torch.cat([rays_o, rays_d, img, mask.unsqueeze(1), depth_map],-1)] # (h*w, 11)
                
            else:
                self.dataset += [torch.cat([rays_o, rays_d, img, mask.unsqueeze(1)],-1)] # (h*w, 10)
                

            
        self.dataset = torch.cat(self.dataset) # (len(self.meta['frames])*h*w, 10)
        #print("self.dataset shape------------------------------", self.dataset[:, 6:9].shape)

    def get_rays_for_visualization(self):
        # Extract ray origins and directions
        rays_o = self.dataset[:, :3].numpy()
        rays_d = self.dataset[:, 3:6].numpy()
        img = self.dataset[:, 6:9].numpy()
        return rays_o, rays_d, img
                
    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, idx):
        return self.dataset[idx]

    
