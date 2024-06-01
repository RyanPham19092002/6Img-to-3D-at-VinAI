import numpy as np
import torch
from visualize import visualize_random_rays
import os

sphere_dataloader = []
folder = "/root/VinAI/6Img-to-3D-at-VinAI/data_VinAI/Town05/ClearNoon/vehicle.tesla.invisible/spawn_point_10/step_41/sphere"
for file_name in os.listdir(folder):
    if file_name.endswith('.npy'):
        file_path = os.path.join(folder, file_name)
        print(file_path)
        with open(file_path, "rb") as f:
            sphere_dataloader.append(np.load(f))
        batch = torch.from_numpy(sphere_dataloader[0])
        ray_origins = batch[:, :3]
        ray_directions = batch[:, 3:6]
        #visualize_random_rays(ray_origins.detach().cpu().numpy(), ray_directions.detach().cpu().numpy(), num_rays=len(ray_origins), ray_length=1.0, name = 'test rays.png')
        print((ray_origins.size(0)))
        print(torch.ones((ray_origins.size(0),1))  * 6)
        print((torch.ones((ray_origins.size(0),1))  * 6).shape)
        exit(0)
        sphere_dataloader = []