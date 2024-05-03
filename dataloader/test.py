import os
import numpy as np
from torch.utils import data
from mmcv.image.io import imread
import json
import random





data_path = "./data_NEO360/Town01/ClearNoon/vehicle.tesla.invisible/spawn_point_10/step_0"

data = dict(
    town = 'Town1',
    weather = 'ClearNoon',
    vehicle = 'vehicle.tesla.invisible',
    spawn_point = "spawn_point_10",
    step = "step_0",
    nuscenes = os.path.join(data_path, "nuscenes"),
    sphere = os.path.join(data_path, "sphere"),
)
with open(os.path.join(data["nuscenes"], "transforms", "transforms_ego.json"), "r") as f:
    
    input_data = json.load(f)
    input_rgb = []
    all_c2w = []
    image_path_list = []
    K = np.zeros((3,4)) # (C,3,4)
    # K[0,0] = input_data['fl_x']             #Focus_length_x = ImageSizeX /(2 * tan(CameraFOV * π / 360))
    K[0,0] = input_data['img_size'][0] / (2*np.tan(input_data['fov'] * np.pi / 360))
    # K[1,1] = input_data['fl_y']             #Focus_length_y = ImageSizey /(2 * tan(CameraFOV * π / 360))
    K[1,1] = input_data['img_size'][1] / (2*np.tan(input_data['fov'] * np.pi / 360))
    K[2,2] = 1
    # K[0,2] = input_data['cx']               #ImageSizeX / 2
    K[0,2] = input_data['img_size'][0] / 2
    # K[1,2] = input_data['cy']               #ImageSizeY / 2
    K[0,2] = input_data['img_size'][1] / 2

    frame = input_data["transform"]
    for key in frame.keys():
        file_path = key + '.png'
        print(file_path)
        print(frame[key])
        image_path = os.path.join(data["nuscenes"], "transforms", "input_images", file_path)
        image_path.replace("\\", "/")

        image_path_list.append(image_path)
        input_rgb.append(imread(image_path, "unchanged")[:,:,:3].astype(np.float32))
        all_c2w.append(frame[key])

