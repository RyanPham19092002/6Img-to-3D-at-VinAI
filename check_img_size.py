import os
from PIL import Image

folder_path = "/root/VinAI/6Img-to-3D-at-VinAI/data_VinAI/Town02/ClearNoon/vehicle.tesla.invisible/spawn_point_10/step_0/nuscenes/transforms/input_images_log_depth"

for filename in os.listdir(folder_path):
    if filename.endswith('.png') or filename.endswith('.jpeg'):
        img_path = os.path.join(folder_path, filename)
        img = Image.open(img_path)
        print(f"Image {filename} has size {img.size}.")
