import os
from PIL import Image

folder_path = "/root/VinAI/6Img-to-3D-at-VinAI/data_VinAI/Town05/ClearNoon/vehicle.tesla.invisible/spawn_point_10/step_0/sphere/transforms/sphere_dataset"

for filename in os.listdir(folder_path):
    if filename.endswith('.jpeg'):
        img_path = os.path.join(folder_path, filename)
        img = Image.open(img_path)
        png_path = os.path.splitext(img_path)[0] + ".png"
        
        img.save(png_path)
        os.remove(img_path)
        print(f"Converted {filename} to PNG format.")
