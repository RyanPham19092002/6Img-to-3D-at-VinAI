import os
from PIL import Image


for town in ["Town02", "Town05"]:
    for index in range(40,52):
        print("town - index---------------------", town, index)
        folder_path = f"./data_VinAI/{town}/ClearNoon/vehicle.tesla.invisible/spawn_point_10/step_{index}/sphere/transforms/sphere_dataset_log_depth"

        for filename in os.listdir(folder_path):
            if filename.endswith('.png') or filename.endswith('.jpeg'):
                img_path = os.path.join(folder_path, filename)
                img = Image.open(img_path)
                print(f"Image {filename} has size {img.size}.")
