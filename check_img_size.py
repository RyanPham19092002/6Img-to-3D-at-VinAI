import os
from PIL import Image

<<<<<<< HEAD
for town in ["Town02", "Town05"]:
    for index in range(41,52):
        print("town - index---------------------", town, index)
        folder_path = f"./data_VinAI/{town}/ClearNoon/vehicle.tesla.invisible/spawn_point_10/step_{index}/nuscenes/transforms/input_images_log_depth"
=======
folder_path = "./VinAI/6Img-to-3D-at-VinAI/data_VinAI_1600/Town05/ClearNoon/vehicle.tesla.invisible/spawn_point_10/step_0/nuscenes/transforms/input_images_log_depth"
>>>>>>> 2915764262959867fa18eaad97414b8be5178ba1

        for filename in os.listdir(folder_path):
            if filename.endswith('.png') or filename.endswith('.jpeg'):
                img_path = os.path.join(folder_path, filename)
                img = Image.open(img_path)
                print(f"Image {filename} has size {img.size}.")
