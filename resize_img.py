import os
from PIL import Image
for imgdepth in ["depthmap", "nonedepth"]:
    for town in ["Town02", "Town05"]:
        for typeimg in ["nuscenes", "sphere"]:
            for index in range(51,52):
                print("town - index---------------------", town, index)
                
                if typeimg == "nuscenes":
                    foldername = "input_images"
                elif typeimg == "sphere":
                    foldername = "sphere_dataset"
                if imgdepth == "depthmap":
                    foldername = foldername + "_log_depth"
                folder_path = f"./data_VinAI/{town}/ClearNoon/vehicle.tesla.invisible/spawn_point_10/step_{index}/{typeimg}/transforms/{foldername}"

                #folder_path = f"./VinAI/6Img-to-3D-at-VinAI/data_VinAI/{town}/ClearNoon/vehicle.tesla.invisible/spawn_point_10/step_{index}/sphere/transforms/sphere_dataset_log_depth"


                for filename in os.listdir(folder_path):
                    if filename.endswith('.png') or filename.endswith('.jpeg'):
                        img_path = os.path.join(folder_path, filename)
                        img = Image.open(img_path)
                        if typeimg == "nuscenes":
                            new_img = img.resize((1600, 928))
                        elif typeimg == "sphere":
                            new_img = img.resize((800, 600))
                        new_img.save(img_path)
                        print(f"Resized image {filename} in {folder_path} successfully.")
