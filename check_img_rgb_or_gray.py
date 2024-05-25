from PIL import Image

<<<<<<< HEAD
depth_path = './data_VinAI_10scene_1600x928/Town05/ClearNoon/vehicle.tesla.invisible/spawn_point_10/step_50/sphere/transforms/sphere_dataset_log_depth/spherical_1.png'
=======
depth_path = './VinAI/6Img-to-3D-at-VinAI/data_VinAI_1600/Town05/ClearNoon/vehicle.tesla.invisible/spawn_point_10/step_0/sphere/transforms/sphere_dataset_log_depth/spherical_1.png'
>>>>>>> 2915764262959867fa18eaad97414b8be5178ba1
depth_map = Image.open(depth_path)
image_mode = depth_map.mode

if image_mode == "RGB":
    print("Ảnh là ảnh màu sắc (RGB)")
elif image_mode == "L":
    print("Ảnh là ảnh đen trắng (grayscale)")
else:
    print("Ảnh không thuộc dạng màu sắc hoặc đen trắng")
