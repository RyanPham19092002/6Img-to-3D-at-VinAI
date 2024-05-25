from PIL import Image

depth_path = './VinAI/6Img-to-3D-at-VinAI/data_VinAI_1600/Town05/ClearNoon/vehicle.tesla.invisible/spawn_point_10/step_0/sphere/transforms/sphere_dataset_log_depth/spherical_1.png'
depth_map = Image.open(depth_path)
image_mode = depth_map.mode

if image_mode == "RGB":
    print("Ảnh là ảnh màu sắc (RGB)")
elif image_mode == "L":
    print("Ảnh là ảnh đen trắng (grayscale)")
else:
    print("Ảnh không thuộc dạng màu sắc hoặc đen trắng")
