from PIL import Image

depth_path = '/root/VinAI/6Img-to-3D-at-VinAI/data_VinAI/Town05/ClearNoon/vehicle.tesla.invisible/spawn_point_10/step_0/nuscenes/transforms/input_images_log_depth/input_camera_1.png'
depth_map = Image.open(depth_path)
image_mode = depth_map.mode

if image_mode == "RGB":
    print("Ảnh là ảnh màu sắc (RGB)")
elif image_mode == "L":
    print("Ảnh là ảnh đen trắng (grayscale)")
else:
    print("Ảnh không thuộc dạng màu sắc hoặc đen trắng")