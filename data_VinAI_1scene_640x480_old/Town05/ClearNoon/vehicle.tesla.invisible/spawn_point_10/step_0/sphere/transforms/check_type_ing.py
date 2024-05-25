from PIL import Image

depth_path = './sphere_dataset_log_depth/spherical_1.png'
depth_map = Image.open(depth_path)
image_mode = depth_map.mode

if image_mode == "RGB":
    print("Ảnh là ảnh màu sắc (RGB)")

    # Lấy thông tin kích thước ảnh
    width, height = depth_map.size

    # In giá trị của từng pixel
    for y in range(height):
        for x in range(width):
            pixel_value = depth_map.getpixel((x, y))
            print(f"Pixel ảnh màu tại tọa độ ({x}, {y}): {pixel_value}")

    grayscale_image = depth_map.convert('L')
    width_g, height_g = grayscale_image.size

    # In giá trị của từng pixel
    for y in range(height_g):
        for x in range(width_g):
            pixel_value = grayscale_image.getpixel((x, y))
            print(f"Pixel gray scale tại tọa độ ({x}, {y}): {pixel_value}")
    # Lưu ảnh đen trắng
    #grayscale_image.save('./sphere_dataset_log_depth/grayscale_image.png')
elif image_mode == "L":
    print("Ảnh là ảnh đen trắng (grayscale)")

else:
    print("Ảnh không thuộc dạng màu sắc hoặc đen trắng")