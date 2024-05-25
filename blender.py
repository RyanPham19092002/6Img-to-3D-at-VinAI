import bpy
import sys

# Đường dẫn đến file .ply cần mở
filepath = './VinAI/6Img-to-3D-at-VinAI/3d_points_visualization.ply'  # Thay thế đường dẫn này bằng đường dẫn thực tế của file .ply

# Xóa mọi đối tượng hiện tại trong scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Mở file .ply
bpy.ops.import_mesh.ply(filepath=filepath)

# Di chuyển camera và zoom để hiển thị toàn bộ đối tượng
bpy.ops.view3d.camera_to_view_selected()
bpy.ops.view3d.view_all(center=False)

# Render
bpy.ops.render.opengl(write_still=True)

# Đóng Blender sau khi hoàn thành
bpy.ops.wm.quit_blender()
