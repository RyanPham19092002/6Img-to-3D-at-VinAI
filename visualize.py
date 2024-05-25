import sys
sys.path.append('./VinAI/6Img-to-3D-at-VinAI')
from dataloader.rays_dataset import RaysDataset
from mmengine.config import Config
from matplotlib import pyplot as plt
import open3d as o3d
import numpy as np
import plotly.graph_objects as go
import torch

o3d.visualization.webrtc_server.enable_webrtc()

def visualize_random_rays(rays_o, rays_d, num_rays=100, ray_length=1.0):
    # Select n random rays
    indices = np.random.choice(len(rays_o), num_rays, replace=False)
    selected_rays_o = rays_o[indices]
    selected_rays_d = rays_d[indices]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the starting points of the rays in red
    ax.scatter(selected_rays_o[:, 0], selected_rays_o[:, 1], selected_rays_o[:, 2], color='r')

    # Plot the rays
    for i in range(num_rays):
        start_point = selected_rays_o[i]
        end_point = start_point + ray_length * selected_rays_d[i]
        ax.plot([start_point[0], end_point[0]],
                [start_point[1], end_point[1]],
                [start_point[2], end_point[2]], color='b')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.savefig('visualize_ray_start_to_end.png')


#wrong---------------------------------------------------------------------------------------------------------
def visualize_3d_points_and_directions(points, directions, output_file='visualize_x_world_and_directions.png'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Convert tensors to numpy arrays for plotting
    sampled_points_np = points.cpu().numpy()
    sampled_directions_np = directions.cpu().numpy()

    # Scatter plot of points
    #ax.scatter(sampled_points_np[:, 0], sampled_points_np[:, 1], sampled_points_np[:, 2], color='r', s=1, label='Points')

    # Plotting directions as arrows
    for i in range(sampled_directions_np.shape[0]):
        ax.plot(
            #sampled_points_np[i, 0], sampled_points_np[i, 1], sampled_points_np[i, 2],
            [sampled_directions_np[i, 0]], sampled_directions_np[i, 1], sampled_directions_np[i, 2],
            length=0.1, normalize=True, color='b'
        )

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Points and Viewing Directions')
    ax.legend()
    print("input Nerf model")
    plt.savefig('visualize_x_world_and_directions.png')

def visualize_triplane(triplane):
    # Kiểm tra số lượng tensor trong tuple
    num_planes = len(triplane)
    
    for plane_idx in range(num_planes):
        plane = triplane[plane_idx]  # Lấy tensor tương ứng từ tuple
        batch_size = plane.shape[0]
        num_channels = plane.shape[1]
        
        print(f'Visualizing plane {plane_idx+1} with shape {plane.shape}')
        
        fig, axes = plt.subplots(batch_size, num_channels, figsize=(15, 5 * batch_size))
        if batch_size == 1:
            axes = [axes]  # Đảm bảo axes là một danh sách khi chỉ có một batch
        
        for b in range(batch_size):
            for c in range(num_channels):
                ax = axes[b][c]
                ax.imshow(plane[b, c, :, :].detach().cpu().numpy(), cmap='viridis')
                ax.axis('off')
        plt.savefig('visualize_x_world_and_directions.png')

def visualize_points(points):
    points_np = points.cpu().numpy()
    num_batches = points_np.shape[0]
    num_rays = points_np.shape[1]

    # Tạo một figure và các axes cho các hình chiếu 2D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Lặp qua tất cả các batch và rays để lấy các điểm (x, y, z) từ points
    for batch_index in range(num_batches):
        for ray_index in range(100):
            print("batch_index -- ray_index", batch_index, ray_index)
            x = points_np[batch_index, ray_index, :, 0]
            y = points_np[batch_index, ray_index, :, 1]
            z = points_np[batch_index, ray_index, :, 2]

            # Vẽ hình chiếu XY
            # axes[0].scatter(x, y, s=10, alpha=0.5)

            # # Vẽ hình chiếu XZ
            # axes[1].scatter(x, z, s=10, alpha=0.5)

            # # Vẽ hình chiếu YZ
            # axes[2].scatter(y, z, s=10, alpha=0.5)
            ax.scatter(x, y, z, color='r')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    #plt.savefig('visualization_img.png')

    # # Thiết lập tiêu đề và nhãn trục cho các hình chiếu
    # axes[0].set_title('XY Projection')
    # axes[0].set_xlabel('X Axis')
    # axes[0].set_ylabel('Y Axis')

    # # axes[1].set_title('XZ Projection')
    # # axes[1].set_xlabel('X Axis')
    # # axes[1].set_ylabel('Z Axis')

    # # axes[2].set_title('YZ Projection')
    # # axes[2].set_xlabel('Y Axis')
    # # axes[2].set_ylabel('Z Axis')
    print("In ảnh")
    # Lưu ảnh lại
    plt.savefig('all_points_2d_projections_1.png')
    print("save")

def visualize_point3d(points):
    points_np = points.cpu().numpy()
    num_batches = points_np.shape[0]
    num_rays = points_np.shape[1]

    # Tạo danh sách các điểm 3D
    x_all = []
    y_all = []
    z_all = []

    # Lặp qua tất cả các batch và rays để lấy các điểm (x, y, z) từ points
    for batch_index in range(num_batches):
        for ray_index in range(100):
            print("batch_index -- ray_index", batch_index, ray_index)
            x = points_np[batch_index, ray_index, :, 0]
            y = points_np[batch_index, ray_index, :, 1]
            z = points_np[batch_index, ray_index, :, 2]
            
            # Extend the lists with the current ray's points
            x_all.extend(x)
            y_all.extend(y)
            z_all.extend(z)
    
    # Print out the lengths of the lists to ensure data is collected
    print(f"Total points collected: {len(x_all)}")

    # Create an Open3D PointCloud object
    points_array = np.vstack((x_all, y_all, z_all)).T
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_array)

    # Save the PointCloud object as a .ply file
    print("Lưu 3d")
    o3d.io.write_point_cloud('3d_points_visualization.ply', point_cloud)
    print("3D visualization saved to '3d_points_visualization.ply'")
    print("save 3d")

def visualize_triplane(triplane):
    fig = plt.figure(figsize=(15, 5))
    
    for i, wing in enumerate(triplane, 1):
        ax = fig.add_subplot(1, 3, i, projection='3d')
        position = torch.tensor([0, 0, 0])  # Mặc định vị trí của mặt là (0, 0, 0)
        length = wing.shape[1]  # Chiều dài của mặt
        width = wing.shape[2]   # Chiều rộng của mặt

        x = [position[0], position[0] + length]
        y = [position[1], position[1]]
        z = [position[2], position[2] + width]

        ax.plot(x, y, z)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Wing {i}')

    fig.suptitle('Triplane Visualization')
    plt.savefig('triplane.png')
#wrong---------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    datapath = "/root/VinAI/6Img-to-3D-at-VinAI/data_VinAI/Town05/ClearNoon/vehicle.tesla.invisible/spawn_point_10/step_0/sphere/"
    config = "config/config.py"  # Load or define your config
    dataset_config = "config/_base_/dataset.py"  # Load or define your dataset config

    config = Config.fromfile(config)
    dataset_config = Config.fromfile(dataset_config).dataset_params

    dataset = RaysDataset(datapath, config, dataset_config=dataset_config.train_data_loader, mode="train",
                          factor=dataset_config.train_data_loader.factor)
    rays_o, rays_d, img_array = dataset.get_rays_for_visualization()
    #img_array = img.detach().cpu().numpy()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(img_array[:, 0], img_array[:, 1], img_array[:, 2], color='r')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.savefig('visualization_img.png')
    #visualize_random_rays(rays_o, rays_d, num_rays=100)  # Visualize 100 random rays
