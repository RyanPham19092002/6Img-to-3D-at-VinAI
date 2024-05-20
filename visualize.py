import sys
sys.path.append('/root/VinAI/6Img-to-3D-at-VinAI')
from dataloader.rays_dataset import RaysDataset

import open3d as o3d
import numpy as np

def visualize_rays(rays_o, rays_d, ray_length=1.0):
    line_sets = []
    for ray_o, ray_d in zip(rays_o, rays_d):
        ray_end = ray_o + ray_length * ray_d
        points = [ray_o, ray_end]
        lines = [[0, 1]]
        colors = [[1, 0, 0]]  # Red color for rays
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector(lines),
        )
        line_set.colors = o3d.utility.Vector3dVector(colors)
        line_sets.append(line_set)
    o3d.visualization.draw_geometries(line_sets)

if __name__ == "__main__":
    config_path = "path/to/config"
    config = None  # Load or define your config
    dataset_config = None  # Load or define your dataset config
    mode = "val"
    factor = 1
    dataset = RaysDataset(config_path, config, dataset_config, mode, factor)
    rays_o, rays_d = dataset.get_rays_for_visualization()
    visualize_rays(rays_o, rays_d)
