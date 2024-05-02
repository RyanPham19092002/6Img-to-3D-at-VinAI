import os
import sys
current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
current_directory = os.path.dirname(current_directory)
rays_dataset_path = os.path.join(current_directory, "dataloader")
rays_dataset_path = os.path.join(rays_dataset_path, "rays_dataset.py")
print(rays_dataset_path)
sys.path.append(current_directory)

from dataloader.rays_dataset import RaysDataset