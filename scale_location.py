import json
import numpy as np

scale_x = (1600/1280)
scale_y = (928/800)

file_json = "/root/VinAI/6Img-to-3D-at-VinAI/data_VinAI/Town05/ClearNoon/vehicle.tesla.invisible/spawn_point_10/step_0/sphere/transforms/transforms_ego_train.json"
with open(file_json, 'r') as f:
    input_data = json.load(f)

frame = input_data["transform"]
for key in frame.keys():
    frame[key] = np.array(frame[key])
    frame[key][0,3] *= scale_x 
    frame[key][1,3] *= scale_y

    frame[key] = frame[key].tolist()


with open(file_json, 'w') as f:
    json.dump(input_data, f)
print("Updated data has been saved successfully.")