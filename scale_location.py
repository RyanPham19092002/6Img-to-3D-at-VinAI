import json
import numpy as np

for town in ["Town02", "Town05"]:
    for index in range(41,52):

        scale_x = (1600/640)
        scale_y = (928/480)
        # if town == "Town02":
        #     mode = "test"
        # elif town == "Town05":
        #     mode = "train"
        print("town - index---------------------", town, index)
        file_json = f"./data_VinAI/{town}/ClearNoon/vehicle.tesla.invisible/spawn_point_10/step_{index}/nuscenes/transforms/transforms_ego.json"
        with open(file_json, 'r') as f:
            input_data = json.load(f)

        frame = input_data["transform"]
        for key in frame.keys():
            frame[key] = np.array(frame[key])
            frame[key][0,3] *= scale_x 
            frame[key][1,3] *= scale_y

            frame[key] = frame[key].tolist()
        
        input_data["img_size"] = [1600,928]

        with open(file_json, 'w') as f:
            json.dump(input_data, f)
        print("Updated data has been saved successfully.")