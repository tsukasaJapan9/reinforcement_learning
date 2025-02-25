import math

import pybullet as p
import numpy as np

ray_num = 10
ray_from = []
ray_to = []
ray_ids = []
ray_len = 2.9
ray_start_len = 0.1
ray_color = [0, 1, 0]
ray_hit_color = [1, 0, 0]


def set(model, lidar):
    for i in range(ray_num):
        ray_x = math.sin(-0.25*math.pi + 0.75*2.0*math.pi*float(i)/ray_num)
        ray_y = math.cos(-0.25*math.pi + 0.75*2.0*math.pi*float(i)/ray_num)
        ray_from.append([ray_x*ray_start_len, ray_y*ray_start_len, 0])
        ray_to.append([ray_x*ray_len, ray_y*ray_len, 0])
        ray_ids.append(
            p.addUserDebugLine(
                ray_from[i], 
                ray_to[i], 
                ray_color, 
                parentObjectUniqueId=model, 
                parentLinkIndex=lidar
            )
        )

def detection(model, lidar):
    distance = np.zeros(ray_num)
    results = p.rayTestBatch(
        ray_from, 
        ray_to, 
        parentObjectUniqueId=model, 
        parentLinkIndex=lidar
    )

    for i in range(ray_num):
        hit_detection = results[i][2]

        if hit_detection == 1.0:
            p.addUserDebugLine(
                ray_from[i], 
                ray_to[i], 
                ray_color, 
                replaceItemUniqueId=ray_ids[i],
                parentObjectUniqueId=model, 
                parentLinkIndex=lidar
            )
            distance[i] = ray_start_len + ray_len
        else:
            x_hit = ray_from[i][0] + hit_detection * (ray_to[i][0] - ray_from[i][0])
            y_hit = ray_from[i][1] + hit_detection * (ray_to[i][1] - ray_from[i][1])
            z_hit = ray_from[i][2] + hit_detection * (ray_to[i][2] - ray_from[i][2])

            local_hit_to = [x_hit, y_hit, z_hit]

            p.addUserDebugLine(
                ray_from[i], 
                local_hit_to, 
                ray_hit_color, 
                replaceItemUniqueId=ray_ids[i],
                parentObjectUniqueId=model, 
                parentLinkIndex=lidar
            )

            distance[i] = np.sqrt(x_hit**2 + y_hit**2)

    return distance