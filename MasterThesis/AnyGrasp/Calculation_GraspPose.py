import os
import argparse
import torch
import numpy as np
import open3d as o3d
from PIL import Image 

from gsnet import AnyGrasp
from graspnetAPI import GraspGroup

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
parser.add_argument('--max_gripper_width', type=float, default=0.1, help='Maximum gripper width (<=0.1m)')
parser.add_argument('--gripper_height', type=float, default=0.03, help='Gripper height')
parser.add_argument('--top_down_grasp', action='store_true', help='Output top-down grasps.')
parser.add_argument('--debug', action='store_true', help='Enable debug mode')
cfgs = parser.parse_args()
cfgs.max_gripper_width = max(0, min(0.1, cfgs.max_gripper_width))

def demo(data_dir):
    anygrasp = AnyGrasp(cfgs)
    anygrasp.load_net()

    # get data
    # colors = np.array(Image.open(os.path.join(data_dir, 'color.png')), dtype=np.float32) / 255.0
    # depths = np.array(Image.open(os.path.join(data_dir, 'depth.png')))
    colors = np.array(Image.open(os.path.join(data_dir, 'rgb_440_0.png')).convert("RGB"), dtype=np.float32) / 255.0
    depths = np.load(os.path.join(data_dir, 'distance_to_image_plane_440_0.npy'))
    #print('colors data type:', colors.dtype) 
    #plt.imshow(colors)
    #plt.axis('off')  
    #plt.show()
    print('depths data type:', depths.dtype) 
    
    # camera data
    width = 1280
    height = 720
    focal_length = 24 # mm
    horiz_aperture = 20.955 # mm
    
    # get camera intrinsics
    fx = (focal_length/horiz_aperture) * width
    fy = fx # should be same in omniverse
    print("fx:", fx)
    print("fy:", fy)
    cx, cy = width/2, height/2 # width is 1280 and height is 720
    scale = 1 # for omniverse camera, since the distance unit of omniverse is also meter
    
    # set workspace to filter output grasps
    xmin, xmax = -0.5, 0.5
    ymin, ymax = -0.5, 0.5
    zmin, zmax = 0.0, 1.0

    lims = [xmin, xmax, ymin, ymax, zmin, zmax]

    # get point cloud
    xmap, ymap = np.arange(depths.shape[1]), np.arange(depths.shape[0])
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = depths / scale
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z

    # set your workspace to crop point cloud
    mask = (points_z > 0) & (points_z < 1)
    points = np.stack([points_x, points_y, points_z], axis=-1)
    points = points[mask].astype(np.float32)
    colors = colors[mask].astype(np.float32)
    print(points.min(axis=0), points.max(axis=0))
    print('points type and shape:', points.dtype, points.shape)
    print('colors type and shape:', colors.dtype, colors.shape)


    gg, cloud = anygrasp.get_grasp(points, colors, lims=lims, apply_object_mask=True, dense_grasp=False, collision_detection=True)

    if len(gg) == 0:
        print('No Grasp detected after collision detection!')

    gg = gg.nms().sort_by_score()
    gg_pick = gg[0:20]
    print(gg_pick.scores)
    print('best grasp translation:', gg_pick[0].translation) # get translation and rotation
    print('best grasp rotation matrix:', gg_pick[0].rotation_matrix)
    print('grasp score:', gg_pick[0].score)

    # visualization
    if cfgs.debug:
        trans_mat = np.array([[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])
        cloud.transform(trans_mat)
        grippers = gg.to_open3d_geometry_list()
        if len(grippers) == 0:
            print("No grippers available for visualization.")
            return
        for gripper in grippers:
            gripper.transform(trans_mat)
        o3d.visualization.draw_geometries([*grippers, cloud])
        o3d.visualization.draw_geometries([grippers[0], cloud])


if __name__ == '__main__':
    
    demo('./example_data/')
