# Calculation_GraspPose.py
#
# This script demonstrates the grasp pose calculation using the AnyGrasp model.
# It processes RGBD images to generate and visualize optimal grasp poses for robotic manipulation.
# The script includes visualization tools for both RGB and depth images, as well as 3D grasp poses.

import os
import argparse
import torch
import numpy as np
import open3d as o3d
from PIL import Image 
import matplotlib.pyplot as plt

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
    """
    Demonstrates grasp pose calculation and visualization using AnyGrasp model.
    
    Args:
        data_dir (str): Directory containing RGB and depth images
        
    The function performs the following steps:
    1. Loads and initializes the AnyGrasp model
    2. Processes RGB and depth images
    3. Generates point cloud from depth data
    4. Calculates grasp poses
    5. Visualizes results if debug mode is enabled
    """
    # Initialize AnyGrasp model
    anygrasp = AnyGrasp(cfgs)
    anygrasp.load_net()

    # Load RGB and depth data
    colors = np.array(Image.open(os.path.join(data_dir, 'rgb_1_0.png')).convert("RGB"), dtype=np.float32) / 255.0
    depths = np.load(os.path.join(data_dir, 'distance_to_image_plane_1_0.npy'))
    #print('colors data type:', colors.dtype) 
    #print('depths data type:', depths.dtype) 
    
    # Visualize RGB and depth images
    plt.figure(figsize=(12, 6))

    # RGB image visualization
    plt.subplot(1, 2, 1)
    plt.imshow(colors)
    plt.axis('off')
    plt.title("Color Image")

    # Depth image visualization
    plt.subplot(1, 2, 2)
    depth_plot = plt.imshow(depths, cmap='gray')
    plt.axis('off')
    plt.title("Depth Image")
    cbar = plt.colorbar(depth_plot, ax=plt.gca(), fraction=0.046, pad=0.04)
    cbar.set_label("Depth (meters)")

    plt.tight_layout()
    plt.show()
    
    # Camera parameters from omniverse
    width = 1280
    height = 720
    focal_length = 24  # mm
    horiz_aperture = 20.955  # mm
    
    # Calculate camera intrinsics
    fx = (focal_length/horiz_aperture) * width
    fy = fx  # Same as fx in Omniverse
    print("fx:", fx)
    print("fy:", fy)
    cx, cy = width/2, height/2  # Principal point (image center)
    scale = 1  # Scale factor (1 for Omniverse as it uses meters)
    
    # Generate point cloud from depth image
    xmap, ymap = np.arange(depths.shape[1]), np.arange(depths.shape[0])
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = depths / scale
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z

    # Filter points within workspace
    mask = (points_z > 0) & (points_z < 1)
    points = np.stack([points_x, points_y, points_z], axis=-1)
    points = points[mask].astype(np.float32)
    colors = colors[mask].astype(np.float32)

    print('Points array shape:', points.dtype, points.shape)
    print('Colors array shape:', colors.dtype, colors.shape)

    # Define workspace boundaries for grasp filtering
    xmin, xmax = -0.5, 0.5
    ymin, ymax = -0.5, 0.5
    zmin, zmax = 0.5, 0.8

    lims = [xmin, xmax, ymin, ymax, zmin, zmax]

    # Generate grasp poses
    gg, cloud = anygrasp.get_grasp(points, colors, lims=lims, apply_object_mask=True, dense_grasp=False, collision_detection=True)

    if len(gg) == 0:
        print('No grasp poses detected after collision detection!')
        return

    # Process and sort grasp poses by score
    gg = gg.nms().sort_by_score()
    gg_pick = gg[0:20]
    print("Grasp scores:", gg_pick.scores)
    print('Best grasp translation:', gg_pick[0].translation)
    print('Best grasp rotation matrix:', gg_pick[0].rotation_matrix)
    print('Best grasp score:', gg_pick[0].score)

    # Visualize grasp poses in 3D if debug mode is enabled
    if cfgs.debug:
        # Transform for visualization (flip z-axis)
        trans_mat = np.array([[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])
        cloud.transform(trans_mat)
        grippers = gg.to_open3d_geometry_list()
        if len(grippers) == 0:
            print("No grippers available for visualization.")
            return
        for gripper in grippers:
            gripper.transform(trans_mat)
        # Visualize all grasp poses
        o3d.visualization.draw_geometries([*grippers, cloud])
        # Visualize best grasp pose
        o3d.visualization.draw_geometries([grippers[0], cloud])

if __name__ == '__main__':
    demo('./example_data/')
