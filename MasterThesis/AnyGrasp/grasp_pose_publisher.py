# grasp_pose_publisher.py
#
# This script functions as a grasp pose generator for robotic manipulation tasks.
# Given RGBD images of target objects, it:
# 1. Processes the images to generate point clouds
# 2. Uses the AnyGrasp model to compute optimal grasp poses
# 3. Filters and validates grasp poses based on robot constraints
# 4. Publishes the best grasp pose for robot execution
#
# Note: AnyGrasp model uses the Minkowski Engine library for sparse convolution operations.
# This script must be run in an environment with Minkowski Engine properly installed.
# Minkowski Engine requires specific CUDA and PyTorch versions - please refer to
# https://github.com/NVIDIA/MinkowskiEngine for compatibility and installation instructions.
#
# The script communicates through ROS nodes, receiving image paths as input
# and publishing grasp poses (translation and rotation) as output for robot control.

import argparse
import numpy as np
from PIL import Image 

import zmq
import json
import time

from gsnet import AnyGrasp

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
parser.add_argument('--max_gripper_width', type=float, default=0.1, help='Maximum gripper width (<=0.1m)')
parser.add_argument('--gripper_height', type=float, default=0.03, help='Gripper height')
parser.add_argument('--top_down_grasp', action='store_true', help='Output top-down grasps.')
parser.add_argument('--debug', action='store_true', help='Enable debug mode')
cfgs = parser.parse_args()
cfgs.max_gripper_width = max(0, min(0.1, cfgs.max_gripper_width))

def is_rotation_exceeding_threshold(R2, threshold_deg=30):
    """
    Check if the absolute angle difference between rotation matrix R2 and R1 along x-axis exceeds the threshold.
    
    Args:
        R2: Target rotation matrix to check
        threshold_deg: Threshold angle in degrees (default: 30)
        
    Returns:
        bool: True if rotation exceeds threshold, False otherwise
    """
    R1 = np.array([[0.0, 0.0, 1.0],
                   [0.0, -1.0, 0.0],
                   [1.0, 0.0, 0.0]])
    threshold_rad = np.deg2rad(threshold_deg)
    
    x_axis_R1 = R1[:, 0]  # x-axis direction from R1
    x_axis_R2 = R2[:, 0]  # x-axis direction from R2

    dot_product = np.dot(x_axis_R1, x_axis_R2)
    cos_theta = np.clip(dot_product, -1.0, 1.0)  # Prevent numerical errors
    theta = np.arccos(cos_theta)

    return theta > threshold_rad

class VisionModel:
    """
    A class that handles grasp pose detection using the AnyGrasp model.
    Processes RGBD images and generates grasp poses for robotic manipulation.
    """
    def __init__(self):
        self.anygrasp = AnyGrasp(cfgs)
        self.anygrasp.load_net()

        # Camera parameters from omniverse
        self.width = 1280
        self.height = 720
        self.focal_length = 24  # mm
        self.horiz_aperture = 20.955  # mm

        # Calculate camera intrinsics
        self.fx = (self.focal_length/self.horiz_aperture) * self.width
        self.fy = self.fx  # Should be same in Omniverse
        self.cx, self.cy = self.width/2, self.height/2
        self.scale = 1  # For Omniverse camera (distance unit is meter)
    
        # Workspace limits for filtering output grasps
        self.xmin, self.xmax = -0.2, 0.5
        self.ymin, self.ymax = -0.5, 0.5
        self.zmin, self.zmax = 0.6, 1.0

        self.lims = [self.xmin, self.xmax, self.ymin, self.ymax, self.zmin, self.zmax]
    
    def process(self, rgb_path, distance_path):
        """
        Process RGB and depth images to generate grasp poses.
        
        Args:
            rgb_path: Path to the RGB image
            distance_path: Path to the depth image
            
        Returns:
            dict: Contains translation and rotation matrix for the best grasp pose
        """
        try:
            colors = np.array(Image.open(rgb_path).convert("RGB"), dtype=np.float32) / 255.0
            depths = np.load(distance_path)
        except Exception as e:
            print(f"Error loading depth file {distance_path}: {e}")

        # Generate point cloud from depth image
        xmap, ymap = np.arange(depths.shape[1]), np.arange(depths.shape[0])
        xmap, ymap = np.meshgrid(xmap, ymap)
        points_z = depths / self.scale
        points_x = (xmap - self.cx) / self.fx * points_z
        points_y = (ymap - self.cy) / self.fy * points_z

        # Filter points within workspace
        mask = (points_z > 0) & (points_z < 1)
        points = np.stack([points_x, points_y, points_z], axis=-1)
        points = points[mask].astype(np.float32)
        colors = colors[mask].astype(np.float32)

        gg, cloud = self.anygrasp.get_grasp(points, colors, lims=self.lims, apply_object_mask=True, dense_grasp=False, collision_detection=True)

        if len(gg) == 0:
            print('No grasp detected after collision detection!')
            return {"translation": None, "rotation": None}

        gg = gg.nms().sort_by_score()
        gg_pick = gg[0:20]

        valid_grasp = []
        for i in gg_pick:
            if not is_rotation_exceeding_threshold(i.rotation_matrix):
                valid_grasp.append({
                    "translation": i.translation.tolist(),
                    "rotation": i.rotation_matrix.tolist()
                })
        if valid_grasp:
            return valid_grasp[0]
        else:
            print(f"No valid grasp after filtering")
            return{"translation": None, "rotation": None}

def convert_to_serializable(obj):
    """
    Recursively convert dictionary containing numpy arrays to JSON serializable format.
    
    Args:
        obj: Input object to convert
        
    Returns:
        Converted object with numpy arrays converted to lists
    """
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def main():
    """
    Main function that initializes ZMQ communication and handles the message loop
    for receiving image paths and publishing grasp poses.
    """
    # Initialize vision model
    model = VisionModel()

    context = zmq.Context()

    # SUB socket: Receive ROS 2 data from bridge node
    sub_socket = context.socket(zmq.SUB)
    sub_socket.connect("tcp://localhost:5555")
    sub_socket.setsockopt_string(zmq.SUBSCRIBE, '')  # Subscribe to all messages

    # PUB socket: Send processed data back to bridge node
    pub_socket = context.socket(zmq.PUB)
    pub_socket.bind("tcp://*:5556")

    print("Publisher is up and listening for messages...")

    # Wait for PUB socket to complete binding
    time.sleep(1)

    take_foto_wp_data = []
    file_paths_data = {}
    simulation_cycle = 0
    grasp_results = {}
    previous_file_paths_nonempty = False  # Track if previous file_paths was non-empty
    
    try:
        while True:
            try:
                message = sub_socket.recv_string()
                data = json.loads(message)

                # Handle different message topics
                if data['topic'] == 'sm_state_wp':
                    print(f"Received from sm_state_wp: {data['data']}")
                elif data['topic'] == 'take_foto_wp':
                    take_foto_wp_data = data['data']
                    print(f"Received from take_foto_wp: {data['data']}")
                elif data['topic'] == 'file_paths':
                    file_paths_data = data['data']
                    print(f"Received from file_paths: {data['data']}")

                    # Check for new simulation cycle
                    if previous_file_paths_nonempty and not file_paths_data:
                        simulation_cycle += 1
                        print(f"New simulation cycle detected: {simulation_cycle}")
                        grasp_results = {}

                    previous_file_paths_nonempty = bool(file_paths_data)

                    # Process environment data if conditions are met
                    if file_paths_data and not grasp_results:
                        env_count = len(take_foto_wp_data)
                        print(f"Detected {env_count} environments.")

                        # Generate grasp poses for each environment
                        for env_id, paths in file_paths_data.items():
                            rgb_path = paths.get('rgb')
                            distance_path = paths.get('distance_to_image_plane')
                            if not rgb_path or not distance_path:
                                print(f'Missing data for environment {env_id}. Skipping...')
                                continue
                            grasp_result = model.process(rgb_path, distance_path)
                            grasp_results[env_id] = grasp_result

                    print(f"Grasp results for cycle {simulation_cycle}: {grasp_results}")

                    # Publish grasp results back to bridge node
                    response = {
                        "cycle": simulation_cycle,
                        "grasp_results": convert_to_serializable(grasp_results)
                    }
                    pub_socket.send_string(json.dumps(response))
                    print(f"Sent grasp results for cycle {simulation_cycle} back to bridge node.")

                else:
                    print(f"Unknown topic: {data['topic']}")
    
            except json.JSONDecodeError:
                print("Error: Received message is not valid JSON.")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
    except KeyboardInterrupt:
        print("Shutting down grasp pose publisher.")
        
    finally:
        sub_socket.close()
        pub_socket.close()
        context.term()

if __name__ == '__main__':
    main() 