# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Operating Room Robotic Manipulation Simulation

This script simulates robotic manipulation tasks in an operating room environment using Isaac Sim.
It implements a state machine-based control system that:
1. Captures RGBD images of the scene
2. Communicates with AnyGrasp for grasp pose detection
3. Controls robot motion for pick-and-place operations
4. Validates successful grasps and object placement

Key Components:
- State Machine: Manages the robot's behavioral states (REST, CAMERA, GRASP, etc.)
- ROS 2 Bridge: Communicates with AnyGrasp system for grasp pose computation
- Vision System: Captures and processes RGBD data
- Motion Control: Executes computed trajectories

Usage:
    ./isaaclab.sh -p source/standalone/environments/state_machine/lift_cube_sm.py --num_envs <num_envs>

Parameters:
    --num_envs: Number of parallel environments to simulate
    --disable_fabric: Disable fabric and use USD I/O operations
"""

"""Launch Omniverse Toolkit first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Operating Room Robotic Manipulation Simulation")
parser.add_argument(
    "--disable_fabric", 
    action="store_true", 
    default=False, 
    help="Disable fabric and use USD I/O operations"
)
parser.add_argument(
    "--num_envs", 
    type=int, 
    default=4, 
    help="Number of parallel environments to simulate"
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(headless=args_cli.headless)
simulation_app = app_launcher.app

"""Rest everything else."""

import gymnasium as gym
import os
import torch
from typing import Tuple
from torchtyping import TensorType
from typeguard import typechecked

import omni.replicator.core as rep

from collections.abc import Sequence

import warp as wp

from omni.isaac.lab.assets.rigid_object.rigid_object_data import RigidObjectData

# env registration in gym
import omni.isaac.lab_tasks  # noqa: F401
import my_custom_env # for own use case
from my_custom_env.write_with_path import write_with_path
from omni.isaac.lab.utils import convert_dict_to_backend
from omni.isaac.lab.utils.math import subtract_frame_transforms, matrix_from_quat, quat_from_matrix


from omni.isaac.lab_tasks.utils.parse_cfg import parse_env_cfg

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Header, Float32MultiArray, Int8MultiArray, String
    from sensor_msgs.msg import PointCloud2, PointField
    import threading
    import json
    import time
    import logging
    print("import rclpy success!")
except:
    print("import rclpy failed")

# ROS 2 Node for Isaac Lab Simulation
class IsaacLab(Node):
    """
    ROS 2 Node that handles communication between Isaac Sim and AnyGrasp.
    
    Publishers:
        - sm_state_wp: Current state of the state machine
        - take_foto_wp: Camera trigger signals
        - file_paths: RGBD image file paths
        
    Subscribers:
        - grasp_results: Grasp poses computed by AnyGrasp
    """
    def __init__(self):
        super().__init__('IsaacLab')
        
        # Initialize publishers
        self.state_publisher_ = self.create_publisher(Int8MultiArray, 'sm_state_wp', 10)
        self.foto_publisher_ = self.create_publisher(Float32MultiArray, 'take_foto_wp', 10)
        self.file_paths_publicher_ = self.create_publisher(String, 'file_paths', 10)
        
        # Initialize subscriber for grasp results
        self.grasp_results_subscription = self.create_subscription(
            String,
            'grasp_results',
            self.listener_callback,
            10
        )
        
        # Timer for periodic publishing (0.5 second interval)
        self.timer = self.create_timer(0.5, self.timer_callback)
        
        # Data storage for communication between main process and ROS node
        self.sm_state_wp_data = None  # State machine states
        self.take_foto_wp_data = None  # Camera trigger signals
        self.file_paths_data = None    # RGBD image paths
        self.grasp_results_data = None  # Received grasp poses
        self.translation_env_nums = None  # Grasp translations
        self.rotation_env_nums = None     # Grasp rotations

    def update_data(self, sm_state, take_foto, file_paths):
        """
        Update data to be published to ROS topics.
        
        Args:
            sm_state: State machine states for each environment
            take_foto: Camera trigger signals for each environment
            file_paths: Dictionary of RGBD image file paths
        """
        self.sm_state_wp_data = sm_state
        self.take_foto_wp_data = take_foto
        self.file_paths_data = file_paths

    def timer_callback(self):
        """Periodic callback to publish data to ROS topics."""
        # Publish state machine states
        if self.sm_state_wp_data is not None:
            msg_sm_state = Int8MultiArray()
            msg_sm_state.data = self.sm_state_wp_data.numpy().tolist()
            self.state_publisher_.publish(msg_sm_state)

        # Publish camera trigger signals
        if self.take_foto_wp_data is not None:
            msg_take_foto = Float32MultiArray()
            msg_take_foto.data = self.take_foto_wp_data.numpy().tolist()
            self.foto_publisher_.publish(msg_take_foto)

        # Publish RGBD image file paths
        if self.file_paths_data is not None:
            try:
                file_paths_json = json.dumps(self.file_paths_data)
                msg_file_paths = String()
                msg_file_paths.data = file_paths_json
                self.file_paths_publicher_.publish(msg_file_paths)
            except Exception as e:
                self.get_logger().error(f"Failed to publish file paths: {e}")

    def listener_callback(self, msg): 
        try:
            data = json.loads(msg.data)
            self.grasp_results_data = data.get('grasp_results', {})
            # self.get_logger().info(f'Received data from Python 3.8: {data}')
            # self.get_logger().info(f'Data of grasp_results : {self.grasp_results_data}')

            # Extract grasp results and convert to PyTorch tensors
            num_envs = len(self.grasp_results_data)  # Determine number of environments from grasp results

            # Create lists for translation and rotation tensors
            translations = []
            rotations = []

            for env_id in range(num_envs):
                env_id_str = str(env_id)
                env_result = self.grasp_results_data.get(env_id_str, {})
                translation = env_result.get('translation')
                rotation = env_result.get('rotation')

                # Set default values if translation or rotation is None
                translation = translation if translation is not None else [0.0, 0.0, 0.0]
                rotation = rotation if rotation is not None else [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
                translations.append(torch.tensor(translation, dtype=torch.float32))
                rotations.append(torch.tensor(rotation, dtype=torch.float32))

            # self.get_logger().info(f"translation: {translations}")
            # self.get_logger().info(f"rotation: {rotations}")

            # Convert translation and rotation lists to tensor matrices
            if translations != [] and rotations != []: 
                translation_env_nums = torch.stack(translations)  # Shape: (num_envs, 3)
                rotation_env_nums = torch.stack(rotations)  # Shape: (num_envs, 3, 3)

                # Debug printing
                # self.get_logger().info(f"translation_env_nums: {translation_env_nums}")
                # self.get_logger().info(f"rotation_env_nums: {rotation_env_nums}")

                # Store results as node attributes for use in main program
                self.translation_env_nums = translation_env_nums
                self.rotation_env_nums = rotation_env_nums
            else:
                # Reset attributes if no valid data
                self.grasp_results_data = None
                self.translation_env_nums = None
                self.rotation_env_nums = None
        
        except json.JSONDecodeError as e:
            self.get_logger().error(f"Invalid JSON format: {e}")
            return
        except Exception as e:
            self.get_logger().error(f'Error in listener_callback: {e}')
        
# ROS 2 initialization function
def ros2_node_thread(publisher_node):
    rclpy.spin(publisher_node)

def adjust_matrices_torch(R2) -> torch.Tensor:
    """
    Adjust the R2 tensor to satisfy the following conditions:
    1. The x-axis of each adjusted matrix is parallel to the x-axis of R1.
    2. The projection ratio of each matrix's z-axis onto R1's z-y plane remains constant.
    
    Args:
        R2 (torch.Tensor): A tensor of shape (n, 3, 3) containing n rotation matrices.
    
    Returns:
        torch.Tensor: Adjusted rotation matrix tensor of shape (n, 3, 3).
    """
    # Define the reference rotation matrix R1
    R1 = torch.tensor([[0.0, 0.0, 1.0],
                       [0.0, -1.0, 0.0],
                       [1.0, 0.0, 0.0]], dtype=R2.dtype, device=R2.device)
    
    # Extract axis vectors from R1
    x_axis_R1 = R1[:, 0]
    y_axis_R1 = R1[:, 1]
    z_axis_R1 = R1[:, 2]
    
    # Ensure R2 has shape (n, 3, 3)
    assert R2.ndim == 3 and R2.shape[1:] == (3, 3), "R2 must have shape (n, 3, 3)"
    
    # Adjust R2's x-axis to be parallel with R1's x-axis
    x_axis_R2_new = x_axis_R1.repeat(R2.shape[0], 1)  # (n, 3)
    
    # Get the original z-axis vector from R2
    z_axis_R2 = R2[:, :, 2]  # (n, 3)
    
    # Project R2's z-axis onto R1's z-y plane
    z_proj = torch.einsum('ij,j->i', z_axis_R2, z_axis_R1)[:, None] * z_axis_R1  # (n, 3)
    y_proj = torch.einsum('ij,j->i', z_axis_R2, y_axis_R1)[:, None] * y_axis_R1  # (n, 3)
    
    # Recalculate the adjusted z-axis while maintaining the projection ratio
    z_axis_R2_new = z_proj + y_proj
    z_axis_R2_new = z_axis_R2_new / z_axis_R2_new.norm(dim=1, keepdim=True)  # normalize (n, 3)
    
    # Calculate new y-axis direction using the right-hand rule
    y_axis_R2_new = torch.cross(z_axis_R2_new, x_axis_R2_new, dim=1)  # (n, 3)
    y_axis_R2_new = y_axis_R2_new / y_axis_R2_new.norm(dim=1, keepdim=True)  # normalize
    
    # Combine the new rotation matrix
    R2_new = torch.stack((x_axis_R2_new, y_axis_R2_new, z_axis_R2_new), dim=-1)  # (n, 3, 3)
    
    return R2_new

def transform_end_effector_coordinate(tensor: torch.Tensor) -> torch.Tensor:
    """
    Transform end-effector coordinates from AnyGrasp to Isaac Lab coordinate system.
    
    Args:
        tensor (torch.Tensor): Input transformation matrix (batch_size x 3 x 3)
        
    Returns:
        torch.Tensor: Transformed coordinate matrix (batch_size x 3 x 3)
    
    The transformation:
    1. Extracts the 2x2 submatrix from positions (0:2, 1:3)
    2. Applies a fixed transformation [[0, -1], [1, 0]]
    3. Reconstructs the full 3x3 matrix with identity elements
    """
    # Fixed transformation matrix for coordinate conversion
    transform_matrix = torch.tensor([[0, -1], [1, 0]], dtype=torch.float32, device=tensor.device)

    # Extract 2x2 submatrix
    sub_matrices = tensor[:, :2, 1:3]  # Shape: (batch_size x 2 x 2)

    # Apply transformation
    transformed_sub_matrices = torch.matmul(transform_matrix, sub_matrices)

    # Reconstruct full 3x3 matrix
    output_tensor = torch.zeros_like(tensor)
    output_tensor[:, :2, :2] = transformed_sub_matrices  # Fill upper-left 2x2
    output_tensor[:, 2, 2] = 1.0  # Set bottom-right element to 1

    return output_tensor

# initialize warp
wp.init()

# State and Control Classes
class TakeFoto:
    """Camera trigger states."""
    ON = wp.constant(1.0)   # Trigger camera capture
    OFF = wp.constant(-1.0)  # Camera idle

class GripperState:
    """End-effector gripper states."""
    OPEN = wp.constant(1.0)   # Open gripper for approach
    CLOSE = wp.constant(-1.0)  # Close gripper for grasping

class RosMsgReceived:
    """ROS message reception states."""
    TRUE = wp.constant(1.0)   # Message received
    FALSE = wp.constant(-1.0)  # Waiting for message

class RobotStates:
    """
    Robot operational states for the manipulation task.
    Each state represents a specific phase of the pick-and-place operation.
    """
    # State definitions
    REST = wp.constant(0)                  # Initial resting position
    CAMERA = wp.constant(1)                # Image capture position
    WAIT_FOR_ROS_MESSAGE = wp.constant(2)  # Waiting for grasp pose
    DEFAULT_POSE = wp.constant(3)          # Pre-grasp position
    APPROACH_ABOVE_OBJECT = wp.constant(4)  # Position above target
    APPROACH_OBJECT = wp.constant(5)        # Moving to grasp position
    GRASP_OBJECT = wp.constant(6)          # Executing grasp
    LIFT_OBJECT = wp.constant(7)           # Lifting grasped object

    # Detailed state descriptions for logging and debugging
    STATE_DESCRIPTIONS = {
        REST: "Robot at rest position",
        CAMERA: "Capturing RGBD image",
        WAIT_FOR_ROS_MESSAGE: "Waiting for grasp pose from AnyGrasp",
        DEFAULT_POSE: "Moving to default position",
        APPROACH_ABOVE_OBJECT: "Moving above target object",
        APPROACH_OBJECT: "Moving to grasp position",
        GRASP_OBJECT: "Executing grasp",
        LIFT_OBJECT: "Lifting grasped object"
    }

class PickSmWaitTime:
    """
    Wait times (in seconds) for each state transition.
    These delays ensure stable execution of each motion phase.
    """
    REST = wp.constant(0.4)                 # Short rest at initial position
    CAMERA = wp.constant(1.6)               # Time for image capture
    DEFAULT_POSE = wp.constant(1.2)         # Time to reach default pose
    APPROACH_ABOVE_OBJECT = wp.constant(1.2) # Time to position above object
    APPROACH_OBJECT = wp.constant(1.2)       # Time to approach object
    GRASP_OBJECT = wp.constant(0.8)         # Time for grasp execution
    LIFT_OBJECT = wp.constant(1.2)          # Time for lifting motion

class SimulationConfig:
    """
    Centralized configuration management for the simulation environment.
    Handles all parameter settings and limits for the robot operation.
    """
    def __init__(self, args):
        # Environment settings
        self.num_envs = args.num_envs
        self.use_fabric = not args.disable_fabric
        
        # Camera configuration
        self.camera_params = {
            'width': 1280,          # Image width in pixels
            'height': 720,          # Image height in pixels
            'focal_length': 24,     # Focal length in mm
            'horiz_aperture': 20.955  # Horizontal aperture in mm
        }
        
        # Robot workspace limits (in meters)
        self.workspace_limits = {
            'x': (-0.5, 0.5),  # Forward/backward limits
            'y': (-0.5, 0.5),  # Left/right limits
            'z': (0.5, 0.8)    # Up/down limits
        }
        
        # Robot motion parameters
        self.motion_params = {
            'max_velocity': 0.5,     # Maximum end-effector velocity (m/s)
            'max_acceleration': 1.0,  # Maximum acceleration (m/s²)
            'gripper_force': 10.0    # Gripper closing force (N)
        }
        
        # Logging configuration
        self.logging = {
            'level': logging.INFO,
            'format': '%(asctime)s - %(levelname)s - %(message)s',
            'file': 'simulation.log'
        }

@wp.kernel
def infer_state_machine(
    dt: wp.array(dtype=float),
    sm_state: wp.array(dtype=int),
    sm_wait_time: wp.array(dtype=float),
    ee_pose: wp.array(dtype=wp.transform),
    camera_pose: wp.array(dtype=wp.transform),
    default_pose: wp.array(dtype=wp.transform),
    grasp_ready_pose:wp.array(dtype=wp.transform),
    object_pose: wp.array(dtype=wp.transform),
    des_object_pose: wp.array(dtype=wp.transform),
    des_ee_pose: wp.array(dtype=wp.transform),
    gripper_state: wp.array(dtype=float),
    take_foto: wp.array(dtype=float),
    ros_msg_received: wp.array(dtype=float),
    offset: wp.array(dtype=wp.transform),
):
    # retrieve thread id
    tid = wp.tid()
    # retrieve state machine state
    state = sm_state[tid]
    # decide next state
    if state == RobotStates.REST:
        des_ee_pose[tid] = ee_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        take_foto[tid] = TakeFoto.OFF
        # wait for a while
        if sm_wait_time[tid] >= PickSmWaitTime.REST:
            # move to next state and reset wait time
            sm_state[tid] = RobotStates.CAMERA
            sm_wait_time[tid] = 0.0
    elif state == RobotStates.CAMERA:
        des_ee_pose[tid] = camera_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        take_foto[tid] = TakeFoto.OFF
        # wait for a while
        if sm_wait_time[tid] >= PickSmWaitTime.CAMERA:
            take_foto[tid] = TakeFoto.ON
            # move to next state and reset wait time
            sm_state[tid] = RobotStates.WAIT_FOR_ROS_MESSAGE
            sm_wait_time[tid] = 0.0
    elif state == RobotStates.WAIT_FOR_ROS_MESSAGE:
        des_ee_pose[tid] = camera_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        take_foto[tid] = TakeFoto.OFF
        # wait till ros msg is received
        if ros_msg_received[tid] == RosMsgReceived.TRUE:
            sm_state[tid] = RobotStates.DEFAULT_POSE
            sm_wait_time[tid] = 0.0
    elif state == RobotStates.DEFAULT_POSE:
        des_ee_pose[tid] = default_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        take_foto[tid] = TakeFoto.OFF
        # TODO: error between current and desired ee pose below threshold
        # wait for a while
        if sm_wait_time[tid] >= PickSmWaitTime.DEFAULT_POSE:
            # move to next state and reset wait time
            sm_state[tid] = RobotStates.APPROACH_ABOVE_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == RobotStates.APPROACH_ABOVE_OBJECT:
        des_ee_pose[tid] = grasp_ready_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        take_foto[tid] = TakeFoto.OFF
        # TODO: error between current and desired ee pose below threshold
        # wait for a while
        if sm_wait_time[tid] >= PickSmWaitTime.APPROACH_OBJECT:
            # move to next state and reset wait time
            sm_state[tid] = RobotStates.APPROACH_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == RobotStates.APPROACH_OBJECT:
        des_ee_pose[tid] = object_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        take_foto[tid] = TakeFoto.OFF
        # TODO: error between current and desired ee pose below threshold
        # wait for a while
        if sm_wait_time[tid] >= PickSmWaitTime.APPROACH_OBJECT:
            # move to next state and reset wait time
            sm_state[tid] = RobotStates.GRASP_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == RobotStates.GRASP_OBJECT:
        des_ee_pose[tid] = object_pose[tid]
        gripper_state[tid] = GripperState.CLOSE
        take_foto[tid] = TakeFoto.OFF
        # wait for a while
        if sm_wait_time[tid] >= PickSmWaitTime.GRASP_OBJECT:
            # move to next state and reset wait time
            sm_state[tid] = RobotStates.LIFT_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == RobotStates.LIFT_OBJECT:
        des_ee_pose[tid] = des_object_pose[tid]
        gripper_state[tid] = GripperState.CLOSE
        take_foto[tid] = TakeFoto.OFF
        # TODO: error between current and desired ee pose below threshold
        # wait for a while
        if sm_wait_time[tid] >= PickSmWaitTime.LIFT_OBJECT:
            # move to next state and reset wait time
            sm_state[tid] = RobotStates.LIFT_OBJECT
            sm_wait_time[tid] = 0.0
    # increment wait time
    sm_wait_time[tid] = sm_wait_time[tid] + dt[tid]


class PickAndLiftSm:
    """A simple state machine in a robot's task space to pick and lift an object.

    The state machine is implemented as a warp kernel. It takes in the current state of
    the robot's end-effector and the object, and outputs the desired state of the robot's
    end-effector and the gripper. The state machine is implemented as a finite state
    machine with the following states:

    1. REST: The robot is at rest.
    2. APPROACH_ABOVE_OBJECT: The robot moves above the object.
    3. APPROACH_OBJECT: The robot moves to the object.
    4. GRASP_OBJECT: The robot grasps the object.
    5. LIFT_OBJECT: The robot lifts the object to the desired pose. This is the final state.
    """

    def __init__(self, dt: float, num_envs: int, device: torch.device | str = "cpu"):
        """Initialize the state machine.

        Args:
            dt: The environment time step.
            num_envs: The number of environments to simulate.
            device: The device to run the state machine on.
        """
        # save parameters
        self.dt = float(dt)
        self.num_envs = num_envs
        self.device = device
        # initialize state machine
        self.sm_dt = torch.full((self.num_envs,), self.dt, device=self.device)
        self.sm_state = torch.full((self.num_envs,), 0, dtype=torch.int32, device=self.device)
        self.sm_wait_time = torch.zeros((self.num_envs,), device=self.device)
        self.default_pose = torch.tensor([0.3, 0.0, 0.3, 0.0, 1.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        self.default_pose = self.default_pose[:, [0, 1, 2, 4, 5, 6, 3]]

        # camera position
        self.camera_pose = torch.tensor([0.25, 0.0, 0.8, 0.0, 1.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        self.camera_pose = self.camera_pose[:, [0, 1, 2, 4, 5, 6, 3]]

        # desired state
        self.des_ee_pose = torch.zeros((self.num_envs, 7), device=self.device)
        self.des_gripper_state = torch.full((self.num_envs,), 0.0, device=self.device)

        # camera state
        self.take_foto = torch.full((self.num_envs,), 0.0, device=self.device)

        # approach above object offset
        self.offset = torch.zeros((self.num_envs, 7), device=self.device)
        self.offset[:, 2] = 0.1
        self.offset[:, -1] = 1.0  # warp expects quaternion as (x, y, z, w)

        # convert to warp
        self.sm_dt_wp = wp.from_torch(self.sm_dt, wp.float32)
        self.sm_state_wp = wp.from_torch(self.sm_state, wp.int32)
        self.sm_wait_time_wp = wp.from_torch(self.sm_wait_time, wp.float32)
        self.camera_pose_wp = wp.from_torch(self.camera_pose.contiguous(), wp.transform)
        self.default_pose_wp = wp.from_torch(self.default_pose.contiguous(), wp.transform)
        self.des_ee_pose_wp = wp.from_torch(self.des_ee_pose, wp.transform)
        self.des_gripper_state_wp = wp.from_torch(self.des_gripper_state, wp.float32)
        self.take_foto_wp = wp.from_torch(self.take_foto, wp.float32)
        self.offset_wp = wp.from_torch(self.offset, wp.transform)

        # simulation environment data
        self.camera_translation_world = torch.tensor([0.25, 0.0, 0.8-0.015], device=self.device).repeat(self.num_envs,1)
        self.camera_rotation_world = torch.tensor([
            [1.0000e+00,         0.0,         0.0],
            [       0.0, -1.0000e+00,         0.0],
            [       0.0,         0.0, -1.0000e+00]
        ], device=self.device).repeat(self.num_envs, 1, 1)
        self.model_coordinate_rotation_to_issaclab = torch.tensor([
            [ 0.0,  0.0, 1.0],
            [ 0.0,  1.0, 0.0],
            [-1.0,  0.0, 0.0]
        ], device=self.device).repeat(self.num_envs, 1, 1)
        self.robot_root_translation_world = torch.tensor([0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        self.robot_root_orientation_world = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        self.ee_T_anygrasp_isaaclab = torch.tensor([
            [ 0.0,  0.0, -1.0],
            [ 0.0,  1.0,  0.0],
            [ 1.0,  0.0,  0.0]
        ], device=self.device).repeat(self.num_envs, 1, 1)

    def reset_idx(self, env_ids: Sequence[int] = None):
        """Reset the state machine."""
        if env_ids is None:
            env_ids = slice(None)
        self.sm_state[env_ids] = 0
        self.sm_wait_time[env_ids] = 0.0

    def compute(self, ee_pose: torch.Tensor, object_pose: torch.Tensor, grasp_ready_pose: torch.Tensor, des_object_pose: torch.Tensor, ros_msg_received: torch.Tensor):
        """Compute the desired state of the robot's end-effector and the gripper."""
        # convert all transformations from (w, x, y, z) to (x, y, z, w)
        ee_pose = ee_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        object_pose = object_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        grasp_ready_pose = grasp_ready_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        des_object_pose = des_object_pose[:, [0, 1, 2, 4, 5, 6, 3]]

        # convert to warp
        ee_pose_wp = wp.from_torch(ee_pose.contiguous(), wp.transform)
        object_pose_wp = wp.from_torch(object_pose.contiguous(), wp.transform)
        grasp_ready_pose = wp.from_torch(grasp_ready_pose.contiguous(), wp.transform)
        des_object_pose_wp = wp.from_torch(des_object_pose.contiguous(), wp.transform)
        ros_msg_received_wp = wp.from_torch(ros_msg_received, wp.float32)

        # run state machine
        wp.launch(
            kernel=infer_state_machine,
            dim=self.num_envs,
            inputs=[
                self.sm_dt_wp,
                self.sm_state_wp,
                self.sm_wait_time_wp,
                ee_pose_wp,
                self.camera_pose_wp,
                self.default_pose_wp,
                grasp_ready_pose,
                object_pose_wp,
                des_object_pose_wp,
                self.des_ee_pose_wp,
                self.des_gripper_state_wp,
                self.take_foto_wp,
                ros_msg_received_wp,
                self.offset_wp,
            ],
            device=self.device,
        )
        # print(f"print wp state:{self.sm_state_wp}")

        # convert transformations back to (w, x, y, z)
        des_ee_pose = self.des_ee_pose[:, [0, 1, 2, 6, 3, 4, 5]]
        # convert to torch
        return torch.cat([des_ee_pose, self.des_gripper_state.unsqueeze(-1)], dim=-1)

    @typechecked
    def get_grasping_pos(self, translations, rotations) -> Tuple[TensorType["num_envs", 7], TensorType["num_envs", 7]]:
        """
        Compute the grasping position and approach position for the robot end-effector.
        
        This function performs a series of coordinate transformations to convert
        grasp poses from camera frame to robot root frame.
        
        Args:
            translations: Grasp positions from AnyGrasp in camera frame
            rotations: Grasp orientations from AnyGrasp in camera frame
        
        Returns:
            Tuple containing:
            - grasp_goal: Target grasp pose in robot root frame (num_envs, 7)
            - grasp_source: Approach pose in robot root frame (num_envs, 7)
            where 7 = [x, y, z, qw, qx, qy, qz]
        """
        # 1. Setup initial poses
        # Get camera pose in world frame
        camera_translation_world = self.camera_translation_world
        camera_rotation_world = self.camera_rotation_world

        # 2. Process end-effector position in camera frame
        grasp_translation_camera = translations
        # Offset from end-effector to hand center
        hand_offset = torch.tensor([0.0, 0.0, 0.1034], device=self.device).repeat(self.num_envs, 1)
        # Apply offset to get end-effector position
        ee_translation_camera = grasp_translation_camera - hand_offset

        # 3. Process end-effector orientation
        # Adjust rotation matrices to satisfy constraints
        grasp_rotation_camera = adjust_matrices_torch(rotations)
        # Transform to Isaac Lab coordinate system
        grasp_rotation_isaaclab = transform_end_effector_coordinate(grasp_rotation_camera)
        # Apply coordinate system transformation
        grasp_rotation_camera_isaaclab = torch.matmul(
            self.model_coordinate_rotation_to_issaclab, 
            grasp_rotation_camera
        )

        # 4. Transform to world frame
        # Convert position to world frame
        grasp_translation_world = (
            camera_translation_world +
            torch.matmul(camera_rotation_world, ee_translation_camera.unsqueeze(-1)).squeeze(-1)
        )
        # Convert orientation to world frame
        grasp_rotation_world = torch.matmul(camera_rotation_world, grasp_rotation_isaaclab)

        # 5. Transform to robot root frame
        # Get robot root pose
        robot_root_translation_world = self.robot_root_translation_world
        robot_root_rotation_world = matrix_from_quat(self.robot_root_orientation_world)
        
        # Compute inverse transformation
        robot_root_rotation_inverse = robot_root_rotation_world.transpose(-2, -1)
        robot_root_translation_inverse = -torch.matmul(
            robot_root_rotation_inverse, 
            robot_root_translation_world.unsqueeze(-1)
        ).squeeze(-1)

        # Apply transformation to position
        grasp_translation_robot_root = (
            torch.matmul(robot_root_rotation_inverse, grasp_translation_world.unsqueeze(-1)).squeeze(-1) +
            robot_root_translation_inverse
        )
        
        # Apply transformation to orientation
        grasp_rotation_robot_root = torch.matmul(robot_root_rotation_inverse, grasp_rotation_world)
        grasp_orientation_robot_root = quat_from_matrix(grasp_rotation_robot_root)

        # 6. Calculate approach position
        # Compute approach direction (5cm back from grasp point)
        approach_direction = torch.matmul(
            grasp_rotation_robot_root, 
            torch.tensor([0.0, 0.0, 1.0], device=self.device).unsqueeze(-1)
        ).squeeze(-1)
        approach_position = grasp_translation_robot_root - 0.05 * approach_direction

        # 7. Combine positions and orientations
        grasp_goal = torch.cat((grasp_translation_robot_root, grasp_orientation_robot_root), dim=1)
        grasp_source = torch.cat((approach_position, grasp_orientation_robot_root), dim=1)

        return grasp_goal, grasp_source


def save_evaluation_results(output_dir, all_anygrasp_results, grasped_object_results, reach_desired_pose_results):
    """Save all evaluation metrics to JSON files."""
    results = {
        "anygrasp_success": all_anygrasp_results,
        "grasp_stability": grasped_object_results,
        "pose_accuracy": reach_desired_pose_results
    }
    
    for metric_name, data in results.items():
        output_file = os.path.join(output_dir, f"{metric_name}.json")
        with open(output_file, "w") as f:
            json.dump(data, f, indent=4)

def main():
    rclpy.init()
    isaaclab_node = IsaacLab()
    ros_thread = threading.Thread(target=ros2_node_thread, args=(isaaclab_node,))
    ros_thread.start()

    # parse configuration
    env_cfg = parse_env_cfg(
        "My-Custom-Env-v1",
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    # create environment
    env = gym.make("My-Custom-Env-v1", cfg=env_cfg)
    # reset environment at start
    env.reset()

    # create action buffers (position + quaternion)
    actions = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
    actions[:, 3] = 1.0
    
    # target object frame orientation 
    desired_orientation = torch.zeros((env.unwrapped.num_envs, 4), device=env.unwrapped.device)
    desired_orientation[:, 1] = 1.0
    
    # create state machine
    pick_sm = PickAndLiftSm(env_cfg.sim.dt * env_cfg.decimation, env.unwrapped.num_envs, env.unwrapped.device)

    # create output directory for all environment in simulation
    output_dir_base = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output")

    # Initialize the paths dictionary of all environment
    all_writer_saved_paths = {}

    # Create subfolders and writers for each environment
    rep_writers = []
    for env_index in range(env.unwrapped.num_envs):
        # Create a subfolder for each environment
        env_output_dir = os.path.join(output_dir_base, f"env_{env_index + 1}_camera")
        os.makedirs(env_output_dir, exist_ok=True)

        # Create a writer for this environment
        rep_writer = rep.BasicWriter(
            output_dir=env_output_dir,
            frame_padding=0,
        )
        rep_writers.append(rep_writer)


    # Defualt Grasp Result
    translation_env_nums = torch.tensor([0.0, 0.0, 0.0], device=env.unwrapped.device).repeat(env.unwrapped.num_envs,1)
    rotation_env_nums = torch.tensor([
        [       0.0,         0.0, 1.0000e+00],
        [       0.0, -1.0000e+00,        0.0],
        [1.0000e+00,         0.0,        0.0]
    ], device=env.unwrapped.device).repeat(env.unwrapped.num_envs, 1, 1)

    cycle_count = 0

    # Initialize dictionaries to store evaluation results
    all_anygrasp_results = {}      # Track AnyGrasp success/failure of generating grasping pose
    grasped_object_results = {}    # Track object grasping stability
    reach_desired_pose_results = {} # Track final pose accuracy

    # Initialize tracking variables for all environment
    previous_distances = torch.zeros((env.unwrapped.num_envs, 3), device=env.unwrapped.device)
    persistent_distance_changes = torch.zeros(env.unwrapped.num_envs, dtype=torch.bool, device=env.unwrapped.device)
    reach_transferring_pose = torch.zeros(env.unwrapped.num_envs, dtype=torch.bool, device=env.unwrapped.device)

    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # step environment
            dones = env.step(actions)[-2]

            # observations
            camera = env.unwrapped.scene["camera"]
            # -- end-effector frame
            ee_frame_sensor = env.unwrapped.scene["ee_frame"]
            tcp_rest_position = ee_frame_sensor.data.target_pos_w[..., 0, :].clone() - env.unwrapped.scene.env_origins
            tcp_rest_orientation = ee_frame_sensor.data.target_quat_w[..., 0, :].clone()
            # print(f"print ee frame position:{tcp_rest_position}")

            # print(f"print current ee_frame:{torch.cat([tcp_rest_position, tcp_rest_orientation], dim=-1)}")
            # -- object frame
            object_data: RigidObjectData = env.unwrapped.scene["object"].data
            object_position = object_data.root_pos_w - env.unwrapped.scene.env_origins
            # print(f"print current object position:{object_position}")
            # -- target object frame position
            desired_position = env.unwrapped.command_manager.get_command("object_pose")[..., :3]
            # print(f"print desired_transferring position:{desired_position}")
            # print(f"print current desired pos:{torch.cat([desired_position, desired_orientation], dim=-1)}")

            current_distances = object_position - tcp_rest_position
            print(f"print current distance between ee and object:{current_distances}")
            
            distances_ee_dtp = desired_position - tcp_rest_position
            print(f"print current distance betweem ee and desired transferring position:{distances_ee_dtp}")
            
            # Monitor object stability and pose accuracy
            for env_idx, sm_state in enumerate(pick_sm.sm_state):
                if sm_state == RobotStates.LIFT_OBJECT:  # State 7: Lifting phase
                    # Check for object stability (no sudden movements)
                    distance_change = torch.abs(current_distances[env_idx] - previous_distances[env_idx])
                    if (distance_change > 0.05).any():
                        persistent_distance_changes[env_idx] = True  # Object is unstable/falling
                    
                    abs_distances_ee_dtp = torch.abs(distances_ee_dtp)
                    if (abs_distances_ee_dtp < 0.01).all():
                        reach_transferring_pose[env_idx] = True
            
            # Update previous distances for next iteration
            previous_distances = current_distances.clone()

            if isaaclab_node.translation_env_nums == None or isaaclab_node.rotation_env_nums == None:
                ros_msg_received = torch.full((env.unwrapped.num_envs,), 0.0, device=env.unwrapped.device)
            else:
                ros_msg_received = torch.full((env.unwrapped.num_envs,), 1.0, device=env.unwrapped.device)
                translation_env_nums = isaaclab_node.translation_env_nums.to(env.unwrapped.device)  # (num_envs, 3)
                rotation_env_nums = isaaclab_node.rotation_env_nums.to(env.unwrapped.device)  # (num_envs, 3, 3)
                    
                # Check for rows in translation_env_nums that are [0.0, 0.0, 0.0]
                zero_translation_mask = torch.all(translation_env_nums == 0.0, dim=1)  # Shape: (num_envs,)

                # Check for matrices in rotation_env_nums that are all zeros
                zero_rotation_mask = torch.all(rotation_env_nums == 0.0, dim=(1, 2))  # Shape: (num_envs,)

                # Replace zero matrices in rotation_env_nums with the specific matrix
                replacement_matrix = torch.tensor([
                [0.0, 0.0, 1.0],
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0]
                ], dtype=torch.float32, device=env.unwrapped.device)

                # Identify indices of zero rotation matrices
                zero_rotation_indices = torch.where(zero_rotation_mask)[0]
                for idx in zero_rotation_indices:
                    rotation_env_nums[idx] = replacement_matrix

                # Combine the masks to identify environments where translation or rotation is invalid
                invalid_mask = zero_translation_mask | zero_rotation_mask  # Logical OR, Shape: (num_envs,)

                # Set ros_msg_received to 0.0 for invalid environments
                ros_msg_received[invalid_mask] = 0.0

            print(f"print translation_env_nums data:{translation_env_nums}")
            print(f"print rotation_env_nums data:{rotation_env_nums}")
            print(f"print ros_msg_received:{ros_msg_received}")

            grasp_goal, grasp_default_source = pick_sm.get_grasping_pos(translation_env_nums,rotation_env_nums)
            actions = pick_sm.compute(
                torch.cat([tcp_rest_position, tcp_rest_orientation], dim=-1),
                grasp_goal,
                grasp_default_source,
                torch.cat([desired_position, desired_orientation], dim=-1),
                ros_msg_received = ros_msg_received,
            )
            
            # Find indices of environments in the CAMERA state
            camera_indices = torch.nonzero(pick_sm.take_foto == TakeFoto.ON, as_tuple=False)

            if len(camera_indices) > 0:
                print(f"print current camera state:{pick_sm.take_foto_wp}")
                print(f"Environments in CAMERA state: {camera_indices}")
                for env_index in camera_indices.flatten():
                    print(f"print env_index:{env_index}")
                    rep_writer = rep_writers[env_index]
                    # Access single camera data for the current environment
                    single_cam_data = convert_dict_to_backend(camera.data.output[env_index], backend="numpy")

                    # Extract additional camera information
                    single_cam_info = camera.data.info[env_index]

                    # Pack data into replicator format
                    rep_output = {"annotators": {}}
                    for key, data, info in zip(single_cam_data.keys(), single_cam_data.values(), single_cam_info.values()):
                        if info is not None:
                            rep_output["annotators"][key] = {"render_product": {"data": data, **info}}
                        else:
                            rep_output["annotators"][key] = {"render_product": {"data": data}}

                    # Save images
                    # Note: Provide on-time data for the replicator to save images
                    rep_output["trigger_outputs"] = {"on_time": camera.frame[env_index]}
                    saved_paths = write_with_path(rep_writer, rep_output,frame_padding=0)
                    # 將該 env_index 的存儲路徑保存到字典中
                    all_writer_saved_paths[env_index.item()] = saved_paths
            
            print(f"print current sm state:{pick_sm.sm_state_wp}")
            isaaclab_node.update_data(pick_sm.sm_state_wp, pick_sm.take_foto_wp, all_writer_saved_paths)
            
            # Record results when episode ends
            if dones.any():
                # Initialize results for current cycle
                cycle_results = {
                    'anygrasp': {},
                    'grasp_stability': {},
                    'pose_accuracy': {}
                }

                # Record results for each environment
                for env_idx, state in enumerate(pick_sm.sm_state_wp.tolist()):
                    key = f"{cycle_count}_{env_idx}"
                    
                    # AnyGrasp success evaluation
                    if state == RobotStates.LIFT_OBJECT:
                        cycle_results['anygrasp'][key] = 1  # Success
                    elif state == RobotStates.WAIT_FOR_ROS_MESSAGE:
                        cycle_results['anygrasp'][key] = 0  # Failed to get grasp pose
                    else:
                        cycle_results['anygrasp'][key] = -1  # Other states

                    # Object stability evaluation
                    cycle_results['grasp_stability'][key] = not persistent_distance_changes[env_idx].item()

                    # Final pose accuracy evaluation
                    cycle_results['pose_accuracy'][key] = reach_transferring_pose[env_idx].item()

                # Update global results
                all_anygrasp_results.update(cycle_results['anygrasp'])
                grasped_object_results.update(cycle_results['grasp_stability'])
                reach_desired_pose_results.update(cycle_results['pose_accuracy'])

                # Increment cycle counter
                cycle_count += 1

                # Reset tracking variables for next episode
                persistent_distance_changes = torch.zeros(env.unwrapped.num_envs, dtype=torch.bool, device=env.unwrapped.device)
                reach_transferring_pose = torch.zeros(env.unwrapped.num_envs, dtype=torch.bool, device=env.unwrapped.device)
                pick_sm.reset_idx(dones.nonzero(as_tuple=False).squeeze(-1))
                all_writer_saved_paths = {}

    # Call the save function at the end of main
    save_evaluation_results(output_dir_base, all_anygrasp_results, grasped_object_results, reach_desired_pose_results)

    env.close()
    isaaclab_node.destroy_node()
    rclpy.shutdown()
    ros_thread.join()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
