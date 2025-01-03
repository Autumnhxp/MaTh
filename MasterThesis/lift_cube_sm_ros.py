# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to run an environment with a pick and lift state machine.

The state machine is implemented in the kernel function `infer_state_machine`.
It uses the `warp` library to run the state machine in parallel on the GPU.

.. code-block:: bash

    ./isaaclab.sh -p source/standalone/environments/state_machine/lift_cube_sm.py --num_envs 32

"""

"""Launch Omniverse Toolkit first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Pick and lift state machine for lift environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to simulate.")
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
    from rclpy.qos import QoSProfile
    from std_msgs.msg import Header, Float32MultiArray, Int8MultiArray, String
    from sensor_msgs.msg import PointCloud2, PointField
    import threading
    import json
    import time
    print("import rclpy success!")
except:
    print("import rclpy failed")

# ROS 2 Publisher Node
class IsaacLab(Node):
    def __init__(self):
        super().__init__('IsaacLab')
        # Publisher
        self.state_publisher_ = self.create_publisher(Int8MultiArray, 'sm_state_wp', 10)
        self.foto_publisher_ = self.create_publisher(Float32MultiArray, 'take_foto_wp', 10)
        self.file_paths_publicher_ = self.create_publisher(String, 'file_paths', 10)
        # Subscriber
        self.grasp_results_subscription = self.create_subscription(
            String,
            'grasp_results',
            self.listener_callback,
            10
        )
        self.timer = self.create_timer(0.5, self.timer_callback)  # 每 0.5 秒發佈一次
        # 用於儲存從主程序更新的資料
        self.sm_state_wp_data = None
        self.take_foto_wp_data = None
        self.file_paths_data = None
        self.grasp_results_data = None
        self.translation_env_nums = None
        self.rotation_env_nums = None

    def update_data(self, sm_state, take_foto, file_paths):
        """更新需要發佈的資料"""
        self.sm_state_wp_data = sm_state
        self.take_foto_wp_data = take_foto
        self.file_paths_data = file_paths

    def timer_callback(self):
        if self.sm_state_wp_data is not None:
            msg_sm_state = Int8MultiArray()
            msg_sm_state.data = self.sm_state_wp_data.numpy().tolist()  # 轉換為列表發佈
            self.state_publisher_.publish(msg_sm_state)
            # self.get_logger().info(f'Published sm_state_wp: {msg_sm_state.data}')
        if self.take_foto_wp_data is not None:
            msg_take_foto = Float32MultiArray()
            msg_take_foto.data = self.take_foto_wp_data.numpy().tolist()  # 轉換為列表發佈
            self.foto_publisher_.publish(msg_take_foto)
            # self.get_logger().info(f'Published take_foto_wp: {msg_take_foto.data}')
        if self.file_paths_data is not None:
            try:
                file_paths_json = json.dumps(self.file_paths_data)
                msg_file_paths = String()
                msg_file_paths.data = file_paths_json
                self.file_paths_publicher_.publish(msg_file_paths)
                # self.get_logger().info(f"Published file paths:\n{file_paths_json}")
            except Exception as e:
                self.get_logger().error(f"Failed to publish file paths: {e}")
    
    def listener_callback(self, msg): 
        try:
            data = json.loads(msg.data)
            self.grasp_results_data = data.get('grasp_results', {})
            # self.get_logger().info(f'Received data from Python 3.8: {data}')
            # self.get_logger().info(f'Data of grasp_results : {self.grasp_results_data}')

            # 提取抓取结果并转换为 PyTorch 张量
            grasp_results = data.get('grasp_results', {})
            num_envs = len(grasp_results)  # 根据抓取结果确定环境数量

            # 创建 translation 和 rotation 的张量列表
            translations = []
            rotations = []

            for env_id in range(num_envs):
                env_id_str = str(env_id)
                env_result = grasp_results.get(env_id_str, {})
                translation = env_result.get('translation')
                rotation = env_result.get('rotation')
            
                translations.append(torch.tensor(translation, dtype=torch.float32))
                rotations.append(torch.tensor(rotation, dtype=torch.float32))

            # self.get_logger().info(f"translation: {translations}")
            # self.get_logger().info(f"rotation: {rotations}")

            # 将 translation 和 rotation 转换为张量矩阵
            if translations != [] and rotations != []: 
                translation_env_nums = torch.stack(translations)  # (num_envs, 3)
                rotation_env_nums = torch.stack(rotations)  # (num_envs, 3, 3)

                # 打印结果（调试用）
                # self.get_logger().info(f"translation_env_nums: {translation_env_nums}")
                # self.get_logger().info(f"rotation_env_nums: {rotation_env_nums}")

                # 将结果保存为节点属性，供主程序调用
                self.translation_env_nums = translation_env_nums
                self.rotation_env_nums = rotation_env_nums
            else:
                self.translation_env_nums = None
                self.rotation_env_nums = None
        
        except json.JSONDecodeError as e:
            self.get_logger().error(f"Invalid JSON format: {e}")
            return
        except Exception as e:
            self.get_logger().error(f'Error in listener_callback: {e}')
        

def adjust_matrices_torch(R2) -> torch.Tensor:
    """
    调整 R2 张量，使其满足以下条件：
    1. 每个调整后的矩阵的 x 轴与 R1 的 x 轴平行。
    2. 每个矩阵的 z 轴投影到 R1 的 z-y 平面的比例保持不变。
    
    参数:
        R2 (torch.Tensor): 一个形状为 (n, 3, 3) 的张量，其中包含 n 个旋转矩阵。
    
    返回:
        torch.Tensor: 调整后的形状为 (n, 3, 3) 的旋转矩阵张量。
    """
    # 定义 R1 的基准旋转矩阵
    R1 = torch.tensor([[0.0, 0.0, 1.0],
                       [0.0, -1.0, 0.0],
                       [1.0, 0.0, 0.0]], dtype=R2.dtype, device=R2.device)
    
    # 提取 R1 的轴向量
    x_axis_R1 = R1[:, 0]
    y_axis_R1 = R1[:, 1]
    z_axis_R1 = R1[:, 2]
    
    # 确保 R2 的形状为 (n, 3, 3)
    assert R2.ndim == 3 and R2.shape[1:] == (3, 3), "R2 的形状必须为 (n, 3, 3)"
    
    # 调整 R2 的 x 轴为与 R1 的 x 轴平行
    x_axis_R2_new = x_axis_R1.repeat(R2.shape[0], 1)  # (n, 3)
    
    # 获取 R2 的原始 z 轴向量
    z_axis_R2 = R2[:, :, 2]  # (n, 3)
    
    # 投影 R2 的 z 轴到 R1 的 z-y 平面
    z_proj = torch.einsum('ij,j->i', z_axis_R2, z_axis_R1)[:, None] * z_axis_R1  # (n, 3)
    y_proj = torch.einsum('ij,j->i', z_axis_R2, y_axis_R1)[:, None] * y_axis_R1  # (n, 3)
    
    # 重新计算调整后的 z 轴，保持比例不变
    z_axis_R2_new = z_proj + y_proj
    z_axis_R2_new = z_axis_R2_new / z_axis_R2_new.norm(dim=1, keepdim=True)  # 归一化 (n, 3)
    
    # 通过右手法则计算新的 y 轴方向
    y_axis_R2_new = torch.cross(z_axis_R2_new, x_axis_R2_new, dim=1)  # (n, 3)
    y_axis_R2_new = y_axis_R2_new / y_axis_R2_new.norm(dim=1, keepdim=True)  # 归一化
    
    # 组合新的旋转矩阵
    R2_new = torch.stack((x_axis_R2_new, y_axis_R2_new, z_axis_R2_new), dim=-1)  # (n, 3, 3)
    
    return R2_new

def change_ee_coordinate_to_isaac_lab(tensor)-> torch.Tensor:
    # 固定变换矩阵
    transform_matrix = torch.tensor([[0, -1], [1, 0]], dtype=torch.float32, device=tensor.device)

    # 提取 [[a, b], [c, d]] 的部分
    sub_matrices = tensor[:, :2, 1:3]  # (n x 2 x 2)

    # 批量矩阵乘法
    transformed_sub_matrices = torch.matmul(transform_matrix, sub_matrices)  # (n x 2 x 2)

    # 构造输出张量
    output_tensor = torch.zeros_like(tensor)
    output_tensor[:, :2, :2] = transformed_sub_matrices  # 填充左上角
    output_tensor[:, 2, 2] = 1.0  # 填充右下角

    return output_tensor

# ROS 2 初始化函數
def ros2_node_thread(publisher_node):
    rclpy.spin(publisher_node)

# initialize warp
wp.init()

class TakeFoto:
    ON = wp.constant(1.0)
    Off = wp.constant(-1.0)

class GripperState:
    """States for the gripper."""

    OPEN = wp.constant(1.0)
    CLOSE = wp.constant(-1.0)

class RosMsgReceived:
    TRUE = wp.constant(1.0)
    FALSE = wp.constant(-1.0)    

class PickSmState:
    """States for the pick state machine."""

    REST = wp.constant(0)
    CAMERA = wp.constant(1)
    WAIT_FOR_ROS_MESSAGE = wp.constant(2)
    DEFAULT_POSE = wp.constant(3)
    APPROACH_ABOVE_OBJECT = wp.constant(4)
    APPROACH_OBJECT = wp.constant(5)
    GRASP_OBJECT = wp.constant(6)
    LIFT_OBJECT = wp.constant(7)


class PickSmWaitTime:
    """Additional wait times (in s) for states for before switching."""

    REST = wp.constant(0.4)
    CAMERA = wp.constant(1.6)
    DEFAULT_POSE = wp.constant(2.0)
    APPROACH_ABOVE_OBJECT = wp.constant(1.2)
    APPROACH_OBJECT = wp.constant(2.0)
    GRASP_OBJECT = wp.constant(0.8)
    LIFT_OBJECT = wp.constant(1.2)


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
    if state == PickSmState.REST:
        des_ee_pose[tid] = ee_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        take_foto[tid] = TakeFoto.Off
        # wait for a while
        if sm_wait_time[tid] >= PickSmWaitTime.REST:
            # move to next state and reset wait time
            sm_state[tid] = PickSmState.CAMERA
            sm_wait_time[tid] = 0.0
    elif state == PickSmState.CAMERA:
        des_ee_pose[tid] = camera_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        take_foto[tid] = TakeFoto.Off
        # wait for a while
        if sm_wait_time[tid] >= PickSmWaitTime.CAMERA:
            take_foto[tid] = TakeFoto.ON
            # move to next state and reset wait time
            sm_state[tid] = PickSmState.WAIT_FOR_ROS_MESSAGE
            sm_wait_time[tid] = 0.0
    elif state == PickSmState.WAIT_FOR_ROS_MESSAGE:
        des_ee_pose[tid] = camera_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        take_foto[tid] = TakeFoto.Off
        # wait till ros msg is received
        if ros_msg_received[tid] == RosMsgReceived.TRUE:
            sm_state[tid] = PickSmState.DEFAULT_POSE
            sm_wait_time[tid] = 0.0
    elif state == PickSmState.DEFAULT_POSE:
        des_ee_pose[tid] = default_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        take_foto[tid] = TakeFoto.Off
        # TODO: error between current and desired ee pose below threshold
        # wait for a while
        if sm_wait_time[tid] >= PickSmWaitTime.DEFAULT_POSE:
            # move to next state and reset wait time
            sm_state[tid] = PickSmState.APPROACH_ABOVE_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == PickSmState.APPROACH_ABOVE_OBJECT:
        des_ee_pose[tid] = grasp_ready_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        take_foto[tid] = TakeFoto.Off
        # TODO: error between current and desired ee pose below threshold
        # wait for a while
        if sm_wait_time[tid] >= PickSmWaitTime.APPROACH_OBJECT:
            # move to next state and reset wait time
            sm_state[tid] = PickSmState.APPROACH_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == PickSmState.APPROACH_OBJECT:
        des_ee_pose[tid] = object_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        take_foto[tid] = TakeFoto.Off
        # TODO: error between current and desired ee pose below threshold
        # wait for a while
        if sm_wait_time[tid] >= PickSmWaitTime.APPROACH_OBJECT:
            # move to next state and reset wait time
            sm_state[tid] = PickSmState.GRASP_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == PickSmState.GRASP_OBJECT:
        des_ee_pose[tid] = object_pose[tid]
        gripper_state[tid] = GripperState.CLOSE
        take_foto[tid] = TakeFoto.Off
        # wait for a while
        if sm_wait_time[tid] >= PickSmWaitTime.GRASP_OBJECT:
            # move to next state and reset wait time
            sm_state[tid] = PickSmState.LIFT_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == PickSmState.LIFT_OBJECT:
        des_ee_pose[tid] = des_object_pose[tid]
        gripper_state[tid] = GripperState.CLOSE
        take_foto[tid] = TakeFoto.Off
        # TODO: error between current and desired ee pose below threshold
        # wait for a while
        if sm_wait_time[tid] >= PickSmWaitTime.LIFT_OBJECT:
            # move to next state and reset wait time
            sm_state[tid] = PickSmState.LIFT_OBJECT
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
        Compute the grasping position for simulation environment

        Args:
            

        Returns:
            torch.Tensor: A tensor of shape (num_envs, 7), where 7 is [3 translation + 4 quaternion].
        """
        # Define grasping pose for the arm
        # Camera pose in world frame
        camera_translation_world = self.camera_translation_world
        camera_rotation_world = self.camera_rotation_world

        # Grasping pose in camera frame
        grasp_translation_camera = translations
        # hand_traslation_from_EE = torch.tensor([0.0, 0.0, 0.1034], device=self.device)
        hand_traslation_from_EE = torch.tensor([0.0, 0.0, 0.1034-0.009], device=self.device).repeat(self.num_envs,1)
        EE_translation_camera = grasp_translation_camera - hand_traslation_from_EE
        grasp_rotation_camera = rotations
        grasp_rotation_camera = adjust_matrices_torch(rotations)
        #grasp_rotation_camera = torch.tensor([
        #    [ 0.0,  -1.0, 0.0],
        #    [ 0.0,  0.0, -1.0],
        #    [ 1.0,  0.0, 0.0]
        #], device=self.device).repeat(self.num_envs, 1, 1)
        print(f"print grasp rotation from camera in AnyGrasp:{grasp_rotation_camera}")
        # grasp_rotation_isaaclab = torch.matmul(torch.matmul(self.ee_T_anygrasp_isaaclab,grasp_rotation_camera), self.ee_T_anygrasp_isaaclab.transpose(-2, -1))
        grasp_rotation_isaaclab = change_ee_coordinate_to_isaac_lab(grasp_rotation_camera)
        #grasp_rotation_isaaclab = torch.tensor([
        #    [ 1.0, 0.0, 0.0],
        #    [ 0.0, 1.0, 0.0],
        #    [ 0.0, 0.0, 1.0]
        #], device =self.device).repeat(self.num_envs, 1, 1)
        print(f"print grasp rotation in isaac lab:{grasp_rotation_isaaclab}")

        AnyGrasp_grasp_coordinate_rotation_isaaclab = self.model_coordinate_rotation_to_issaclab
        grasp_rotation_camera_isaaclab = torch.matmul(AnyGrasp_grasp_coordinate_rotation_isaaclab, grasp_rotation_camera)
        print(f"print grasp rotation from camera in isaac lab camera coordinate:{grasp_rotation_camera_isaaclab}")


        # Transform grasping position to world frame
        grasp_translation_world = (
            camera_translation_world +
            torch.matmul(camera_rotation_world, EE_translation_camera.unsqueeze(-1)).squeeze(-1)
        )

        # Transform grasping orientation to world frame
        grasp_rotation_world = torch.matmul(camera_rotation_world, grasp_rotation_isaaclab)
        print(f"print grasp rotation from camera in isaac lab world coordinate:{grasp_rotation_world}")


        # Output results
        # print("Grasping position in world coordinates (translation):")
        # print(grasp_translation_world)

        # print("Grasping orientation in world coordinates (rotation matrix):")
        # print(grasp_rotation_world.cpu().numpy())

        # Robot root pose in the world frame
        robot_root_translation_world = self.robot_root_translation_world
        robot_root_orientation_world = self.robot_root_orientation_world  # Quaternion (w, x, y, z)
        robot_root_rotation_world = matrix_from_quat(robot_root_orientation_world)

        # Compute inverse transformation (world → robot root)
        robot_root_rotation_inverse = robot_root_rotation_world.transpose(-2, -1)  # Transpose of rotation matrix
        robot_root_translation_inverse = -torch.matmul(robot_root_rotation_inverse, robot_root_translation_world.unsqueeze(-1)).squeeze(-1)

        # Transform grasping pose to robot root frame
        grasp_translation_robot_root = (
            torch.matmul(robot_root_rotation_inverse, grasp_translation_world.unsqueeze(-1)).squeeze(-1) +
            robot_root_translation_inverse
        )
        
        
        grasp_rotation_robot_root = torch.matmul(robot_root_rotation_inverse, grasp_rotation_world)
        print(f"print grasp rotation from robot root:{grasp_rotation_robot_root}" ) 
        # grasp_rotation_robot_root = torch.matmul(grasp_rotation_robot_root,self.ee_rotation_to_table)
        grasp_orientation_robot_root = quat_from_matrix(grasp_rotation_robot_root)
        print(f"grasp orientation quat:{grasp_orientation_robot_root}")
        # grasp_orientation_robot_root = torch.tensor([0.0, 1.0, 0.0, 0.0], device=self.device).repeat(self.num_envs,1)
        # grasp_rotation_robot_root = matrix_from_quat(grasp_orientation_robot_root)


        # 从 grasp_orientation_robot_root 计算方向向量
        direction_vector = torch.matmul(grasp_rotation_robot_root, torch.tensor([0.0, 0.0, 1.0], device=self.device).unsqueeze(-1)).squeeze(-1)

        # 计算起点（source_point），即距离 grasp_translation_robot_root 为 0.05 且方向相反的点
        source_point = grasp_translation_robot_root - 0.05 * direction_vector
        # Output results
        # print("Grasping position in robot root coordinates (translation):")
        # print(grasp_translation_robot_root)

        # print("Grasping orientation in robot root coordinates (rotation matrix):")
        # print(grasp_rotation_robot_root.cpu().numpy())

        # print("Grasping orientation in robot root coordinates (quaternion):")
        # print(grasp_orientation_robot_root)

        # Define grasp translation and orientation as a single goal
        grasp_goal = torch.cat((grasp_translation_robot_root, grasp_orientation_robot_root), dim=1) # Concatenate translation and quaternion
        grasp_source = torch.cat((source_point, grasp_orientation_robot_root), dim=1) # Concatenate translation and quaternion
        return grasp_goal, grasp_source


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
    # desired object orientation (we only do position control of object)
    desired_orientation = torch.zeros((env.unwrapped.num_envs, 4), device=env.unwrapped.device)
    desired_orientation[:, 1] = 1.0
    # create state machine
    pick_sm = PickAndLiftSm(env_cfg.sim.dt * env_cfg.decimation, env.unwrapped.num_envs, env.unwrapped.device)

    # create replicator writers for each environment
    output_dir_base = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output")

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
        all_writer_saved_paths = {}

        # Defualt Grasp Result
        translation_env_nums = torch.tensor([0.0, 0.0, 0.0], device=env.unwrapped.device).repeat(env.unwrapped.num_envs,1)
        rotation_env_nums = torch.tensor([
            [       0.0,         0.0, 1.0000e+00],
            [       0.0, -1.0000e+00,        0.0],
            [1.0000e+00,         0.0,        0.0]
        ], device=env.unwrapped.device).repeat(env.unwrapped.num_envs, 1, 1)


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
            # print(f"print current ee_frame:{torch.cat([tcp_rest_position, tcp_rest_orientation], dim=-1)}")
            # -- object frame
            object_data: RigidObjectData = env.unwrapped.scene["object2"].data
            object_position = object_data.root_pos_w - env.unwrapped.scene.env_origins
            # print(f"print current object pos:{torch.cat([object_position, desired_orientation], dim=-1)}")
            # -- target object frame
            desired_position = env.unwrapped.command_manager.get_command("object_pose")[..., :3]
            # print(f"print current desired pos:{torch.cat([desired_position, desired_orientation], dim=-1)}")

            # print(f"print ros node translation_env_nums data:{isaaclab_node.translation_env_nums}")
            # print(f"print ros node rotation_env_nums data:{isaaclab_node.rotation_env_nums}")
            
            if isaaclab_node.translation_env_nums == None or isaaclab_node.rotation_env_nums == None:
                ros_msg_received = torch.full((env.unwrapped.num_envs,), 0.0, device=env.unwrapped.device)
            else:
                ros_msg_received = torch.full((env.unwrapped.num_envs,), 1.0, device=env.unwrapped.device)
                translation_env_nums = isaaclab_node.translation_env_nums.to(env.unwrapped.device)  # (num_envs, 3)
                rotation_env_nums = isaaclab_node.rotation_env_nums.to(env.unwrapped.device)  # (num_envs, 3, 3)

            print(f"print translation_env_nums data:{translation_env_nums}")
            print(f"print rotation_env_nums data:{rotation_env_nums}")

                # 使用 translation 和 rotation 张量
                # print(f"Translation Matrices:\n{translation_env_nums}")
                # print(f"Rotation Matrices:\n{rotation_env_nums}")

            # advance state machine
            # actions = pick_sm.compute(
            #    torch.cat([tcp_rest_position, tcp_rest_orientation], dim=-1),
            #    torch.cat([object_position, desired_orientation], dim=-1),
            #    torch.cat([desired_position, desired_orientation], dim=-1),
            # )
            # advance state machine anygrasp
            # grasp_goal = torch.tensor([0.25, 0.0, 0.8, 0.0, 1.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
            grasp_goal, grasp_default_source = pick_sm.get_grasping_pos(translation_env_nums,rotation_env_nums)
            actions = pick_sm.compute(
                torch.cat([tcp_rest_position, tcp_rest_orientation], dim=-1),
                grasp_goal,
                grasp_default_source,
                torch.cat([desired_position, desired_orientation], dim=-1),
                ros_msg_received=ros_msg_received,
            )
            
            


            # print(f"print current actions:{actions}")
            # print(f"print current camera data:{camera.data.output[0]}") #env 1
            # print(f"print current camera data:{camera.data.output[1]}") #env 2

            
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

            # 集中打印所有檔案存取路徑
            # print("\nAll saved file paths:")
            # for env_index, paths in all_saved_paths.items():
            #     print(f"Environment {env_index}:")
            #     for annotator, path in paths.items():
            #         if path:
            #             print(f"  {annotator}: {path}")
            # print(f"Print All File Paths:", all_saved_paths)
            
            print(f"print current sm state:{pick_sm.sm_state_wp}")
            isaaclab_node.update_data(pick_sm.sm_state_wp, pick_sm.take_foto_wp, all_writer_saved_paths)
            
            
            # else:
                # print("No environments are in the CAMERA state.")

            # print(f"print current camera state:{pick_sm.take_foto_wp}")

            # reset state machine
            if dones.any():
                pick_sm.reset_idx(dones.nonzero(as_tuple=False).squeeze(-1))
                all_writer_saved_paths = {}
                # Defualt Grasp Result
                translation_env_nums = torch.tensor([0.0, 0.0, 0.0], device=env.unwrapped.device).repeat(env.unwrapped.num_envs,1)
                rotation_env_nums = torch.tensor([
                    [       0.0,         0.0, 1.0000e+00],
                    [       0.0, -1.0000e+00,        0.0],
                    [1.0000e+00,         0.0,        0.0]
                ], device=env.unwrapped.device).repeat(env.unwrapped.num_envs, 1, 1)


    # close the environment
    env.close()
    isaaclab_node.destroy_node()
    rclpy.shutdown()
    ros_thread.join()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
