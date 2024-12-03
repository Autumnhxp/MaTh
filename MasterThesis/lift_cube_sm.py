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
import torch
from collections.abc import Sequence

import warp as wp

from omni.isaac.lab.assets.rigid_object.rigid_object_data import RigidObjectData

# env registration in gym
import omni.isaac.lab_tasks  # noqa: F401
import my_custom_env # for own use case

from omni.isaac.lab.utils.math import subtract_frame_transforms, matrix_from_quat, quat_from_matrix


from omni.isaac.lab_tasks.utils.parse_cfg import parse_env_cfg

# initialize warp
wp.init()

class CameraState:
    ON = wp.constant(1.0)
    Off = wp.constant(-1.0)

class GripperState:
    """States for the gripper."""

    OPEN = wp.constant(1.0)
    CLOSE = wp.constant(-1.0)


class PickSmState:
    """States for the pick state machine."""

    REST = wp.constant(0)
    CAMERA = wp.constant(1)
    APPROACH_ABOVE_OBJECT = wp.constant(2)
    APPROACH_OBJECT = wp.constant(3)
    GRASP_OBJECT = wp.constant(4)
    LIFT_OBJECT = wp.constant(5)


class PickSmWaitTime:
    """Additional wait times (in s) for states for before switching."""

    REST = wp.constant(0.5)
    CAMERA = wp.constant(3)
    APPROACH_ABOVE_OBJECT = wp.constant(1.5)
    APPROACH_OBJECT = wp.constant(1.6)
    GRASP_OBJECT = wp.constant(0.3)
    LIFT_OBJECT = wp.constant(1.0)


@wp.kernel
def infer_state_machine(
    dt: wp.array(dtype=float),
    sm_state: wp.array(dtype=int),
    sm_wait_time: wp.array(dtype=float),
    ee_pose: wp.array(dtype=wp.transform),
    camera_pose: wp.array(dtype=wp.transform),
    default_pose:wp.array(dtype=wp.transform),
    object_pose: wp.array(dtype=wp.transform),
    des_object_pose: wp.array(dtype=wp.transform),
    des_ee_pose: wp.array(dtype=wp.transform),
    gripper_state: wp.array(dtype=float),
    camera_state: wp.array(dtype=float),
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
        camera_state[tid] = CameraState.Off
        # wait for a while
        if sm_wait_time[tid] >= PickSmWaitTime.REST:
            # move to next state and reset wait time
            sm_state[tid] = PickSmState.CAMERA
            sm_wait_time[tid] = 0.0
    elif state == PickSmState.CAMERA:
        des_ee_pose[tid] = camera_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        camera_state[tid] = CameraState.ON
        # wait for a while
        if sm_wait_time[tid] >= PickSmWaitTime.CAMERA:
            # move to next state and reset wait time
            sm_state[tid] = PickSmState.APPROACH_ABOVE_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == PickSmState.APPROACH_ABOVE_OBJECT:
        des_ee_pose[tid] = default_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        camera_state[tid] = CameraState.Off
        # TODO: error between current and desired ee pose below threshold
        # wait for a while
        if sm_wait_time[tid] >= PickSmWaitTime.APPROACH_OBJECT:
            # move to next state and reset wait time
            sm_state[tid] = PickSmState.APPROACH_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == PickSmState.APPROACH_OBJECT:
        des_ee_pose[tid] = object_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        camera_state[tid] = CameraState.Off
        # TODO: error between current and desired ee pose below threshold
        # wait for a while
        if sm_wait_time[tid] >= PickSmWaitTime.APPROACH_OBJECT:
            # move to next state and reset wait time
            sm_state[tid] = PickSmState.GRASP_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == PickSmState.GRASP_OBJECT:
        des_ee_pose[tid] = object_pose[tid]
        gripper_state[tid] = GripperState.CLOSE
        camera_state[tid] = CameraState.Off
        # wait for a while
        if sm_wait_time[tid] >= PickSmWaitTime.GRASP_OBJECT:
            # move to next state and reset wait time
            sm_state[tid] = PickSmState.LIFT_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == PickSmState.LIFT_OBJECT:
        des_ee_pose[tid] = des_object_pose[tid]
        gripper_state[tid] = GripperState.CLOSE
        camera_state[tid] = CameraState.Off
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
        self.camera_state = torch.full((self.num_envs,), 0.0, device=self.device)

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
        self.camera_state_wp = wp.from_torch(self.camera_state, wp.float32)
        self.offset_wp = wp.from_torch(self.offset, wp.transform)

    def reset_idx(self, env_ids: Sequence[int] = None):
        """Reset the state machine."""
        if env_ids is None:
            env_ids = slice(None)
        self.sm_state[env_ids] = 0
        self.sm_wait_time[env_ids] = 0.0

    def compute(self, ee_pose: torch.Tensor, object_pose: torch.Tensor, des_object_pose: torch.Tensor):
        """Compute the desired state of the robot's end-effector and the gripper."""
        # convert all transformations from (w, x, y, z) to (x, y, z, w)
        ee_pose = ee_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        object_pose = object_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        des_object_pose = des_object_pose[:, [0, 1, 2, 4, 5, 6, 3]]

        # convert to warp
        ee_pose_wp = wp.from_torch(ee_pose.contiguous(), wp.transform)
        object_pose_wp = wp.from_torch(object_pose.contiguous(), wp.transform)
        des_object_pose_wp = wp.from_torch(des_object_pose.contiguous(), wp.transform)

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
                object_pose_wp,
                des_object_pose_wp,
                self.des_ee_pose_wp,
                self.des_gripper_state_wp,
                self.camera_state_wp,
                self.offset_wp,
            ],
            device=self.device,
        )
        print(f"print wp state:{self.sm_state_wp}")

        # convert transformations back to (w, x, y, z)
        des_ee_pose = self.des_ee_pose[:, [0, 1, 2, 6, 3, 4, 5]]
        # convert to torch
        return torch.cat([des_ee_pose, self.des_gripper_state.unsqueeze(-1)], dim=-1)

    def get_grasping_pos(self):
        # Define grasping pose for the arm
        # Camera pose in world frame
        camera_translation_world = torch.tensor([0.25002, -8.7417e-08, 0.7850], device=self.device)
        camera_rotation_world = torch.tensor([
            [1.0000e+00,  4.2478e-06,  7.1212e-05],
            [4.2478e-06, -1.0000e+00, -3.5747e-07],
            [7.1212e-05,  3.5778e-07, -1.0000e+00]
        ], device=self.device)

        # Grasping pose in camera frame
        grasp_translation_camera = torch.tensor([0.05017354, -0.00239286, 0.70051754], device=self.device)
        # hand_traslation_from_EE = torch.tensor([0.0, 0.0, 0.1034], device=self.device)
        hand_traslation_from_EE = torch.tensor([0.0, 0.0, 0.0], device=self.device)
        EE_translation_camera = grasp_translation_camera - hand_traslation_from_EE
        grasp_rotation_camera = torch.tensor([
            [-0.08158159,  0.0,  0.99666667],
            [ 0.0,        -1.0,  0.0       ],
            [ 0.99666667,  0.0,  0.08158159]
        ], device=self.device)
        AnyGrasp_grasp_coordinate_rotation_isaaclab = torch.tensor([
            [ 0.0,  0.0, 1.0],
            [ 0.0, -1.0, 0.0],
            [ 1.0,  0.0, 0.0]
        ], device=self.device)
        grasp_rotation_camera = torch.matmul(AnyGrasp_grasp_coordinate_rotation_isaaclab, grasp_rotation_camera)

        # Transform grasping position to world frame
        grasp_translation_world = (
            camera_translation_world +
            torch.matmul(camera_rotation_world, EE_translation_camera)
        )

        # Transform grasping orientation to world frame
        grasp_rotation_world = torch.matmul(camera_rotation_world, grasp_rotation_camera)

        # Output results
        # print("Grasping position in world coordinates (translation):")
        # print(grasp_translation_world)

        # print("Grasping orientation in world coordinates (rotation matrix):")
        # print(grasp_rotation_world.cpu().numpy())

        # Robot root pose in the world frame
        robot_root_translation_world = torch.tensor([3.7253e-09, 2.3283e-10, 0.0], device=self.device)
        robot_root_orientation_world = torch.tensor([1.0, -1.4891e-10, -8.7813e-10, -8.4725e-11], device=self.device)  # Quaternion (w, x, y, z)
        robot_root_rotation_world = matrix_from_quat(robot_root_orientation_world)

        # Compute inverse transformation (world → robot root)
        robot_root_rotation_inverse = robot_root_rotation_world.T  # Transpose of rotation matrix
        robot_root_translation_inverse = -torch.matmul(robot_root_rotation_inverse, robot_root_translation_world)

        # Transform grasping pose to robot root frame
        grasp_translation_robot_root = (
            torch.matmul(robot_root_rotation_inverse, grasp_translation_world) +
            robot_root_translation_inverse
        )
        grasp_rotation_robot_root = torch.matmul(robot_root_rotation_inverse, grasp_rotation_world)
        grasp_orientation_robot_root = quat_from_matrix(grasp_rotation_robot_root)
        # grasp_orientation_robot_root = torch.tensor([0.0, 1.0, 0.0, 0.0], device='cuda:0')
        # Output results
        # print("Grasping position in robot root coordinates (translation):")
        # print(grasp_translation_robot_root)

        # print("Grasping orientation in robot root coordinates (rotation matrix):")
        # print(grasp_rotation_robot_root.cpu().numpy())

        # print("Grasping orientation in robot root coordinates (quaternion):")
        # print(grasp_orientation_robot_root)

        # Define grasp translation and orientation as a single goal
        return torch.cat((grasp_translation_robot_root, grasp_orientation_robot_root), dim=0)  # Concatenate translation and quaternion


def main():
    # parse configuration
    env_cfg = parse_env_cfg(
        "My-Custom-Env-v0",
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    # create environment
    env = gym.make("My-Custom-Env-v0", cfg=env_cfg)
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

    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # step environment
            dones = env.step(actions)[-2]

            # observations
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
            print(f"print current desired pos:{torch.cat([desired_position, desired_orientation], dim=-1)}")

            # advance state machine
            # actions = pick_sm.compute(
            #    torch.cat([tcp_rest_position, tcp_rest_orientation], dim=-1),
            #    torch.cat([object_position, desired_orientation], dim=-1),
            #    torch.cat([desired_position, desired_orientation], dim=-1),
            # )
            # advance state machine anygrasp
            # grasp_goal = torch.tensor([0.25, 0.0, 0.8, 0.0, 1.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
            grasp_goal = pick_sm.get_grasping_pos().repeat(env.unwrapped.num_envs, 1)
            actions = pick_sm.compute(
                torch.cat([tcp_rest_position, tcp_rest_orientation], dim=-1),
                grasp_goal,
                torch.cat([desired_position, desired_orientation], dim=-1),
            )

            print(f"print current actions:{actions}")
            # reset state machine
            if dones.any():
                pick_sm.reset_idx(dones.nonzero(as_tuple=False).squeeze(-1))

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
