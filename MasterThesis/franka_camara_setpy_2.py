import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates franka simulation environment")
parser.add_argument(
    "--draw",
    action="store_true",
    default=False,
    help="Draw the pointcloud from camera at index specified by ``--camera_id``.",
)
parser.add_argument(
    "--save",
    action="store_true",
    default=False,
    help="Save the data from camera at index specified by ``--camera_id``.",
)
parser.add_argument(
    "--camera_id",
    type=int,
    choices={0, 1},
    default=0,
    help=(
        "The camera ID to use for displaying points or saving the camera data. Default is 0."
        " The viewport will always initialize with the perspective of camera 0."
    ),
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
args_cli.enable_cameras = True
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np
import os
import torch

import omni.isaac.core.utils.prims as prim_utils
import omni.replicator.core as rep

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import AssetBaseCfg
from omni.isaac.lab.assets import Articulation,RigidObject,RigidObjectCfg
from omni.isaac.lab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg,CollisionPropertiesCfg
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG
from omni.isaac.lab.sensors.camera import Camera, CameraCfg
from omni.isaac.lab.utils import convert_dict_to_backend
from omni.isaac.lab.utils.math import subtract_frame_transforms

##
# Pre-defined configs
##
# isort: off
from omni.isaac.lab_assets import (
    FRANKA_PANDA_CFG,
)

# isort: on


def define_origins(num_origins: int, spacing: float) -> list[list[float]]:
    """Defines the origins of the the scene."""
    # create tensor based on number of environments
    env_origins = torch.zeros(num_origins, 3)
    # create a grid of origins
    num_rows = np.floor(np.sqrt(num_origins))
    num_cols = np.ceil(num_origins / num_rows)
    xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols), indexing="xy")
    env_origins[:, 0] = spacing * xx.flatten()[:num_origins] - spacing * (num_rows - 1) / 2
    env_origins[:, 1] = spacing * yy.flatten()[:num_origins] - spacing * (num_cols - 1) / 2
    env_origins[:, 2] = 0.0
    # return the origins
    return env_origins.tolist()

def define_sensor() -> Camera:
    """Defines the camera sensor to add to the scene."""
    # Setup camera sensor
    camera_cfg = CameraCfg(
        # This means the camera sensor will be attached to these prims.
        prim_path="/World/Origin.*/Robot/panda_hand/CameraSensor",
        update_period=0,
        height=720,
        width=1280,
        data_types=[
            "rgb",
            "distance_to_image_plane",
            "normals",
            #"semantic_segmentation",
            #"instance_segmentation_fast",
            #"instance_id_segmentation_fast",
        ],
        #colorize_semantic_segmentation=True,
        #colorize_instance_id_segmentation=True,
        #colorize_instance_segmentation=True,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24, #focal_length=60.72
            focus_distance=400.0, 
            horizontal_aperture=20.955, 
            clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.015), rot=(1.0, 0.0, 0.0, 0.0), convention="ros"),
    )
    # Create camera
    camera = Camera(cfg=camera_cfg)

    return camera


def design_scene() -> tuple[dict, list[list[float]]]:
    """Designs the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Create a dictionary for the scene entities
    scene_entities = {}

    # Create separate groups called "Origin1", "Origin2", "Origin3"...
    # Each group will have a mount and a robot on top of it
    origins = define_origins(num_origins=1, spacing=2.0)

    # Origin 1 with Franka Panda
    prim_utils.create_prim("/World/Origin1", "Xform", translation=origins[0])
    # -- Human
    human_cfg = sim_utils.UsdFileCfg(usd_path = f"{ISAAC_NUCLEUS_DIR}/People/Characters/F_Medical_01/F_Medical_01.usd",scale=(1.0, 1.0, 1.0))
    human_cfg.func("/World/Origin1/human", human_cfg, translation=(0.2, 0.7, 0.0), orientation=(0.0, 0.0, 0.0, 0.0))
    # -- Table
    table_cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/ThorlabsTable/table_instanceable.usd")
    table_cfg.func("/World/Origin1/Table", table_cfg, translation=(0.0, 0.0, 0.8))
    # -- Object
    cfg_cube = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                    scale=(0.8, 0.8, 0.8),
                    rigid_props=RigidBodyPropertiesCfg(
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=1,
                        max_angular_velocity=1000.0,
                        max_linear_velocity=1000.0,
                        max_depenetration_velocity=5.0,
                        disable_gravity=False,)
    )
    cfg_cylinder = sim_utils.MeshCylinderCfg(
        radius=0.05,
        height=0.1,
        rigid_props=RigidBodyPropertiesCfg(
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=1,
                        max_angular_velocity=1000.0,
                        max_linear_velocity=1000.0,
                        max_depenetration_velocity=5.0,
                        disable_gravity=False,),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.216),
        collision_props=CollisionPropertiesCfg(collision_enabled=True,
                                               contact_offset=0.001,
                                               min_torsional_patch_radius=0.008,
                                               rest_offset=0,
                                               torsional_patch_radius=0.1,),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.5, 0.3),metallic=0.5),
        physics_material=sim_utils.RigidBodyMaterialCfg(),
    )

    object_cfg1= RigidObjectCfg(
        prim_path="/World/Origin1/Object1",
        spawn=cfg_cube,
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.8),rot=(1, 0, 0, 0)),
    )
    object_cfg2= RigidObjectCfg(
        prim_path="/World/Origin1/Object2",
        spawn=cfg_cylinder,
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.3, 0.0, 0.8),rot=(1, 0, 0, 0)),
    )
    scene_entities[f"env1_grasping_object1"] = RigidObject(cfg=object_cfg1)
    scene_entities[f"env1_grasping_object2"] = RigidObject(cfg=object_cfg2)
    
    # -- Robot
    franka_arm_cfg = FRANKA_PANDA_CFG.replace(prim_path="/World/Origin1/Robot")
    franka_arm_cfg.init_state.pos = (0.0, 0.0, 0.8)
    scene_entities[f"env1_robot"] = Articulation(cfg=franka_arm_cfg)

    # -- Sensors
    camera = define_sensor()
    scene_entities["env1_camera"] = camera

    # # Origin 2 with UR10
    # prim_utils.create_prim("/World/Origin2", "Xform", translation=origins[1])
    # # -- Table
    # cfg = sim_utils.UsdFileCfg(
    #     usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/Stand/stand_instanceable.usd", scale=(2.0, 2.0, 2.0)
    # )
    # cfg.func("/World/Origin2/Table", cfg, translation=(0.0, 0.0, 1.03))
    # # -- Robot
    # ur10_cfg = UR10_CFG.replace(prim_path="/World/Origin2/Robot")
    # ur10_cfg.init_state.pos = (0.0, 0.0, 1.03)
    # ur10 = Articulation(cfg=ur10_cfg)

    # # Origin 3 with Kinova JACO2 (7-Dof) arm
    # prim_utils.create_prim("/World/Origin3", "Xform", translation=origins[2])
    # # -- Table
    # cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/ThorlabsTable/table_instanceable.usd")
    # cfg.func("/World/Origin3/Table", cfg, translation=(0.0, 0.0, 0.8))
    # # -- Robot
    # kinova_arm_cfg = KINOVA_JACO2_N7S300_CFG.replace(prim_path="/World/Origin3/Robot")
    # kinova_arm_cfg.init_state.pos = (0.0, 0.0, 0.8)
    # kinova_j2n7s300 = Articulation(cfg=kinova_arm_cfg)

    # # Origin 4 with Kinova JACO2 (6-Dof) arm
    # prim_utils.create_prim("/World/Origin4", "Xform", translation=origins[3])
    # # -- Table
    # cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/ThorlabsTable/table_instanceable.usd")
    # cfg.func("/World/Origin4/Table", cfg, translation=(0.0, 0.0, 0.8))
    # # -- Robot
    # kinova_arm_cfg = KINOVA_JACO2_N6S300_CFG.replace(prim_path="/World/Origin4/Robot")
    # kinova_arm_cfg.init_state.pos = (0.0, 0.0, 0.8)
    # kinova_j2n6s300 = Articulation(cfg=kinova_arm_cfg)

    # # Origin 5 with Sawyer
    # prim_utils.create_prim("/World/Origin5", "Xform", translation=origins[4])
    # # -- Table
    # cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd")
    # cfg.func("/World/Origin5/Table", cfg, translation=(0.55, 0.0, 1.05))
    # # -- Robot
    # kinova_arm_cfg = KINOVA_GEN3_N7_CFG.replace(prim_path="/World/Origin5/Robot")
    # kinova_arm_cfg.init_state.pos = (0.0, 0.0, 1.05)
    # kinova_gen3n7 = Articulation(cfg=kinova_arm_cfg)

    # # Origin 6 with Kinova Gen3 (7-Dof) arm
    # prim_utils.create_prim("/World/Origin6", "Xform", translation=origins[5])
    # # -- Table
    # cfg = sim_utils.UsdFileCfg(
    #     usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/Stand/stand_instanceable.usd", scale=(2.0, 2.0, 2.0)
    # )
    # cfg.func("/World/Origin6/Table", cfg, translation=(0.0, 0.0, 1.03))
    # # -- Robot
    # sawyer_arm_cfg = SAWYER_CFG.replace(prim_path="/World/Origin6/Robot")
    # sawyer_arm_cfg.init_state.pos = (0.0, 0.0, 1.03)
    # sawyer = Articulation(cfg=sawyer_arm_cfg)

    return scene_entities, origins


def run_simulator(sim: sim_utils.SimulationContext, entities: dict, origins: torch.Tensor):
    """Runs the simulation loop."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    
    
    # Create controller
    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=origins.shape[0], device=sim.device)

    robot_info = {
        'ee_frame_idx': None,
        'ee_jacobi_idx': None,
        'arm_joint_ids': None,
    }

    for key, value in entities.items():
        if key.endswith("_robot"):
            robot = value
            # Obtain the frame index of the end-effector
            robot_info['ee_frame_idx'] = robot.find_bodies("panda_hand")[0][0]
            robot_info['ee_jacobi_idx'] = robot_info['ee_frame_idx']-1
            # Obtain joint indices
            robot_info['arm_joint_ids'] = robot.find_joints("panda_joint.*")[0]
            break
    else:
            print("Could not find a robot")

    # Define goals for the arm
    ee_goals = [
        [0.3, 0.3, 0.3, 0.0, 1.0, 0.0, 0.0],
        [0.3, 0.3, 0.3, 0.0, 1.0, 0.0, 0.0],
        [0.3, 0.3, 0.3, 0.0, 1.0, 0.0, 0.0],
    ]

    ee_goals = torch.tensor(ee_goals, device=sim.device)
    
    # Track the given command
    current_goal_idx = 0
    
    # Create buffers to store actions
    ik_commands = torch.zeros(origins.shape[0], diff_ik_controller.action_dim, device=sim.device)
    ik_commands[:] = ee_goals[current_goal_idx]

    # Markers
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))
        
    # Camera Initialization -------
    # extract entities for simplified notation
    camera: Camera = entities["env1_camera"]

    # Create replicator writer
    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output", "env1_camera")
    rep_writer = rep.BasicWriter(
        output_dir=output_dir,
        frame_padding=0,
        #colorize_instance_id_segmentation=camera.cfg.colorize_instance_id_segmentation,
        #colorize_instance_segmentation=camera.cfg.colorize_instance_segmentation,
        #colorize_semantic_segmentation=camera.cfg.colorize_semantic_segmentation,
    )

    ## Camera positions, targets, orientations
    #camera_positions = torch.tensor([[0.4, 0.0, 3.0]], device=sim.device) + origins[0]
    #camera_targets = torch.tensor([[0.4, 0.0, 0.0]], device=sim.device) + origins[0]
    #
    ## Set pose: There are two ways to set the pose of the camera.
    #camera.set_world_poses_from_view(camera_positions, camera_targets)

    # Index of the camera to use for visualization and saving
    camera_index = args_cli.camera_id

    # Camera Initialization End ------

    
    
    # Simulate physics
    while simulation_app.is_running():
        # reset
        if count % 2000 == 0:
            # reset counters
            sim_time = 0.0
            count = 0
            # reset the scene entities
            for key, value in entities.items():
                if key.endswith("_robot"):
                    robot = value
                    # root state
                    root_state = robot.data.default_root_state.clone()
                    num_str = key[len("env"):key.index("_robot")]
                    index = int(num_str)-1
                    root_state[:, :3] += origins[index]
                    robot.write_root_state_to_sim(root_state)
                    # set joint positions
                    joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
                    robot.write_joint_state_to_sim(joint_pos, joint_vel)
                    # clear internal buffers
                    robot.reset()
                    print("[INFO]: Resetting robots state...")
                    
                    # reset actions
                    ik_commands[:] = ee_goals[current_goal_idx]
                    joint_pos_des = joint_pos[:, robot_info["arm_joint_ids"]].clone()
                    # reset controller
                    diff_ik_controller.reset()
                    diff_ik_controller.set_command(ik_commands)
                    # change goal
                    current_goal_idx = (current_goal_idx + 1) % len(ee_goals)
                    print("[INFO]: Resetting robots controller...")

        else:
            # obtain quantities from simulation
            jacobian = robot.root_physx_view.get_jacobians()[:, robot_info["ee_jacobi_idx"], :, robot_info["arm_joint_ids"]]
            ee_pose_w = robot.data.body_state_w[:, robot_info["ee_frame_idx"], 0:7]
            root_pose_w = robot.data.root_state_w[:, 0:7]
            joint_pos = robot.data.joint_pos[:, robot_info["arm_joint_ids"]]
            # compute frame in root frame
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
            )
            # compute the joint commands
            joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

        # apply actions to the robots
        for key, value in entities.items():
            if key.endswith("_robot"):
                robot = value
                # generate random joint positions
                # joint_pos_target = robot.data.default_joint_pos + torch.randn_like(robot.data.joint_pos) * 0.1
                # joint_pos_target = robot.data.default_joint_pos 
                # joint_pos_target = joint_pos_target.clamp_(
                #     robot.data.soft_joint_pos_limits[..., 0], robot.data.soft_joint_pos_limits[..., 1]
                # )
                # apply action to the robot
                robot.set_joint_position_target(joint_pos_des, joint_ids=robot_info["arm_joint_ids"])
                # write data to sim
                robot.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        for key, value in entities.items():
            if key.endswith("_robot"):
                robot = value
                robot.update(sim_dt)
        # obtain quantities from simulation
        ee_pose_w = robot.data.body_state_w[:, robot_info["ee_frame_idx"], 0:7]
        # update marker positions
        ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
        goal_marker.visualize(ik_commands[:, 0:3] + origins.shape[0], ik_commands[:, 3:7])

        # Camera
        # Update camera data
        camera.update(dt=sim_dt)

        # Print camera info
        print(camera)
        if "rgb" in camera.data.output.keys():
            print("Received shape of rgb image        : ", camera.data.output["rgb"].shape)
        if "distance_to_image_plane" in camera.data.output.keys():
            print("Received shape of depth image      : ", camera.data.output["distance_to_image_plane"].shape)
        if "normals" in camera.data.output.keys():
            print("Received shape of normals          : ", camera.data.output["normals"].shape)
        if "semantic_segmentation" in camera.data.output.keys():
            print("Received shape of semantic segm.   : ", camera.data.output["semantic_segmentation"].shape)
        if "instance_segmentation_fast" in camera.data.output.keys():
            print("Received shape of instance segm.   : ", camera.data.output["instance_segmentation_fast"].shape)
        if "instance_id_segmentation_fast" in camera.data.output.keys():
            print("Received shape of instance id segm.: ", camera.data.output["instance_id_segmentation_fast"].shape)
        print("-------------------------------")

        # Extract camera data
        if args_cli.save:
            # Save images from camera at camera_index
            # note: BasicWriter only supports saving data in numpy format, so we need to convert the data to numpy.
            # tensordict allows easy indexing of tensors in the dictionary
            single_cam_data = convert_dict_to_backend(camera.data.output[camera_index], backend="numpy")

            # Extract the other information
            single_cam_info = camera.data.info[camera_index]

            # Pack data back into replicator format to save them using its writer
            rep_output = {"annotators": {}}
            for key, data, info in zip(single_cam_data.keys(), single_cam_data.values(), single_cam_info.values()):
                if info is not None:
                    rep_output["annotators"][key] = {"render_product": {"data": data, **info}}
                else:
                    rep_output["annotators"][key] = {"render_product": {"data": data}}
            # Save images
            # Note: We need to provide On-time data for Replicator to save the images.
            rep_output["trigger_outputs"] = {"on_time": camera.frame[camera_index]}
            rep_writer.write(rep_output)


def main():
    """Main function."""
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg()
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])
    # design scene
    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene_entities, scene_origins)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
