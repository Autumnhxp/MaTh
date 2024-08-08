# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})


from omni.isaac.core import World
from omni.isaac.core.utils.stage import open_stage
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_stage_units
from omni.isaac.franka.controllers.rmpflow_controller import RMPFlowController
from omni.isaac.franka.tasks import FollowTarget
from omni.isaac.franka import Franka
from omni.isaac.core.objects import DynamicCylinder
from pxr import UsdGeom
from omni.isaac.core.prims.rigid_prim import RigidPrim

usd_file_path = "omniverse://localhost/Users/Autumn/basic_environment/Environment.usd"
franka_prim_path = "/Xform/Franka"
object_prim_path = "/World/Cylinde"

# open stage
open_stage(usd_file_path)
print(f"Loading USD file form path: {usd_file_path}")

my_world = World(stage_units_in_meters=1.0)


prim_franka = get_prim_at_path(franka_prim_path)
if prim_franka:
    print(f"Successfully loaded Franka at {franka_prim_path}")
else:
    print(f"Failed to load Franka at {franka_prim_path}")

prim_object = get_prim_at_path(object_prim_path)
if prim_object:
    print(f"Successfully loaded Object at {object_prim_path}")
    

else:
    print(f"Failed to load Object at {object_prim_path}")

# Ensure the object is a RigidPrim
try:
    target = RigidPrim(prim_path=object_prim_path, name="Cylinder")
    print("RigidPrim initialized successfully.")
except Exception as e:
    print(f"Error initializing RigidPrim: {e}")

target_position, target_orientation = target.get_world_pose()
print(f"target position: {target_position} and target position type: {type(target_position)}")
print(f"target orientation: {target_orientation} and target orientation type: {type(target_orientation)}")

my_task = FollowTarget(name="follow_target_task",
                       target_prim_path=object_prim_path,
                       target_name=target.name,
                       target_position=target_position,
                       target_orientation=target_orientation,
                       franka_prim_path=franka_prim_path)
my_world.add_task(my_task)
my_world.reset()
task_params = my_world.get_task("follow_target_task").get_params()
franka_name = task_params["robot_name"]["value"]
target_name = task_params["target_name"]["value"]

# target_name = my_task["target_name"]["value"]

franka = my_world.scene.get_object(franka_name)
# franka = Franka(prim_path=franka_prim_path)
my_controller = RMPFlowController(name="target_follower_controller", robot_articulation=franka)
articulation_controller = franka.get_articulation_controller() 

simulation_app.update()

reset_needed = False

while simulation_app.is_running():
    my_world.step(render=True)
    if my_world.is_stopped() and not reset_needed:
        reset_needed = True
    if my_world.is_playing():
        if reset_needed:
            my_world.reset()
            my_controller.reset()
            reset_needed = False
        observations = my_world.get_observations()
        
        actions = my_controller.forward(
            target_end_effector_position=observations[target_name]["position"],
            target_end_effector_orientation=observations[target_name]["orientation"],
        #    target_end_effector_position = target_position / get_stage_units(),
        #    target_end_effector_orientation = target_orientation,
        )
        articulation_controller.apply_action(actions)

simulation_app.close()