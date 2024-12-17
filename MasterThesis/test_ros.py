# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to run IsaacSim via the AppLauncher

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p source/standalone/tutorials/06_ros/publish_pointcloud_test.py

"""

"""Launch Isaac Sim Simulator first."""


import argparse
from omni.isaac.lab.app import AppLauncher

# create argparser
parser = argparse.ArgumentParser(description="Tutorial on running IsaacSim via the AppLauncher.")
parser.add_argument("--size", type=float, default=1.0, help="Side-length of cuboid")
# SimulationApp arguments https://docs.omniverse.nvidia.com/py/isaacsim/source/extensions/omni.isaac.kit/docs/index.html?highlight=simulationapp#omni.isaac.kit.SimulationApp
parser.add_argument(
    "--width", type=int, default=1280, help="Width of the viewport and generated images. Defaults to 1280"
)
parser.add_argument(
    "--height", type=int, default=720, help="Height of the viewport and generated images. Defaults to 720"
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import omni.isaac.lab.sim as sim_utils
import omni
import carb
import numpy as np
import array
import threading
import omni.kit.app
print(f'print omni.kit.app', omni.kit.app.__file__)
ext_manager = omni.kit.app.get_app().get_extension_manager()
ext_manager.set_extension_enabled_immediate("omni.isaac.ros2_bridge", True)

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile
    from std_msgs.msg import Header
    from sensor_msgs.msg import PointCloud2, PointField
    print("import rclpy success!")
except:
    print("import rclpy failed")

class RobotBaseNode(Node):
    def __init__(self):
        super().__init__('sensor_driver_node')
        qos_profile = QoSProfile(depth=10)
        self.lidar_pub = self.create_publisher(PointCloud2, f'/point_cloud2', qos_profile)
        timer_period = 0.1
        self.lidar_timer = self.create_timer(timer_period, self.lidar_timer_callback)

    def lidar_timer_callback(self):
        self.pub_pointcloud()

    def pub_pointcloud(self):
        try:
            # 模拟实际32线激光点云数据，点的数量下降发布耗时会下降，但实际点个数有几万个
            data_points = np.ones((65536, 4))
            point_cloud = PointCloud2()
            point_cloud.header = Header(frame_id="odom")
            point_cloud.header.stamp = self.get_clock().now().to_msg()
            point_cloud.height = data_points.shape[1]
            point_cloud.width = data_points.shape[0]
            point_cloud.point_step = data_points.dtype.itemsize
            point_cloud.row_step = data_points.dtype.itemsize * data_points.shape[0]
            fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
            ]
            point_cloud.fields = fields
            # Convert numpy points to array.array
            ###########这部分不耗时###############
            memory_view = memoryview(data_points)
            casted = memory_view.cast('B')
            array_array = array.array('B')
            array_array.frombytes(casted)
            point_cloud.data = array_array
            #####################################
            # 发布消息耗时严重
            self.lidar_pub.publish(point_cloud)
        except Exception as error:
            carb.log_error("pub point cloud failed!" + str(error))

def run_ros2_node():  
    try:
        rclpy.init()  
        base_node = RobotBaseNode()
        rclpy.spin(base_node)  
    except Exception as error:
        # If anything causes your compute to fail report the error and return False
        carb.log_error("init ros2 node failed!" + str(error))

def design_scene():
    """Designs the scene by spawning ground plane, light, objects and meshes from usd files."""
    # Ground-plane
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)

    # spawn distant light
    cfg_light_distant = sim_utils.DistantLightCfg(
        intensity=3000.0,
        color=(0.75, 0.75, 0.75),
    )
    cfg_light_distant.func("/World/lightDistant", cfg_light_distant, translation=(1, 0, 10))

    # spawn a cuboid
    cfg_cuboid = sim_utils.CuboidCfg(
        size=[args_cli.size] * 3,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0)),
    )
    # Spawn cuboid, altering translation on the z-axis to scale to its size
    cfg_cuboid.func("/World/Object", cfg_cuboid, translation=(0.0, 0.0, args_cli.size / 2))


def main():
    """Main function."""

    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.01)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.0, 0.0, 2.5], [-0.5, 0.0, 0.5])

    # Design scene by adding assets to it
    design_scene()

    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")

    ros_thread = threading.Thread(target=run_ros2_node)
    ros_thread.start()

    print("[INFO]: start ros thread...")

    # Simulate physics
    while simulation_app.is_running():
        # perform step
        sim.step()

    rclpy.shutdown()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()