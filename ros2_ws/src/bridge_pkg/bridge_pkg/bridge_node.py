# bridge_pkg/bridge_pkg/bridge_node.py
#
# This script implements a ROS 2 bridge node that connects the Isaac Sim simulation environment
# with the AnyGrasp model environment. It facilitates communication between:
# 1. Isaac Sim: Provides simulation state, camera triggers, and RGBD image paths
# 2. AnyGrasp: Processes RGBD data and generates grasp poses
#
# Important Note:
# The AnyGrasp environment requires Minkowski Engine library for sparse convolution operations.
# The bridge node communicates with an AnyGrasp instance running in a separate environment
# that has Minkowski Engine properly installed. This separation allows the bridge to run in
# any Python environment while AnyGrasp runs in its specific Minkowski-enabled environment.
#
# Communication Flow:
# Isaac Sim (ROS 2) -> Bridge Node -> ZMQ -> AnyGrasp Environment
# AnyGrasp Environment -> ZMQ -> Bridge Node -> ROS 2 (Isaac Sim)

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Int8MultiArray, Float32MultiArray
import zmq
import json
import threading
from functools import partial

class BridgeNode(Node):
    def __init__(self):
        super().__init__('bridge_node')

        # ROS 2 Subscribers
        self.example_subscription = self.create_subscription(
            String,
            'example_topic',
            partial(self.listener_callback, topic_name='example_topic'),
            10)
        self.example_subscription  # Prevent unused variable warning

        # Subscribe to Simulation State from Isaac Sim
        self.isaaclab_state_subscription = self.create_subscription(
            Int8MultiArray,
            'sm_state_wp',
            partial(self.listener_callback, topic_name='sm_state_wp'),
            10
        )
        self.isaaclab_state_subscription

        # Subscribe to Camera Trigger Events from Isaac Sim
        self.isaaclab_take_foto_subscription = self.create_subscription(
            Float32MultiArray,
            'take_foto_wp',
            partial(self.listener_callback, topic_name='take_foto_wp'),
            10
        )
        self.isaaclab_take_foto_subscription

        # Subscribe to RGBD Image File Paths from Isaac Sim
        self.isaaclab_file_paths_subscription = self.create_subscription(
            String,
            'file_paths',
            partial(self.listener_callback, topic_name = 'file_paths'),
            10
        )
        self.isaaclab_file_paths_subscription

        # ROS 2 Publisher: Publishes grasp poses from AnyGrasp back to Isaac Sim
        self.publisher_ = self.create_publisher(String, 'grasp_results', 10)

        # Initialize ZMQ context for communication with AnyGrasp environment
        self.zmq_context = zmq.Context()

        # PUB socket: Forwards ROS 2 data to AnyGrasp environment
        self.pub_socket = self.zmq_context.socket(zmq.PUB)
        try:
            self.pub_socket.bind("tcp://*:5555")  # Bind to local port 5555
            self.get_logger().info('PUB socket bound to tcp://*:5555')
        except Exception as e:
            self.get_logger().error(f'Failed to bind PUB socket: {e}')

        # SUB socket: Receives grasp poses from AnyGrasp environment
        self.sub_socket = self.zmq_context.socket(zmq.SUB)
        try:
            self.sub_socket.connect("tcp://localhost:5556")  # Connect to AnyGrasp environment port
            self.get_logger().info('SUB socket connected to tcp://localhost:5556')
            self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, '')  # Subscribe to all messages
        except Exception as e:
            self.get_logger().error(f'Failed to bind SUB socket: {e}')
        
        # Start receiving thread for grasp poses
        self.receive_thread = threading.Thread(target=self.receive_data)
        self.receive_thread.daemon = True
        self.receive_thread.start()

        self.get_logger().info('BridgeNode is up and running.')

    def listener_callback(self, msg, topic_name):
        """
        Callback function for ROS 2 subscribers. Forwards messages from Isaac Sim to AnyGrasp environment.
        
        Args:
            msg: ROS 2 message (String, Int8MultiArray, or Float32MultiArray)
            topic_name: Name of the ROS 2 topic
        """
        try:
            # Package message content as JSON with topic name
            data = {'topic': topic_name, 'data': None}

            if topic_name == 'file_paths':
                # Parse file paths JSON string
                try:
                    data['data'] = json.loads(msg.data)
                except json.JSONDecodeError as e:
                    self.get_logger().error(f"Invalid JSON format for file paths: {e}")
                    return
            else:
                # Handle array messages (simulation state and camera triggers)
                if hasattr(msg, 'data'):
                    if hasattr(msg.data, 'tolist'):  # Handle MultiArray types
                        data['data'] = msg.data.tolist()
                    else:
                        data['data'] = list(msg.data)
                else:
                    self.get_logger().warning(f'Unsupported message format for topic: {topic_name}')
                    return

            # Serialize and send via ZMQ to AnyGrasp environment
            json_data = json.dumps(data)
            self.pub_socket.send_string(json_data)
            self.get_logger().info(f'Forwarded to AnyGrasp environment from topic {topic_name}: {json_data}')
        except Exception as e:
            self.get_logger().error(f'Error in listener_callback for topic {topic_name}: {e}')

    def receive_data(self):
        """
        Thread function that continuously receives grasp poses from AnyGrasp environment
        and publishes them to ROS 2 for Isaac Sim to consume.
        """
        while True:
            try:
                message = self.sub_socket.recv_string()
                data = json.loads(message)
                self.get_logger().info(f'Received grasp pose from AnyGrasp: {data}')
                # Publish grasp pose to ROS 2
                ros_msg = String()
                ros_msg.data = json.dumps(data)
                self.publisher_.publish(ros_msg)
            except Exception as e:
                self.get_logger().error(f'Error receiving grasp pose: {e}')

def main(args=None):
    rclpy.init(args=args)
    bridge_node = BridgeNode()
    try:
        rclpy.spin(bridge_node)
    except KeyboardInterrupt:
        pass
    bridge_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
