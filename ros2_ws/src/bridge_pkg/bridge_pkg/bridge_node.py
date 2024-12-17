# bridge_pkg/bridge_pkg/bridge_node.py

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

        # ROS 2 订阅者
        self.example_subscription = self.create_subscription(
            String,
            'example_topic',
            partial(self.listener_callback, topic_name='example_topic'),
            10)
        self.example_subscription  # 防止未使用变量警告

        # Subscribe Simulation State
        self.isaaclab_state_subscription = self.create_subscription(
            Int8MultiArray,
            'sm_state_wp',
            partial(self.listener_callback, topic_name='sm_state_wp'),
            10
        )
        self.isaaclab_state_subscription

        # Subscribe Take Foto Action
        self.isaaclab_take_foto_subscription = self.create_subscription(
            Float32MultiArray,
            'take_foto_wp',
            partial(self.listener_callback, topic_name='take_foto_wp'),
            10
        )
        self.isaaclab_take_foto_subscription

        # Subscribe File Paths
        self.isaaclab_file_paths_subscription = self.create_subscription(
            String,
            'file_paths',
            partial(self.listener_callback, topic_name = 'file_paths'),
            10
        )
        self.isaaclab_file_paths_subscription

        # ROS 2 发布者，用于将数据从 Python 3.8 客户端发布到 ROS 2
        self.publisher_ = self.create_publisher(String, 'grasp_results', 10)

        # 设置 ZeroMQ 上下文和套接字，重命名为 zmq_context 以避免冲突
        self.zmq_context = zmq.Context()

        # PUB 套接字：用于向 Python 3.8 客户端发布 ROS 2 数据
        self.pub_socket = self.zmq_context.socket(zmq.PUB)
        try:
            self.pub_socket.bind("tcp://*:5555")  # 绑定到本地端口 5555
            self.get_logger().info('PUB socket bound to tcp://*:5555')
        except Exception as e:
            self.get_logger().error(f'Failed to bind PUB socket: {e}')

        # SUB 套接字：用于从 Python 3.8 客户端接收数据
        self.sub_socket = self.zmq_context.socket(zmq.SUB)
        try:
            self.sub_socket.connect("tcp://localhost:5556")  # 连接到 Python 3.8 客户端的端口 5556
            self.get_logger().info('SUB socket connected to tcp://localhost:5556')
            self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, '')  # 订阅所有消息
        except Exception as e:
            self.get_logger().error(f'Failed to bind SUB socket: {e}')
        
            


        # 启动接收线程
        self.receive_thread = threading.Thread(target=self.receive_data)
        self.receive_thread.daemon = True
        self.receive_thread.start()

        self.get_logger().info('BridgeNode is up and running.')

    def listener_callback(self, msg, topic_name):
        try:
            # 将消息内容包装为 JSON，并添加主题名称
            data = {'topic': topic_name, 'data': None}

            if topic_name == 'file_paths':
                # 直接将字符串转换为 JSON 对象
                try:
                    data['data'] = json.loads(msg.data)  # 确保是合法的 JSON 格式
                except json.JSONDecodeError as e:
                    self.get_logger().error(f"Invalid JSON format for topic {topic_name}: {e}")
                    return
            else:
                if hasattr(msg, 'data'):
                    if hasattr(msg.data, 'tolist'):  # 处理 MultiArray 类型
                        data['data'] = msg.data.tolist()
                    else:
                        data['data'] = list(msg.data)
                else:
                    self.get_logger().warning(f'Unsupported message format for topic: {topic_name}')
                    return

            # 序列化为 JSON 并通过 ZeroMQ 发送
            json_data = json.dumps(data)
            self.pub_socket.send_string(json_data)
            self.get_logger().info(f'Forwarded message to Python 3.8 from topic {topic_name}: {json_data}')
        except Exception as e:
            self.get_logger().error(f'Error in listener_callback for topic {topic_name}: {e}')

    def receive_data(self):
        while True:
            try:
                message = self.sub_socket.recv_string()
                data = json.loads(message)
                self.get_logger().info(f'Received data from Python 3.8: {data}')
                # 将接收到的数据发布到 ROS 2 话题
                ros_msg = String()
                ros_msg.data = json.dumps(data)
                self.publisher_.publish(ros_msg)
            except Exception as e:
                self.get_logger().error(f'Error receiving data: {e}')

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
