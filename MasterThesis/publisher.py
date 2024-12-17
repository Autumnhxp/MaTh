import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class PublisherNode(Node):
    def __init__(self):
        super().__init__('example_publisher')
        self.publisher = self.create_publisher(String, 'example_topic', 10)
        self.bridge_subscriber = self.create_subscription(
            String,
            'bridge_topic',  # 假設 bridge_node 發布在 'bridge_topic'
            self.bridge_callback,
            10
        )
        self.counter = 1
        self.waiting = False
        self.timer = self.create_timer(1.0, self.publish_message)
        self.get_logger().info("Publisher initialized and publishing to 'example_topic'")

    def publish_message(self):
        if self.waiting:
            # 當處於等待狀態時，不發布新的訊息
            return

        msg = String()
        msg.data = f'Hello from ROS 2 Publisher! Time: {self.counter}'
        self.publisher.publish(msg)
        self.get_logger().info(f'Publishing: {msg.data}')

        if self.counter % 10 == 0:
            self.get_logger().info(f'Time {self.counter} is a multiple of 10. Waiting for bridge_node message...')
            self.waiting = True
            self.timer.cancel()  # 暫停定時器，等待消息
        self.counter += 1

    def bridge_callback(self, msg):
        if self.waiting:
            self.get_logger().info(f'Received from bridge_node: {msg.data}. Resuming publishing...')
            self.waiting = False
            self.timer = self.create_timer(1.0, self.publish_message)  # 恢復定時器

def main(args=None):
    rclpy.init(args=args)
    node = PublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down publisher...')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
