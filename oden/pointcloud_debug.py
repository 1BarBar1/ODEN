import rclpy
from rclpy.node import Node

import numpy as np
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2


class PointCloudInspector(Node):

    def __init__(self):
        super().__init__('pointcloud_inspector')
        self.subscription = self.create_subscription(
            PointCloud2,
            '/points',
            self.callback,
            10
        )

    def callback(self, msg: PointCloud2):
        # Decode cloud into Python tuples
        points = np.array(list(
            point_cloud2.read_points(
                msg,
                field_names=('x', 'y', 'z', 'class'),
                skip_nans=False
            )
        ), dtype=np.float32)

        # Print shape
        self.get_logger().info(f"Shape: {points.shape}")

        # Print unique class values
        classes = points[:, 3]
        unique_classes = np.unique(classes)
        self.get_logger().info(f"Unique classes: {unique_classes}")

        # Optional: print first few rows
        self.get_logger().info(f"First 5 points:\n{points[:5]}")


def main():
    rclpy.init()
    node = PointCloudInspector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()