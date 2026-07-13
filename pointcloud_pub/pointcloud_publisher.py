from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import rclpy
import numpy as np


class PointCloudPublisher(Node):
    def __init__(self, node: Node, topic_name: str):
        self.publisher = node.create_publisher(PointCloud2, topic_name, 10)

    def create_pointcloud2(self, points, frame):
        from sensor_msgs.msg import PointField
        from std_msgs.msg import Header

        # 1. CHANGE THIS to expect 4 columns (x, y, z, class)
        assert points.ndim == 2 and points.shape[1] == 4

        points = np.asarray(points, dtype=np.float32)
        points = np.ascontiguousarray(points)

        msg = PointCloud2()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame

        msg.height = 1
        msg.width = points.shape[0]

        # 4 fields * 4 bytes = 16 bytes per point
        msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="class", offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        msg.is_bigendian = False
        msg.point_step = 16
        msg.row_step = msg.point_step * msg.width
        msg.is_dense = not np.isnan(points).any()

        msg.data = points.tobytes()

        return msg
