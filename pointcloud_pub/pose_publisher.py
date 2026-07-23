import rclpy
from rclpy.node import Node
import numpy as np

from geometry_msgs.msg import PoseArray, Pose
from std_msgs.msg import Header


class PosePublisher(Node):
    def __init__(self, node):
        self.publisher = node.create_publisher(PoseArray, 'poses', 10)

    def create_pose_array(self, centroids, orientations, header): # <-- Change argument to header
        
        """
        centroids: numpy array of shape (n,3) or (n,4)
        columns: [x,y,z,(optional class)]
        """

        print("values of centroids:", centroids.values())

        # Ensure centroids is a 2D array
        centroids = np.asarray(list(centroids.values()), dtype=np.float32)
        assert centroids.ndim == 2, f"centroids must be 2D, got {centroids.ndim}D"

        msg = PoseArray()
        msg.header = header
        print(range(centroids.shape[0]))
        for cluster_id in range(centroids.shape[0]):
            print(cluster_id)
            centroid = centroids[cluster_id]
            quaternion = orientations[cluster_id]
            
            pose = Pose()
            
            # Centroid is a 1D array: [x, y, z]
            pose.position.x = float(centroid[0])
            pose.position.y = float(centroid[1])
            pose.position.z = float(centroid[2])

            # Quaternion is a 1D array: [x, y, z, w]
            pose.orientation.x = float(quaternion[0])
            pose.orientation.y = float(quaternion[1])
            pose.orientation.z = float(quaternion[2])
            pose.orientation.w = float(quaternion[3])

            msg.poses.append(pose)

        return msg
