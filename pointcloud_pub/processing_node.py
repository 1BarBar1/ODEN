from pointcloud_pub.vlm import Clipseg, YoloSamCombo, Yolo26e
import numpy as np
from pointcloud_pub.pointcloud_publisher import PointCloudPublisher
from pointcloud_pub.pose_publisher import PosePublisher
from message_filters import Subscriber, ApproximateTimeSynchronizer
from sensor_msgs.msg import CameraInfo
import rclpy
from rclpy.node import Node
from rclpy.clock import Clock
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import open3d as o3d


class ProcessingNode(Node):
    def __init__(self, seg_grasping_object, seg_obstacle):
        super().__init__("processing_node")
        print("Processing node initialized")
        self.seg_grasping_object = seg_grasping_object
        self.seg_obstacle = seg_obstacle
        self.bridge = CvBridge()

        self.pc_grasping_object_pub = PointCloudPublisher(
            node=self, topic_name="object"
        )
        self.pc_environment_pub = PointCloudPublisher(
            node=self, topic_name="environment"
        )
        self.po_pub = PosePublisher(node=self)
        self.K = None

        self.frame = "camera_color_optical_frame"

        self.color_frame = None
        self.depth_frame = None

        # Match the Image topics (Reliable + Transient Local)
        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10,
        )
        print("Image QoS Profile set to Reliable + Volatile")
        # Match the Camera Info (Reliable + Volatile)
        info_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10,
        )
        # Image subscribers (using the Transient Local profile)
        self.color_sub = Subscriber(
            self, Image, "/camera/camera/color/image_raw", qos_profile=image_qos
        )
        self.depth_sub = Subscriber(
            self,
            Image,
            "/camera/camera/aligned_depth_to_color/image_raw",
            qos_profile=image_qos,
        )

        # Camera Info subscriber (using the Volatile profile)
        self.info_sub = self.create_subscription(
            CameraInfo,
            "/camera/camera/aligned_depth_to_color/camera_info",
            self.info_cb,
            info_qos,
        )
        self.time_sync = ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub], queue_size=5, slop=0.05
        )
        self.time_sync.registerCallback(self.synced_cb)

    def info_cb(self, msg):
        if self.K is None:
            self.K = np.array(msg.k).reshape(3, 3)
            self.frame = msg.header.frame_id

            self.get_logger().info("Camera info received")

            self.destroy_subscription(self.info_sub)
        if self.K is not None:
            self.get_logger().info(
                "Camera info already received, ignoring further messages"
            )

    def extract_point_clouds(
        self, depth_image, mask_grasping_object, cx, cy, fx, fy, bbox_padding=0.05
    ):
        # Ensure mask is boolean for easy logic
        mask_grasping_object = np.asarray(mask_grasping_object, dtype=bool)

        if mask_grasping_object.size == 0 or not np.any(mask_grasping_object):
            print("Warning: No mask found for grasping_object segmentation.")
            return np.array([]), np.array([])
        # Create a grid of all (u, v) pixel coordinates
        h, w = depth_image.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))

        # Convert the entire depth image to meters at once
        Z = depth_image / 1000.0

        # Create a mask to filter out invalid depths (Z <= 0 or Z > 6.0)
        valid_depth = (Z > 0) & (Z <= 6.0)

        # Calculate X and Y for the entire image simultaneously
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy

        # Stack X, Y, Z into a single 3D array of shape (H, W, 3)
        point_cloud = np.dstack((X, Y, Z))

        # Combine the valid depth mask with your grasping_object/environment masks
        # Using bitwise '&' to combine boolean conditions
        grasping_object_mask = valid_depth & mask_grasping_object
        env_mask = (
            valid_depth & ~mask_grasping_object
        )  # ~ inverts the boolean mask (gets the 0s)

        # Extract the valid points into flat lists/arrays of shape (N, 3)
        mask_points = point_cloud[grasping_object_mask]
        environment = point_cloud[env_mask]
        if mask_points.size > 0:
            # Calculate the min and max X, Y, Z coordinates to form the bounding box
            # Subtract/add padding to expand the box slightly
            min_bounds = np.min(mask_points, axis=0) - bbox_padding
            max_bounds = np.max(mask_points, axis=0) + bbox_padding

            # Find which environment points fall INSIDE this 3D bounding box
            # environment[:, 0] is X, environment[:, 1] is Y, environment[:, 2] is Z
            inside_bbox = (
                (environment[:, 0] >= min_bounds[0])
                & (environment[:, 0] <= max_bounds[0])
                & (environment[:, 1] >= min_bounds[1])
                & (environment[:, 1] <= max_bounds[1])
                & (environment[:, 2] >= min_bounds[2])
                & (environment[:, 2] <= max_bounds[2])
            )

            # Use the bitwise NOT operator (~) to keep only the points OUTSIDE the bounding box
            environment = environment[~inside_bbox]

        return mask_points, environment

    def synced_cb(self, color_msg, depth_msg):
        if self.K is None:
            print("no K")
            return
        # print("Processing synchronized color and depth frames")
        color_frame = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding="bgr8")
        depth_frame = self.bridge.imgmsg_to_cv2(
            depth_msg, desired_encoding="passthrough"
        )

        depth_image = depth_frame.astype(np.uint16)
        fx = self.K[0, 0]
        fy = self.K[1, 1]
        cx = self.K[0, 2]
        cy = self.K[1, 2]

        if self.seg_grasping_object:
            mask_points = []
            environment = []
            mask_grasping_object = self.seg_grasping_object.get_segmentation(
                color_frame
            )
            mask_grasping_object = np.asarray(mask_grasping_object, dtype=bool)
            mask_points, environment_np = self.extract_point_clouds(
                depth_image, mask_grasping_object, cx, cy, fx, fy
            )

            try:
                if environment_np.size == 0:
                    print("Warning: No points found for grasping_object segmentation.")
                    return
                environment = o3d.geometry.PointCloud()
                environment.points = o3d.utility.Vector3dVector(environment_np)

                voxel_size = 0.01
                # environment = environment.voxel_down_sample(voxel_size)
                grasping_object = np.asarray(mask_points)

                grasping_object_np = np.asarray(mask_points)
                if grasping_object_np.size == 0:
                    print("Warning: No points found for grasping_object segmentation.")
                    return

                grasping_object = o3d.geometry.PointCloud()
                grasping_object.points = o3d.utility.Vector3dVector(grasping_object_np)

                voxel_size = 0.01
                grasping_object = grasping_object.voxel_down_sample(voxel_size)

                # Check if points exist before clustering to avoid the crash
                if len(grasping_object.points) == 0:
                    print("Warning: Point cloud empty after downsampling.")
                else:
                    eps = 0.05  # 5 centimeters
                    min_points = 10
                    labels = np.array(
                        grasping_object.cluster_dbscan(
                            eps=eps, min_points=min_points, print_progress=False
                        )
                    )

                    # Labels of -1 represent noise points that do not belong to any cluster
                    print(labels)
                    max_label = labels.max()
                    print(f"Point cloud broken into {max_label + 1} distinct clusters.")

                    centroids = {}

                    for cluster_id in range(max_label + 1):
                        # Extract indices belonging exclusively to the current cluster
                        cluster_indices = np.where(labels == cluster_id)[0]

                        # Slice the downsampled point cloud to isolate the cluster geometry
                        cluster_pcd = grasping_object.select_by_index(cluster_indices)

                        # Method A: Use Open3D's built-in center calculation
                        # centroid = cluster_pcd.get_center()

                        # Method B: Direct mathematical mean via NumPy (Alternative)
                        cluster_points = np.asarray(cluster_pcd.points)
                        centroid = cluster_points.mean(axis=0)

                        centroids[cluster_id] = centroid
                        print(
                            f"Cluster {cluster_id}: Points = {len(cluster_indices)} | Centroid = {centroid}"
                        )
                    grasping_object_array = np.asarray(grasping_object.points)
                    grasping_object_array = np.column_stack(
                        (grasping_object_np, np.ones(grasping_object_np.shape[0]))
                    )
                    environment_array = np.asarray(environment.points)
                    environment_array = np.column_stack(
                        (environment_array, np.ones(environment_array.shape[0]))
                    )

                    if len(grasping_object.points) > 0:
                        self.get_logger().info(
                            f"Publishing grasping_object point cloud with {grasping_object_array.shape[0]} points."
                        )
                        msg_grasping_object = (
                            self.pc_grasping_object_pub.create_pointcloud2(
                                grasping_object_array, self.frame
                            )
                        )
                        self.pc_grasping_object_pub.publisher.publish(
                            msg_grasping_object
                        )
                    if environment_array.size > 0:
                        self.get_logger().info(
                            f"Publishing environment point cloud with {environment_array.shape[0]} points."
                        )
                        msg_env = self.pc_environment_pub.create_pointcloud2(
                            environment_array, self.frame
                        )
                        self.pc_environment_pub.publisher.publish(msg_env)
                    if len(centroids) > 0:
                        msg_pose = self.po_pub.create_pose_array(centroids, self.frame)
                        self.po_pub.publisher.publish(msg_pose)
            except Exception as e:
                print("mask error")

        self.color_frame = None
        self.depth_frame = None


def main(args=None):
    print("Starting processing node...")

    seg_grasping_object = YoloSamCombo(["brick", "object", "tool", "box", ""])
    seg_obstacle = None

    rclpy.init(args=args)
    node = ProcessingNode(seg_grasping_object, seg_obstacle)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
