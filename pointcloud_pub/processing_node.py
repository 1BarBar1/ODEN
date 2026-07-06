from pointcloud_pub.vlm import Clipseg, YoloSamCombo, Yolo26e
import numpy as np
import math
import time
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
    def __init__(self, seg_human, seg_obstacle):
        super().__init__("processing_node")
        print("Processing node initialized")
        self.seg_human = seg_human
        self.seg_obstacle = seg_obstacle
        self.bridge = CvBridge()
        self.pc_human_pub = PointCloudPublisher(topic_name="human")
        self.pc_obstacle_pub = PointCloudPublisher(topic_name="obstacle")
        self.po_pub = PosePublisher()
        self.K = None

        self.frame = "camera1_depth_optical_frame"

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

    def transformation(self, points):

        xyz = points[:, :3]  # spatial coordinates
        classes = points[:, 3]  # class labels

        # --- Rotation Z (90°)
        a = math.radians(-90)
        Rz = np.array(
            [[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]]
        )

        # --- Rotation Y (90°)
        a = math.radians(90)
        Ry1 = np.array(
            [[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]]
        )

        # --- Rotation Y (42°)
        a = math.radians(50)
        Ry2 = np.array(
            [[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]]
        )

        rotated = xyz @ Rz.T
        rotated = rotated @ Ry1.T
        rotated = rotated @ Ry2.T

        rotated[:, 2] += 1.2

        return np.column_stack((rotated, classes))

    def synced_cb(self, color_msg, depth_msg):
        if self.K is None:
            print("no K")
            return
        print("Processing synchronized color and depth frames")
        color_frame = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding="bgr8")
        depth_frame = self.bridge.imgmsg_to_cv2(
            depth_msg, desired_encoding="passthrough"
        )

        depth_image = depth_frame.astype(np.uint16)
        fx = self.K[0, 0]
        fy = self.K[1, 1]
        cx = self.K[0, 2]
        cy = self.K[1, 2]

        if self.seg_human:
            mask_points = []
            mask_human = self.seg_human.get_segmentation(color_frame)
            mask_human = np.asarray(mask_human, dtype=bool)
            if mask_human.size == 0:
                print("Warning: No mask found for human segmentation.")
                return
            print(f"Mask shape: {mask_human.shape}, dtype: {mask_human.dtype}")
            print(mask_human)
            ys, xs = np.where(mask_human > 0.05)

            for v, u in zip(ys, xs):
                Z = depth_image[v, u] / 1000.0
                if Z <= 0:
                    continue
                if Z > 6.0:
                    continue

                X = (u - cx) * Z / fx
                Y = (v - cy) * Z / fy

                mask_points.append([X, Y, Z])
            try:
                human = np.asarray(mask_points)

                human_np = np.asarray(mask_points)
                if human_np.size == 0:
                    print("Warning: No points found for human segmentation.")
                    return
                # 2. Create the Open3D object and assign the points
                human = o3d.geometry.PointCloud()
                human.points = o3d.utility.Vector3dVector(human_np)

                # 3. Now perform the processing
                voxel_size = 0.05
                human = human.voxel_down_sample(voxel_size)

                # Check if points exist before clustering to avoid the crash
                if len(human.points) == 0:
                    print("Warning: Point cloud empty after downsampling.")
                else:
                    eps = 0.05  # 5 centimeters
                    min_points = 5
                    labels = np.array(
                        human.cluster_dbscan(
                            eps=eps, min_points=min_points, print_progress=False
                        )
                    )

                    # 4. Find the Centroid of Each Cluster
                    # Labels of -1 represent noise points that do not belong to any cluster
                    print(labels)
                    max_label = labels.max()
                    print(f"Point cloud broken into {max_label + 1} distinct clusters.")

                    centroids = {}

                    for cluster_id in range(max_label + 1):
                        # Extract indices belonging exclusively to the current cluster
                        cluster_indices = np.where(labels == cluster_id)[0]

                        # Slice the downsampled point cloud to isolate the cluster geometry
                        cluster_pcd = human.select_by_index(cluster_indices)

                        # Method A: Use Open3D's built-in center calculation
                        centroid = cluster_pcd.get_center()

                        # Method B: Direct mathematical mean via NumPy (Alternative)
                        # cluster_points = np.asarray(cluster_pcd.points)
                        # centroid = cluster_points.mean(axis=0)

                        centroids[cluster_id] = centroid
                        print(
                            f"Cluster {cluster_id}: Points = {len(cluster_indices)} | Centroid = {centroid}"
                        )
                    human_array = np.asarray(human.points)
                    human_array = np.column_stack(
                        (human_np, np.ones(human_np.shape[0]))
                    )
                    print("human_array shape:", human_array.shape)
                    # centroid_transformed = self.transformation(centroids)
                    # human = self.transformation(np.asarray(human.points))

                    if len(human.points) > 0:
                        msg_human = self.pc_human_pub.create_pointcloud2(
                            human_array, self.frame
                        )
                        self.pc_human_pub.publisher.publish(msg_human)
                    if len(centroids) > 0:
                        msg_pose = self.po_pub.create_pose_array(centroids, self.frame)
                        self.po_pub.publisher.publish(msg_pose)
            except OverflowError as e:
                print("mask error")

        # start = time.time()
        if self.seg_obstacle:
            mask_points = []

            mask_obstacle, logits = self.seg_obstacle.get_segmentation(color_frame)
            conf_obstacle = mask_obstacle.max(axis=0)

            ys, xs = np.where(conf_obstacle > 0.08)
            # start1 = time.time()
            for v, u in zip(ys, xs):
                Z = depth_image[v, u] / 1000.0
                if Z <= 0:
                    continue
                if Z > 3.0:
                    continue

                X = (u - cx) * Z / fx
                Y = (v - cy) * Z / fy

                mask_points.append([X, Y, Z, 1])
            try:
                mask_points = np.asarray(mask_points)
                obstacle = mask_points[mask_points[:, 3] == 1]
                # obstacle = transformation(obstacle)
                if obstacle.size > 0:
                    msg_obs = self.pc_obstacle_pub.create_pointcloud2(
                        obstacle, self.frame
                    )
                    self.pc_obstacle_pub.publisher.publish(msg_obs)
            except (IndexError, ValueError):
                print("mask error")

        self.color_frame = None
        self.depth_frame = None


def main(args=None):
    print("Starting processing node...")
    track_obs = False
    if track_obs:
        # Clipseg(prompts=["human"])
        seg_human = None
        seg_obstacle = Clipseg(["obstacle"])
    else:
        seg_human = YoloSamCombo(["hammer", ""])
        seg_obstacle = None

    rclpy.init(args=args)
    node = ProcessingNode(seg_human, seg_obstacle)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.pc_human_pub.destroy_node()
        node.pc_obstacle_pub.destroy_node()
        node.destroy_node()
        rclpy.shutdown()
