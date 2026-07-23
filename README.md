The `pointcloud_pub` package is a ROS 2 Python package designed to process synchronized color and depth camera feeds to segment specific objects and generate 3D spatial data. By leveraging advanced Vision-Language Models (VLMs) and object detection architectures, this package extracts isolated point clouds for target objects, computes their 3D cluster centroids, and separates the remaining environment into a distinct point cloud.

### Key Features

*   **Multimodal Segmentation Integration**: Supports multiple segmentation backends including CLIPSeg (`CIDAS/clipseg-rd64-refined`), a YOLO-World and MobileSAM combination, and YOLOE-26 (`yoloe-26s-seg.pt`).
*   **Time Synchronized Processing**: Utilizes `message_filters.ApproximateTimeSynchronizer` to align incoming color and depth image frames along with camera intrinsic data.
*   **3D Point Cloud Generation**: Converts 2D segmentation masks and depth images into 3D coordinate space using Open3D.
*   **Object Clustering and Pose Estimation**: Applies DBSCAN clustering to the target object's point cloud to identify distinct physical items and calculates their exact 3D centroid poses.
*   **Environment Separation**: Filters out the targeted objects and their bounding boxes to publish a clean point cloud of the surrounding environment.

---

### Prerequisites

Ensure you have a working ROS 2 installation with `ament_python` support. The following Python libraries are required to run the perception and processing pipelines:

*   `rclpy`, `sensor_msgs`, `std_msgs`, `geometry_msgs`
*   `cv_bridge` and `opencv-python` (cv2)
*   `numpy`
*   `open3d`
*   `torch` (PyTorch)
*   `transformers` (Hugging Face)
*   `ultralytics` (YOLO)
*   `Pillow` (PIL)
#### recomended pip install

```bash
# 1. Upgrade pip and core build tools first
python3 -m pip install --break-system-packages --upgrade pip setuptools wheel

# 2. Lock setuptools and force NumPy into the 1.x branch
pip install --break-system-packages "setuptools<80" "numpy<2"

# 3. Install the GPU-enabled PyTorch stack (CUDA 12.6)
pip install --break-system-packages torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu126](https://download.pytorch.org/whl/cu126)

# 4. Install remaining pipeline requirements
pip install --break-system-packages opencv-python open3d transformers ultralytics Pillow matplotlib scipy ftfy regex

# 5. Force install CLIP from source without using a broken cache
pip install --break-system-packages --no-cache-dir git+[https://github.com/ultralytics/CLIP.git](https://github.com/ultralytics/CLIP.git)
```
---

### Installation & Setup

1.  **Clone the repository** into your ROS 2 workspace's `src` directory.
2.  **Download the Required Models**: Before running the package, you must download the necessary model weights. A provided script allows you to download the CLIPSeg model locally:
    ```bash
    python3 test/down_model.py
    ```
    *Note: If using the YOLO+SAM combination, ensure that `yolov8s-worldv2.pt` and `mobile_sam.pt` are correctly placed in the `models/` directory as referenced in `vlm.py`*.
3.  **Build the package** from the root of your workspace using `colcon`:
    ```bash
    colcon build --packages-select pointcloud_pub
    ```
4.  **Source the setup file**:
    ```bash
    source install/setup.bash
    ```

---

### Usage
Start the Realsence camera

```bash
ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true
```

The aligened depth is impotant for accuressy to the real world!

To start the main processing node, run the following command:

```bash
ros2 run pointcloud_pub publisher
```
---
### Truble shooting
for the error "ImportError: 
A module that was compiled using NumPy 1.x cannot be run in
NumPy 2.5.1 as it may crash. To support both 1.x and 2.x
versions of NumPy, modules must be compiled with NumPy 2.0.
Some module may need to rebuild instead e.g. with 'pybind11>=2.12'."
```bash
pip install "numpy<2"
```
ModuleNotFoundError: No module named 'clip'
```bash
pip install git+https://github.com/ultralytics/CLIP.git
```
