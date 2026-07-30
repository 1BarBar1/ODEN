
import numpy as np
import open3d as o3d

class Depth:
  def __init__(self, depth_image,K):
    #going from numpy object to open 3d
    self.K = K
    
    depth_image = depth_image.astype(np.uint16)
    depth_o3d = o3d.geometry.Image(depth_image)


    #intrinsic matrix
    intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(
        width=depth_image.shape[1],
        height=depth_image.shape[0],
        fx = self.K[0,0],
        fy = self.K[1,1],
        cx = self.K[0,2],
        cy = self.K[1,2]
        )
    #transforming depth image to point cloud
    self.pcd = o3d.geometry.PointCloud.create_from_depth_image(
        depth_o3d,
        intrinsic_o3d,
        depth_scale=1000.0,   # Scaling from millimeters to metes
        depth_trunc=3.0      # clip far points (meters)
        )
    
    '''
    R1 = self.pcd.get_rotation_matrix_from_xyz((0,0, np.pi))

    self.pcd.rotate(R1, center=(0,0,0))
    R2 = self.pcd.get_rotation_matrix_from_xyz((0,np.pi,0))

    self.pcd.rotate(R2, center=(0,0,0))
    '''
  def o3dPoints_to_np(self):
    return np.asarray(self.pcd.points)

  def RANSAC(self,distance_threshold=0.05, ransac_n=3, num_iterations=100):
    plane_model, inliers = self.pcd.segment_plane(
        distance_threshold,
        ransac_n,
        num_iterations
        )
    self.pcd = self.pcd.select_by_index(inliers, invert=True)


  def voxel_mapping(self,voxel_size=0.05):

    #creating voxel grid from point cloud
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
        self.pcd,
        voxel_size
        )
    voxels = voxel_grid.get_voxels()

    indices = np.array([v.grid_index for v in voxels])

    return voxel_grid, indices

  def visualize(self):
    o3d.visualization.draw_geometries([self.pcd])
  def rotate(self,R, center=(0, 0, 0)):
    self.pcd = self.pcd.rotate(R, center=(0, 0, 0))