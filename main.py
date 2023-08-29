import numpy as np
import open3d as o3d
import pygicp
from utils.registration import *
from utils.visualize import *

if __name__ == '__main__':
    # Load data
    map_pcd = o3d.io.read_point_cloud('map.pcd')
    scan_data = np.load('hdl_400/150.npy')
    scan_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(scan_data[:, :3]))

    # Normal estimation
    map_pcd.estimate_normals()
    scan_pcd.estimate_normals()

    voxel_size = 0.1 # 0.1m voxel
    T = np.identity(4) # 초기 relative transformation matrix (global pose)

    # Visualize initial state of data
    print('Visualize LiDAR points before registration!')
    draw_registration_result(scan_pcd, map_pcd, T)

    # Voxelize 3D point clouds
    scan_pcd_down, scan_pcd_fpfh = preprocess_point_cloud(scan_pcd, 0.5)
    map_pcd_down, map_pcd_fpfh = preprocess_point_cloud(map_pcd, 0.5)
    print('Voxelized done!')

    # Coarse global localization (Feature-based)
    T_coarse = execute_coarse_global_registration(scan_pcd_down, map_pcd_down, scan_pcd_fpfh, map_pcd_fpfh, voxel_size)
    print('Coarse estimation done!')
    draw_registration_result(scan_pcd, map_pcd, T_coarse.transformation)

    # Local pose estimation for refinement using pygICP
    coarse_pcd = scan_pcd.transform(T_coarse.transformation)
    source = np.array(coarse_pcd.points).reshape(-1,3)
    target = np.array(map_pcd.points).reshape(-1,3)
    T_refine = pygicp.align_points(target, source)
    print('Refine estimation done!')

    # Print estimated pose (rotation, translation)
    quaternion, translation_vector = pose_to_quaternion_translation(T_refine)
    print('Rotation:', quaternion)
    print('Translation:', translation_vector)

    # Visualizae aligned LiDAR points
    print('Visualize LiDAR points after registration!')
    draw_registration_result(scan_pcd, map_pcd, T_refine)
