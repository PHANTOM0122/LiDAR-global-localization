import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

def preprocess_point_cloud(pcd, voxel_size):

    # 3차원 점군 데이터 복셀화
    pcd_down = pcd.voxel_down_sample(voxel_size)
    radius_normal = 2.0 # 2m

    # Point cloud 법선 벡터 추정
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    # FPFH 특징 추출
    search_radius = 8
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=search_radius, max_nn=500))

    return pcd_down, pcd_fpfh

def execute_coarse_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):

    # Feature descriptor distance 임계치 설정
    distance_threshold = 8

    # Coarse global registration using FPFH + RANSAC
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source=source_down,
        target=target_down,
        source_feature=source_fpfh,
        target_feature=target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        ransac_n=3,
        checkers=[o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(2), o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.99))

    return result

def pose_to_quaternion_translation(pose_matrix):
    rotation_matrix = pose_matrix[:3, :3]
    translation_vector = pose_matrix[:3, 3]

    rotation = R.from_matrix(rotation_matrix)
    quaternion = rotation.as_quat()

    return quaternion, translation_vector
