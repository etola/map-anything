
import numpy as np
from open3d import o3d
from PIL import Image
import matplotlib.cm as cm

def compute_depthmap(points_3d, intrinsics, cam_from_world, target_size):
    """
    Project 3D points to depth map in camera coordinate system.
    
    Args:
        points_3d: (N, 3) array of 3D world coordinates
        intrinsics: (3, 3) intrinsics matrix for a target_size x target_size image
        cam_from_world (4, 4): camera pose: transformation matrix that maps world coordinates to camera coordinates when multiplied
        target_size: target image size for depth map
        
    Returns:
        depth_map: (target_size, target_size) numpy array with depth values
    """
    if len(points_3d) == 0:
        return np.zeros((target_size, target_size), dtype=np.float32)
    
    # Convert 3D points to homogeneous coordinates
    points_3d_homo = np.hstack([points_3d, np.ones((len(points_3d), 1))])
    
    # Transform to camera coordinates
    cam_coords = (cam_from_world[:3, :] @ points_3d_homo.T).T
    
    # Extract depth values (z-coordinates in camera frame)
    depths = cam_coords[:, 2]
    
    # Filter points behind camera
    valid_mask = depths > 0
    if not np.any(valid_mask):
        return np.zeros((target_size, target_size), dtype=np.float32)
    
    cam_coords = cam_coords[valid_mask]
    depths = depths[valid_mask]
    
    # Project to image coordinates
    proj_coords = (intrinsics @ cam_coords.T).T
    proj_coords = proj_coords / proj_coords[:, 2:3]  # Normalize by z
    
    # Convert to pixel coordinates
    pixel_x = proj_coords[:, 0].astype(int)
    pixel_y = proj_coords[:, 1].astype(int)
    
    # Filter points within image bounds
    valid_pixels = (
        (pixel_x >= 0) & (pixel_x < target_size) &
        (pixel_y >= 0) & (pixel_y < target_size)
    )
    
    depth_map = np.zeros((target_size, target_size), dtype=np.float32)
    
    if np.any(valid_pixels):
        pixel_x = pixel_x[valid_pixels]
        pixel_y = pixel_y[valid_pixels]
        pixel_depths = depths[valid_pixels]
        
        # Handle multiple points per pixel by taking the closest depth
        # Replace 0s with infinity so that any actual depth will be smaller
        depth_map_working = np.where(depth_map == 0, np.inf, depth_map)
        # Use minimum.at to handle multiple points mapping to same pixel
        np.minimum.at(depth_map_working, (pixel_y, pixel_x), pixel_depths)
        # Replace any remaining infinities with 0 (shouldn't happen given our data)
        depth_map[:] = np.where(depth_map_working == np.inf, 0, depth_map_working)
    
    return depth_map



def depthmap_to_camera_frame(depthmap: np.ndarray, intrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert depth image to a pointcloud in camera frame using numpy.

    Args:
        depthmap: HxW numpy array
        intrinsics: 3x3 numpy array

    Returns:
        pointmap in camera frame (HxWx3 array), and a mask specifying valid pixels.
    """
    height, width = depthmap.shape
    
    # Create pixel coordinate grids
    x_grid, y_grid = np.meshgrid(np.arange(width), np.arange(height), indexing='xy')
    
    # Extract intrinsics parameters
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1] 
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    
    # Convert to 3D points in camera frame
    depth_z = depthmap
    xx = (x_grid - cx) * depth_z / fx
    yy = (y_grid - cy) * depth_z / fy
    pts3d_cam = np.stack([xx, yy, depth_z], axis=-1)

    # Create valid mask for non-zero depth pixels
    valid_mask = depthmap > 0.0

    return pts3d_cam, valid_mask

def depthmap_to_world_frame(depthmap: np.ndarray, intrinsics: np.ndarray, cam_from_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert depth image to a pointcloud in world frame using numpy.

    Args:
        depthmap: HxW numpy array
        intrinsics: 3x3 numpy array
        cam_from_world: 4x4 numpy array

    Returns:
        pointmap in world frame (HxWx3 array), and a mask specifying valid pixels.
    """
    # Get 3D points in camera frame
    pts3d_cam, valid_mask = depthmap_to_camera_frame(depthmap, intrinsics)
    
    # Convert points from camera frame to world frame
    height, width = depthmap.shape
    
    # Convert to homogeneous coordinates
    pts3d_cam_homo = np.concatenate([
        pts3d_cam, 
        np.ones((height, width, 1))
    ], axis=-1)

    cam_to_world = np.linalg.inv(cam_from_world)

    # Transform to world coordinates: pts_world = cam_to_world @ pts_cam_homo
    # Reshape for matrix multiplication: (H*W, 4) @ (4, 4) -> (H*W, 4)
    pts3d_cam_homo_flat = pts3d_cam_homo.reshape(-1, 4)
    pts3d_world_homo_flat = pts3d_cam_homo_flat @ cam_to_world.T
    
    # Reshape back and take only xyz coordinates
    pts3d_world = pts3d_world_homo_flat[:, :3].reshape(height, width, 3)
    
    return pts3d_world, valid_mask

def colorize_heatmap(data_map, colormap='jet', data_range=None, save_path=None):
    """
    Colorize a data map (depth, confidence, etc.) for visualization and optionally save it.
    
    Args:
        data_map / confidence_map: (H, W) numpy array with data values
        colormap: matplotlib colormap name (default: 'plasma')
        save_path: optional path to save the colorized image
        data_range: optional tuple (min_value, max_value) for consistent scaling across multiple maps        
    Returns:
        colorized_image: (H, W, 3) RGB array of colorized data map
    """
    # Handle case where data map is all zeros
    if np.max(data_map) == 0:
        # Create a black image for zero values
        colorized = np.zeros((data_map.shape[0], data_map.shape[1], 3), dtype=np.uint8)
        if save_path:
            Image.fromarray(colorized).save(save_path)
        return colorized
    
    # Determine data range for normalization
    valid_mask = data_map > 0
    if np.any(valid_mask):
        actual_min_value = np.min(data_map[valid_mask])
        actual_max_value = np.max(data_map[valid_mask])
        
        # Use provided data range if available, otherwise use actual range
        if data_range is not None:
            min_value, max_value = data_range
            # Ensure the provided range encompasses the actual data
            min_value = min(min_value, actual_min_value)
            max_value = max(max_value, actual_max_value)
        else:
            min_value, max_value = actual_min_value, actual_max_value

        # Create normalized data map
        normalized_data = np.zeros_like(data_map)
        if max_value > min_value:
            normalized_data[valid_mask] = np.clip((data_map[valid_mask] - min_value) / (max_value - min_value), 0, 1)
        else:
            normalized_data[valid_mask] = 1.0
    else:
        normalized_data = np.zeros_like(data_map)
    
    # Apply colormap
    cmap = cm.get_cmap(colormap)
    colorized = cmap(normalized_data)

    # Set invalid pixels to black
    colorized[~valid_mask] = [0, 0, 0, 1]
    
    # Convert to 8-bit RGB
    colorized_rgb = (colorized[:, :, :3] * 255).astype(np.uint8)

    return colorized_rgb

def save_point_cloud(pts, colors, save_path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(save_path, pcd)
