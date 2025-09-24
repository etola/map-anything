

import os
import numpy as np
from colmap_utils import ColmapReconstruction, build_image_id_mapping
from PIL import Image
import matplotlib.cm as cm

from typing import List

class DensificationProblem:
    """
    Densification problem class for storing depth related data for a scene.

    for each image, in depth_data we store:
        'image_id'              : image_id,
        'image_name'            : image_name,
        'scaled_image'          : image_scaled in target_size x target_size,
        'depth_map'             : depth_map in target_size x target_size or None,
        'confidence_map'        : confidence_map in target_size x target_size or None,
        'prior_depth_map'       : prior depth map in target_size x target_size or None,
        'depth_range'           : (min_depth, max_depth) tuple for consistent scaling, or None if no valid depths,
        'confidence_range'      : (min_confidence, max_confidence) tuple for consistent scaling, or None if no valid confidences,
        'camera_intrinsics'     : K_scaled,  # scaled intrinsics for target_size x target_size image
        'camera_pose'           : pose_4x4,        # Use 4x4 cam_from_world pose matrix for projecting world points to camera frame
        'original_intrinsics'   : K,       # original intrinsics for original image,
        'target_size'           : target_size

    """
    
    def __init__(self, 
                 scene_folder: str,
                 target_size: int,
                 output_folder: str
                 ) -> None:

        """
        Initialize the depth data class.
        
        Args:
            reconstruction: COLMAP reconstruction object
            scene_folder: Path to scene folder containing images
            target_size: Target image size
            output_folder: Path to output folder
        """
        self.scene_folder = scene_folder

        sparse_dir = os.path.join(self.scene_folder, "sparse")
        if not os.path.isdir(sparse_dir):
            raise ValueError(f"Sparse directory {sparse_dir} does not exist")
        images_dir = os.path.join(self.scene_folder, "images")
        if not os.path.isdir(images_dir):
            raise ValueError(f"Images directory {images_dir} does not exist")

        self.reconstruction = ColmapReconstruction(sparse_dir)

        self.reference_reconstruction = None
        self.target_size = target_size

        self.output_folder = os.path.join(self.scene_folder, output_folder)
        os.makedirs(self.output_folder, exist_ok=True)
        self.depth_data_folder = os.path.join(self.output_folder, "depth_data")
        os.makedirs(self.depth_data_folder, exist_ok=True)

        self.scene_depth_data = {}        # stores all the depth data for each image
        self.active_image_ids = self.reconstruction.get_all_image_ids()
        self.source_to_target_image_id_mapping = {}

    def clear(self) -> None:
        self.scene_depth_data = {}
        self.active_image_ids = self.reconstruction.get_all_image_ids()
        self.source_to_target_image_id_mapping = {}

    def initialize_depth_data(self, image_id: int) -> None:
        if not self.reconstruction.has_image(image_id):
            raise ValueError(f"Image {image_id} not found in reconstruction")

        # Extract camera intrinsics and scale for target size
        camera = self.reconstruction.get_image_camera(image_id)
        original_width, original_height = camera.width, camera.height
        
        # Get intrinsics matrix and scale it for the target resolution
        K = self.reconstruction.get_camera_calibration_matrix(image_id)
        
        # Scale intrinsics for resized image
        scale_x = self.target_size / original_width
        scale_y = self.target_size / original_height
        K_scaled = K.copy()
        K_scaled[0, :] *= scale_x  # Scale fx and cx
        K_scaled[1, :] *= scale_y  # Scale fy and cy
            
        # Get camera pose (world to camera transformation)
        pose_4x4 = np.eye(4)
        cam_from_world = self.reconstruction.get_image_cam_from_world(image_id)
        pose_4x4[:3, :] = cam_from_world.matrix()  # 3x4 transformation matrix

        depth_data = {
            'image_id': image_id,
            'image_name': self.reconstruction.get_image_name(image_id),
            'scaled_image': None,
            'depth_map': None,
            'confidence_map': None,
            'prior_depth_map': None,
            'depth_range': None,
            'confidence_range': None,
            'camera_intrinsics': K_scaled,  # Use scaled intrinsics
            'camera_pose': pose_4x4,        # Use 4x4 cam_from_world pose matrix
            'original_intrinsics': K,       # Also save original intrinsics for reference,
            'target_size': self.target_size
        }
        self.scene_depth_data[image_id] = depth_data

    def get_depth_data(self, image_id: int) -> dict:
        assert image_id in self.scene_depth_data, f"Image {image_id} not found in scene depth data"
        return self.scene_depth_data[image_id]

    def get_active_image_ids(self) -> list:
        return self.active_image_ids

    def initialize_from_folder(self):
        import glob
        self.active_image_ids = []
        depth_data_files = glob.glob(os.path.join(self.depth_data_folder, "*.npz"))
        for df in depth_data_files:
            data = np.load(df)
            image_id = int(data['image_id'])
            self.scene_depth_data[image_id] = data
            self.active_image_ids.append(image_id)
            assert self.target_size == data['target_size']
            assert self.reconstruction.has_image(image_id)
        self.active_image_ids.sort()

    def initialize_with_reference(self, reference_reconstruction) -> None:

        if isinstance(reference_reconstruction, ColmapReconstruction):
            self.reference_reconstruction = reference_reconstruction
        elif isinstance(reference_reconstruction, str):
            self.reference_reconstruction = ColmapReconstruction(reference_reconstruction)
        else:
            raise ValueError("Reference reconstruction must be a ColmapReconstruction object or a string path to a COLMAP reconstruction")
    
        self.source_to_target_image_id_mapping =  build_image_id_mapping(self.reconstruction, self.reference_reconstruction)
        valid_target_image_ids = self.reference_reconstruction.get_image_ids_with_valid_points()

        # active image ids are the image ids that have a valid mapping from the source reconstruction to the reference reconstruction
        self.active_image_ids = [img_id for img_id in self.reconstruction.get_all_image_ids() if self.source_to_target_image_id_mapping[img_id] is not None and  self.source_to_target_image_id_mapping[img_id] in valid_target_image_ids]
        print(f"Found {len(self.active_image_ids)}/{self.reconstruction.get_num_images()} active image ids")

        # missing image ids are the image ids that do not have a valid mapping from the source reconstruction to the reference reconstruction
        self.missing_image_ids = [img_id for img_id in self.reconstruction.get_all_image_ids() if img_id not in self.active_image_ids]
        print(f"Missing Image IDs: {self.missing_image_ids}")
        for img_id in self.missing_image_ids:
            print(f"  {img_id}: {self.reconstruction.get_image_name(img_id)}")

        for img_id in self.active_image_ids:
            self.initialize_depth_data(img_id)
            self.initialize_prior_depth_data_from_reference(img_id)
            self.initialize_scaled_image(img_id)

    def initialize_prior_depth_data_from_reference(self, image_id: int) -> None:
        if self.reference_reconstruction is None:
            return
        depth_data = self.get_depth_data(image_id)
        ref_image_id = self.source_to_target_image_id_mapping[image_id]
        assert ref_image_id is not None, f"No target image id found for image {image_id}"
        prior_depth_map, depth_range = compute_prior_depthmap(self.reference_reconstruction, ref_image_id, depth_data['camera_intrinsics'], depth_data['camera_pose'], self.target_size, min_track_length=1)
        if prior_depth_map is None:
            print(f"Warning: No prior depth map found for image {image_id}")
        depth_data['prior_depth_map'] = prior_depth_map
        depth_data['depth_range'] = depth_range

    def initialize_scaled_image(self, image_id: int) -> None:
        depth_data = self.get_depth_data(image_id)
        if depth_data['scaled_image'] is not None:
            target_size = depth_data['target_size']
            assert depth_data['scaled_image'].shape == (target_size, target_size, 3), f"Scaled image shape {depth_data['scaled_image'].shape} does not match target size {target_size}"
            return
        image_path = os.path.join(self.scene_folder, depth_data['image_name'])
        img = Image.open(image_path)
        if img.mode == "RGBA":
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(background, img)
        img = img.convert("RGB")
        depth_data['scaled_image'] = img.resize((depth_data['target_size'], depth_data['target_size']), Image.Resampling.BICUBIC)

    def update_depth_data(self, image_id: int, depth_map: np.ndarray, confidence_map: np.ndarray) -> None:
        depth_data = self.get_depth_data(image_id)
        if depth_map.ndim == 4:
            depth_map = depth_map.squeeze(0).squeeze(-1)
        if confidence_map.ndim == 4:
            confidence_map = confidence_map.squeeze(0).squeeze(-1)
        depth_data['depth_map'] = depth_map
        depth_data['confidence_map'] = confidence_map
        depth_data['confidence_range'] = (np.min(confidence_map), np.max(confidence_map))

        target_size = depth_data['target_size']
        assert depth_map.shape == (target_size, target_size), f"Depth map shape {depth_map.shape} does not match target size {target_size}"
        assert confidence_map.shape == (target_size, target_size), f"Confidence map shape {confidence_map.shape} does not match target size {target_size}"

    def save_depth_data(self, image_id: int) -> None:
        depth_data = self.get_depth_data(image_id)
        filepath = os.path.join(self.depth_data_folder, f"depth_{image_id:06d}.npz")
        np.savez_compressed(filepath, **depth_data)

    def save(self) -> None:
        for img_id in self.scene_depth_data:
            self.save_depth_data(img_id)

    def load_prior_depth_data(self, image_id: int) -> None:
        filepath = os.path.join(self.depth_data_folder, f"depth_{image_id:06d}.npz")
        self.scene_depth_data[image_id] = np.load(filepath)

    def get_point_cloud(self, image_id: int, max_points: int = 1000000, conf_threshold: float = 0.0) -> tuple:
        depth_data = self.get_depth_data(image_id)

        depth_map = depth_data['depth_map']  # (H, W)
        confidence_map = depth_data['confidence_map']  # (H, W)
        camera_intrinsics = depth_data['camera_intrinsics']  # (3, 3) - scaled intrinsics
        cam_from_world = depth_data['camera_pose']  # (4, 4) - camera_from_world pose
        image_id = depth_data['image_id']
        image_name = depth_data['image_name']
        scaled_image = depth_data['scaled_image']
        
        # Filter by confidence threshold
        if conf_threshold > 0.0 and confidence_map is not None:
            valid_mask = confidence_map >= conf_threshold
            depth_map_filtered = depth_map.copy()
            depth_map_filtered[~valid_mask] = 0
            print(f"Filtered by confidence >= {conf_threshold}: {np.sum(valid_mask)}/{np.prod(depth_map.shape)} pixels")
        else:
            depth_map_filtered = depth_map

        # Check if we have valid depth values
        if np.max(depth_map_filtered) == 0:
            print("Warning: No valid depth values after filtering")
            return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3), {}
        
        # Compute 3D points from depth using the saved camera parameters (numpy implementation)
        pts3d_np, valid_mask_np = depthmap_to_world_frame(depth_map_filtered, camera_intrinsics, cam_from_world)

        if not valid_mask_np.any():
            print("Warning: No valid points found in depth map")
            return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3), {}

        # Extract valid points
        valid_pts = pts3d_np[valid_mask_np]

        colors = np.ones((len(valid_pts), 3), dtype=np.float32) * 0.5  # Gray color
        if scaled_image is not None:
            simg = scaled_image.reshape(-1, 3) / 255.0
            colors = simg[valid_mask_np]

        # Subsample if max_points is specified
        if max_points is not None and len(valid_pts) > max_points:
            indices = np.random.choice(len(valid_pts), max_points, replace=False)
            valid_pts = valid_pts[indices]
            colors = colors[indices]

        return valid_pts, colors

    def get_batches_geometric(self, batch_size: int) -> List[List[int]]:
        """
        Split image IDs into batches using COLMAP reconstruction quality metrics.
        Each batch consists of a reference image and its best partner images.
        Once images are used in a batch, they cannot be reference images for future batches.
        
        Args:
            reconstruction: ColmapReconstruction object
            image_ids: List of image IDs
            batch_size: Maximum number of images per batch
            
        Returns:
            List of batches, where each batch is a list of image IDs
        """

        # Ensure image point mappings are built
        self.reconstruction._ensure_image_point_maps()

        batches = []
        used_as_reference = set()  # Images that have been used as reference images
        
        # Sort image IDs for consistent processing order
        remaining_candidates = sorted(self.active_image_ids)
        
        while len(used_as_reference) < len(self.active_image_ids):

            reference_image_id = None
            for img_id in remaining_candidates:
                if img_id not in used_as_reference:
                    reference_image_id = img_id
                    break
            
            if reference_image_id is None:
                break

            best_partners = self.reconstruction.find_best_partner_for_image(
                reference_image_id, 
                min_points=50,  # Lower threshold for more flexibility
                parallax_sample_size=50
            )
            
            valid_partners = [pid for pid in best_partners if pid != -1 and pid in self.active_image_ids]
            
            # Start batch with reference image
            batch = [reference_image_id]
            
            # Add best partners up to batch_size
            for partner in valid_partners:
                if len(batch) >= batch_size:
                    break
                if partner not in batch:  # Avoid duplicates
                    batch.append(partner)


            # If batch is too small, fill with any remaining images that haven't been references
            if len(batch) < batch_size:
                for img_id in remaining_candidates:
                    if len(batch) >= batch_size:
                        break
                    if img_id not in batch and img_id not in used_as_reference:
                        batch.append(img_id)
            
            # Mark ALL images in this batch as used (cannot be reference images anymore)
            for img_id in batch:
                used_as_reference.add(img_id)

            batches.append(batch)
            
        all_batched_images = set()
        for batch in batches:
            all_batched_images.update(batch)
        
        remaining_unprocessed = [img_id for img_id in self.active_image_ids if img_id not in all_batched_images]
        
        if remaining_unprocessed:
            batches.append(remaining_unprocessed)
        
        return batches

    def get_batches_sequential(self, batch_size: int) -> List[List[int]]:
        """
        Simple sequential split into batches.
        Args:
            batch_size: Maximum number of images per batch
        Returns:
            List of batches, where each batch is a list of image IDs
        """
        batches = []
        for i in range(0, len(self.active_image_ids), batch_size):
            batch = self.active_image_ids[i:i + batch_size]
            batches.append(batch)
        return batches

    def save_heatmap(self, image_id: int, what_to_save: str = "depth_map") -> None:
        if what_to_save not in ["depth_map", "confidence_map", "prior_depth_map", "all"]:
            raise ValueError(f"Invalid what_to_save: {what_to_save}")

        depth_data = self.get_depth_data(image_id)

        if what_to_save == "depth_map" and depth_data['depth_map'] is not None:
            rgb = colorize_heatmap(depth_data['depth_map'], depth_data['depth_range'])
            Image.fromarray(rgb).save(os.path.join(self.depth_data_folder, f"depth_{image_id:06d}.png"))

        elif what_to_save == "confidence_map" and depth_data['confidence_map'] is not None:
            rgb = colorize_heatmap(depth_data['confidence_map'], depth_data['confidence_range'])
            Image.fromarray(rgb).save(os.path.join(self.depth_data_folder, f"confidence_{image_id:06d}.png"))

        elif what_to_save == "prior_depth_map" and depth_data['prior_depth_map'] is not None:
            rgb = colorize_heatmap(depth_data['prior_depth_map'], depth_data['depth_range'])
            Image.fromarray(rgb).save(os.path.join(self.depth_data_folder, f"prior_depth_{image_id:06d}.png"))

        elif what_to_save == "all":
            empty = np.zeros((depth_data['target_size'], depth_data['target_size'], 3), dtype=np.uint8)
            depth_rgb = colorize_heatmap(depth_data['depth_map'], depth_data['depth_range']) if depth_data['depth_map'] is not None else empty
            conf_rgb  = colorize_heatmap(depth_data['confidence_map'], depth_data['confidence_range']) if depth_data['confidence_map'] is not None else empty
            prior_rgb = colorize_heatmap(depth_data['prior_depth_map'], depth_data['depth_range']) if depth_data['prior_depth_map'] is not None else empty

            combined = np.concatenate([prior_rgb, depth_rgb, conf_rgb], axis=1)
            Image.fromarray(combined).save(os.path.join(self.depth_data_folder, f"data_{image_id:06d}.png"))



def compute_prior_depthmap(reconstruction, image_id, scaled_intrinsics, cam_from_world, target_size, min_track_length=1, verbose=False):
    """
    Compute prior depth map from reference COLMAP reconstruction for a specific image.
    This depth map will be provided to the MapAnything model as prior information via the 'depth_z' key.
    
    Args:
        reference_reconstruction: ColmapReconstruction object containing prior 3D points
        image_id: COLMAP image ID to compute depth map for
        scaled_intrinsics: (3, 3) scaled camera intrinsics matrix
        cam_from_world: (4, 4) camera pose matrix (ie when you multiply a 3D world point by this matrix, you get the 3D point in the camera frame)
        target_size: target image size for depth map
        min_track_length: minimum track length for 3D points to include        
    Returns:
        tuple: (prior_depth, depth_range) where:
            - prior_depth: (target_size, target_size) numpy array with depth values, or None if no points
            - depth_range: (min_depth, max_depth) tuple for consistent scaling, or None if no valid depths
    """
    try:
        # Get visible 3D points from reference reconstruction
        points_3d, _points_2d, _point_ids = reconstruction.get_visible_3d_points(image_id, min_track_length=min_track_length)

        if len(points_3d) == 0:
            if verbose:
                print(f"Warning: No visible 3D points found in reference reconstruction for image {image_id}")
            return None, None
        
        if verbose:
            print(f"Computing prior depth from {len(points_3d)} 3D points for image {image_id}")
        
        # Project 3D points to a depth map in camera frame
        depth_map = compute_depthmap(points_3d, scaled_intrinsics, cam_from_world, target_size)

        # Check if depth map has valid depths
        valid_depths = np.sum(depth_map > 0)
        if valid_depths == 0:
            if verbose:
                print(f"Warning: No valid depths after projection for image {image_id}")
            return None, None

        if verbose:
            print(f"Generated prior depth map with {valid_depths} valid pixels for image {image_id}")
        
        # Get depth range for consistent scaling
        valid_mask = depth_map > 0
        depth_range = None
        if np.any(valid_mask):
            min_depth = np.min(depth_map[valid_mask])
            max_depth = np.max(depth_map[valid_mask])
            depth_range = (min_depth, max_depth)
        
        return depth_map, depth_range
        
    except Exception as e:
        print(f"Error computing prior depth for image {image_id}: {e}")
        return None, None

def compute_depthmap(points_3d, intrinsics, cam_from_world, target_size):
    """
    Project 3D points to depth map in camera coordinate system.
    
    Args:
        points_3d: (N, 3) array of 3D world coordinates
        intrinsics: (3, 3) scaled intrinsics matrix
        cam_from_world (4, 4): camera to world transformation matrix
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

def colorize_heatmap(data_map, colormap='plasma', data_range=None, save_path=None):
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


