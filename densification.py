

import os
import numpy as np
from colmap_utils import ColmapReconstruction, build_image_id_mapping, compute_image_depthmap
from PIL import Image
import matplotlib.cm as cm

import open3d as o3d

from typing import List

from geometric_utility import depthmap_to_world_frame, colorize_heatmap, save_point_cloud

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

        self.cloud_folder = os.path.join(self.output_folder, "point_clouds")
        os.makedirs(self.cloud_folder, exist_ok=True)

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
        prior_depth_map, depth_range = compute_image_depthmap(self.reference_reconstruction, ref_image_id, depth_data['camera_intrinsics'], depth_data['camera_pose'], self.target_size, min_track_length=1)
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
        image_path = os.path.join(self.scene_folder, "images", depth_data['image_name'])
        assert os.path.exists(image_path), f"Image {image_id}: {image_path} does not exist"
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
        elif depth_map.ndim == 3: # 1xHxW
            depth_map = depth_map.squeeze(0)
        if confidence_map.ndim == 4: # 1xHxWx1
            confidence_map = confidence_map.squeeze(0).squeeze(-1)
        elif confidence_map.ndim == 3: # 1xHxW
            confidence_map = confidence_map.squeeze(0)

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
            simg_array = np.array(scaled_image, dtype=np.float32) / 255.0  # Shape: (518, 518, 3)
            colors = simg_array[valid_mask_np]  # Shape: (N_valid_pixels, 3)

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

    def save_heatmap(self, image_id: int, what_to_save: str = "all") -> None:
        if what_to_save not in ["depth_map", "confidence_map", "prior_depth_map", "all"]:
            raise ValueError(f"Invalid what_to_save: {what_to_save}")

        depth_data = self.get_depth_data(image_id)

        if what_to_save == "depth_map" and depth_data['depth_map'] is not None:
            rgb = colorize_heatmap(depth_data['depth_map'], data_range=depth_data['depth_range'])
            Image.fromarray(rgb).save(os.path.join(self.depth_data_folder, f"depth_{image_id:06d}.png"))

        elif what_to_save == "confidence_map" and depth_data['confidence_map'] is not None:
            rgb = colorize_heatmap(depth_data['confidence_map'], data_range=depth_data['confidence_range'])
            Image.fromarray(rgb).save(os.path.join(self.depth_data_folder, f"confidence_{image_id:06d}.png"))

        elif what_to_save == "prior_depth_map" and depth_data['prior_depth_map'] is not None:
            rgb = colorize_heatmap(depth_data['prior_depth_map'], data_range=depth_data['depth_range'])
            Image.fromarray(rgb).save(os.path.join(self.depth_data_folder, f"prior_depth_{image_id:06d}.png"))

        elif what_to_save == "all":
            empty = np.zeros((depth_data['target_size'], depth_data['target_size'], 3), dtype=np.uint8)
            depth_rgb = colorize_heatmap(depth_data['depth_map'], data_range=depth_data['depth_range']) if depth_data['depth_map'] is not None else empty
            conf_rgb  = colorize_heatmap(depth_data['confidence_map'], data_range=depth_data['confidence_range']) if depth_data['confidence_map'] is not None else empty
            prior_rgb = colorize_heatmap(depth_data['prior_depth_map'], data_range=depth_data['depth_range']) if depth_data['prior_depth_map'] is not None else empty

            combined = np.concatenate([depth_data['scaled_image'], prior_rgb, depth_rgb, conf_rgb], axis=1)
            Image.fromarray(combined).save(os.path.join(self.depth_data_folder, f"data_{image_id:06d}.png"))

    def save_point_cloud(self, image_id: int, save_path: str = None) -> None:
        pts, colors = self.get_point_cloud(image_id)
        pc_path = save_path if save_path is not None else os.path.join(self.cloud_folder, f"pointcloud_{image_id:06d}.ply")
        save_point_cloud(pts, colors, pc_path)
