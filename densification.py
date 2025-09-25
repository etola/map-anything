import os
import numpy as np
from colmap_utils import ColmapReconstruction, build_image_id_mapping, compute_image_depthmap
from PIL import Image
import matplotlib.cm as cm
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import open3d as o3d

from typing import List

from geometric_utility import depthmap_to_world_frame, colorize_heatmap, save_point_cloud, compute_depthmap


class ParallelExecutor:
    """
    Generic parallel executor for running functions with image IDs in parallel.
    """
    
    def __init__(self, max_workers: int = None):
        """
        Initialize the parallel executor.
        
        Args:
            max_workers: Maximum number of worker threads. If None, uses CPU count.
        """
        self.max_workers = max_workers
    
    def run_in_parallel(self, function, image_id_list: List[int], 
                       progress_desc: str = "Processing", 
                       max_workers: int = None, **kwargs) -> List:
        """
        Execute a function in parallel for each image ID.
        
        Args:
            function: Function to execute. Should accept (image_id, **kwargs) as arguments.
            image_id_list: List of image IDs to process.
            progress_desc: Description for the progress bar.
            max_workers: Override the default max_workers for this execution.
            **kwargs: Additional keyword arguments to pass to the function.
            
        Returns:
            List of results from the function calls (in order of completion).
        """
        if not image_id_list:
            return []
            
        # Determine number of workers
        workers = max_workers or self.max_workers
        if workers is None:
            workers = min(len(image_id_list), os.cpu_count() or 1)
        
        print(f"    {progress_desc}: {len(image_id_list)} items using {workers} workers...")
        
        results = []
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # Submit all tasks
            future_to_img_id = {
                executor.submit(function, img_id, **kwargs): img_id 
                for img_id in image_id_list
            }
            
            # Process completed tasks with progress bar
            with tqdm(total=len(image_id_list), desc=progress_desc, unit="item") as pbar:
                for future in as_completed(future_to_img_id):
                    img_id = future_to_img_id[future]
                    try:
                        result = future.result()
                        results.append(result)
                        pbar.update(1)
                    except Exception as exc:
                        print(f'Processing item {img_id} generated an exception: {exc}')
                        results.append(None)  # Add None for failed items
                        pbar.update(1)
        
        return results
    
    def run_in_parallel_no_return(self, function, image_id_list: List[int], 
                                 progress_desc: str = "Processing", 
                                 max_workers: int = None, **kwargs) -> None:
        """
        Execute a function in parallel for each image ID without collecting results.
        More memory efficient when you don't need the return values.
        
        Args:
            function: Function to execute. Should accept (image_id, **kwargs) as arguments.
            image_id_list: List of image IDs to process.
            progress_desc: Description for the progress bar.
            max_workers: Override the default max_workers for this execution.
            **kwargs: Additional keyword arguments to pass to the function.
        """
        if not image_id_list:
            return
            
        # Determine number of workers
        workers = max_workers or self.max_workers
        if workers is None:
            workers = min(len(image_id_list), os.cpu_count() or 1)
        
        print(f"    {progress_desc}: {len(image_id_list)} items using {workers} workers...")
        
        import time
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # Submit all tasks
            future_to_img_id = {
                executor.submit(function, img_id, **kwargs): img_id 
                for img_id in image_id_list
            }
            
            print(f"    All {len(image_id_list)} tasks submitted, waiting for completion...")
            
            # Process completed tasks with progress bar
            with tqdm(total=len(image_id_list), desc=progress_desc, unit="item", 
                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
                for future in as_completed(future_to_img_id):
                    img_id = future_to_img_id[future]
                    try:
                        future.result()  # Don't store the result
                        pbar.update(1)
                    except Exception as exc:
                        print(f'Processing item {img_id} generated an exception: {exc}')
                        pbar.update(1)
        
        elapsed = time.time() - start_time
        print(f"    Completed {progress_desc} in {elapsed:.2f} seconds ({elapsed/len(image_id_list):.2f}s per item)")

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
        'consistency_map'       : consistency map of the estimated depth map with the partner images map in target_size x target_size or None,
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
        self._lock = threading.Lock()  # Thread lock for safe dictionary access
        self.parallel_executor = ParallelExecutor()  # Parallel execution helper

        # parameters for fusion
        self.fusion_max_partners = 4  # Reduced from 8 to 4 for faster processing
        self.fusion_min_points = 50
        self.fusion_parallax_sample_size = 50
        self.fusion_consistency_threshold = 0.05
        self.fusion_min_consistency_count = 2


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
            'consistency_map': None,
            'depth_range': None,
            'confidence_range': None,
            'camera_intrinsics': K_scaled,  # Use scaled intrinsics
            'camera_pose': pose_4x4,        # Use 4x4 cam_from_world pose matrix
            'original_intrinsics': K,       # Also save original intrinsics for reference,
            'target_size': self.target_size
        }
        with self._lock:
            self.scene_depth_data[image_id] = depth_data

    def get_depth_data(self, image_id: int) -> dict:
        assert image_id in self.scene_depth_data, f"Image {image_id} not found in scene depth data"
        return self.scene_depth_data[image_id]

    def get_active_image_ids(self) -> list:
        return self.active_image_ids

    def _load_depth_data_file(self, depth_data_file: str) -> dict:
        data = np.load(depth_data_file, allow_pickle=True, )
        image_id = int(data['image_id'])
        self.initialize_depth_data(image_id)
        depth_data = self.get_depth_data(image_id)
        for key in list(data.keys()):
            if data[key].ndim == 0:
                depth_data[key] = data[key].item()
            else:
                depth_data[key] = data[key]
        assert self.target_size == depth_data['target_size']
        assert self.reconstruction.has_image(image_id)
        return depth_data

    def initialize_from_folder(self):
        import glob
        self.active_image_ids = []
        depth_data_files = glob.glob(os.path.join(self.depth_data_folder, "*.npz"))
        print(f"Loading {len(depth_data_files)} depth data files...")
        for df in tqdm(depth_data_files, desc="Loading depth data", unit="file"):
            depth_data = self._load_depth_data_file(df)
            image_id = depth_data['image_id']
            self.active_image_ids.append(image_id)
        self.active_image_ids.sort()

    def initialize_with_reference(self, reference_reconstruction) -> None:
        print("Initializing depth data for all active images using reference reconstruction...")
        if isinstance(reference_reconstruction, ColmapReconstruction):
            self.reference_reconstruction = reference_reconstruction
        elif isinstance(reference_reconstruction, str):
            self.reference_reconstruction = ColmapReconstruction(reference_reconstruction)
        else:
            raise ValueError("Reference reconstruction must be a ColmapReconstruction object or a string path to a COLMAP reconstruction")

        print("    Building image id mapping...")
        self.source_to_target_image_id_mapping =  build_image_id_mapping(self.reconstruction, self.reference_reconstruction)

        valid_target_image_ids = self.reference_reconstruction.get_image_ids_with_valid_points()
        print(f"    Nbr of images with 3d points in reference reconstruction: {len(valid_target_image_ids)}/{self.reference_reconstruction.get_num_images()}")

        print("    Filtering images with no 3d points in reference reconstruction...")
        # active image ids are the image ids that have a valid mapping from the source reconstruction to the reference reconstruction
        self.active_image_ids = [img_id for img_id in self.reconstruction.get_all_image_ids() if self.source_to_target_image_id_mapping[img_id] is not None and  self.source_to_target_image_id_mapping[img_id] in valid_target_image_ids]
        print(f"    Found {len(self.active_image_ids)}/{self.reconstruction.get_num_images()} active image ids")

        # # missing image ids are the image ids that do not have a valid mapping from the source reconstruction to the reference reconstruction
        # self.missing_image_ids = [img_id for img_id in self.reconstruction.get_all_image_ids() if img_id not in self.active_image_ids]
        # print(f"Missing Image IDs: {self.missing_image_ids}")
        # for img_id in self.missing_image_ids:
        #     print(f"  {img_id}: {self.reconstruction.get_image_name(img_id)}")

        for img_id in self.active_image_ids:
            self.initialize_depth_data(img_id)

        # Option 2: Reduced workers (recommended)
        self.parallel_executor.run_in_parallel_no_return(
            self.initialize_prior_depth_data_from_reference,
            self.active_image_ids,
            progress_desc="Initializing Prior Depth Data from Reference",
            max_workers=2  # Reduce workers to avoid overwhelming system
        )
        
        self.parallel_executor.run_in_parallel_no_return(
            self.initialize_scaled_image,
            self.active_image_ids,
            progress_desc="Loading and Scaling Images"
        )


    def initialize_prior_depth_data_from_reference(self, image_id: int) -> None:
        if self.reference_reconstruction is None:
            return
        depth_data = self.get_depth_data(image_id)
        ref_image_id = self.source_to_target_image_id_mapping[image_id]
        assert ref_image_id is not None, f"No target image id found for image {image_id}"
        
        # This is the expensive operation - 3D point projection
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

    def save(self, max_workers: int = None) -> None:
        """
        Save all depth data files in parallel.
        Args:
            max_workers: Maximum number of worker threads for parallel saving.
        """
        img_ids = list(self.scene_depth_data.keys())
        
        self.parallel_executor.run_in_parallel_no_return(
            self.save_depth_data,
            img_ids,
            progress_desc="Saving depth data",
            max_workers=max_workers
        )

    def load_prior_depth_data(self, image_id: int) -> None:
        filepath = os.path.join(self.depth_data_folder, f"depth_{image_id:06d}.npz")
        self.scene_depth_data[image_id] = np.load(filepath)

    def get_point_cloud(self, image_id: int, use_prior_depth: bool=False, conf_threshold: float=0.0, consistency_threshold: int=0, get_color: bool=True) -> tuple:
        depth_data = self.get_depth_data(image_id)

        if use_prior_depth:
            depth_map = depth_data['prior_depth_map']
        else:
            depth_map = depth_data['depth_map']

        if depth_map is None:
            return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3), {}

        confidence_map = depth_data['confidence_map']  # (H, W)
        consistency_map = depth_data['consistency_map']  # (H, W)
        camera_intrinsics = depth_data['camera_intrinsics']  # (3, 3) - scaled intrinsics
        cam_from_world = depth_data['camera_pose']  # (4, 4) - camera_from_world pose
        scaled_image = depth_data['scaled_image']
        
        # Filter by confidence threshold
        if conf_threshold > 0.0 and confidence_map is not None:
            valid_mask = confidence_map >= conf_threshold
            depth_map_filtered = depth_map.copy()
            depth_map_filtered[~valid_mask] = 0
            # print(f"Filtered by confidence >= {conf_threshold}: {np.sum(valid_mask)}/{np.prod(depth_map.shape)} pixels")
        else:
            depth_map_filtered = depth_map

        # Filter by consistency threshold
        if consistency_threshold > 0 and consistency_map is not None:
            valid_mask = consistency_map >= consistency_threshold
            depth_map_filtered[~valid_mask] = 0
            # print(f"Filtered by consistency >= {consistency_threshold}: {np.sum(valid_mask)}/{np.prod(depth_map_filtered.shape)} pixels")

        if depth_map_filtered is None:
            return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3), {}

        # Check if we have valid depth values
        if np.max(depth_map_filtered) == 0:
            print(f"Warning: No valid depth values after filtering: Image Id {image_id:06d}")
            return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3), {}
        
        # Compute 3D points from depth using the saved camera parameters
        pts3d, valid_mask = depthmap_to_world_frame(depth_map_filtered, camera_intrinsics, cam_from_world)

        if not valid_mask.any():
            print(f"Warning: No valid points found in depth map: Image Id {image_id:06d}")
            return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3), {}

        # Extract valid points
        pts3d = pts3d[valid_mask]

        if get_color:
            colors = np.ones((len(pts3d), 3), dtype=np.float32) * 0.5  # Gray color
            if scaled_image is not None:
                simg_array = np.array(scaled_image, dtype=np.float32) / 255.0  # Shape: (518, 518, 3)
                colors = simg_array[valid_mask]  # Shape: (N_valid_pixels, 3)
        else:
            colors = None

        return pts3d, colors, valid_mask

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

            best_partners = self.reconstruction.find_best_partners_for_image(
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
            consistency_rgb = colorize_heatmap(depth_data['consistency_map'], data_range=(0, self.fusion_min_consistency_count)) if depth_data['consistency_map'] is not None else empty

            combined = np.concatenate([depth_data['scaled_image'], prior_rgb, depth_rgb, conf_rgb, consistency_rgb], axis=1)
            Image.fromarray(combined).save(os.path.join(self.depth_data_folder, f"data_{image_id:06d}.png"))

    def save_cloud(self, image_id: int, use_prior_depth: bool=False, consistent_points: bool=False, file_name: str=None) -> None:
        if self.get_depth_data(image_id)['depth_map'] is None:
            return
        pts, colors, _ = self.get_point_cloud(image_id, use_prior_depth=use_prior_depth, consistency_threshold=self.fusion_min_consistency_count if consistent_points else 0)
        if file_name is None:
            file_name = f"pointcloud_{image_id:06d}.ply"
        pc_path = os.path.join(self.cloud_folder, file_name)
        if len(pts) == 0:
            return
        save_point_cloud(pts, colors, pc_path)

    def find_similar_images_for_image(self, ref_image_id: int) -> List[int]:
        similar_image_ids = self.reconstruction.find_similar_images_for_image(
            image_id=ref_image_id,
            min_points=self.fusion_min_points
        )
        if len(similar_image_ids) == 0:
            return []
        valid_partners = [pid for pid in similar_image_ids if pid in self.active_image_ids]
        return valid_partners

    def compute_consistency_image(self, image_id: int):
        partner_image_ids = self.find_similar_images_for_image(image_id)
        partner_image_ids = partner_image_ids[:self.fusion_max_partners]
        print(f"Computing consistency for image {image_id}: Partner images: {partner_image_ids}")

        depth_data = self.get_depth_data(image_id)
        depth_map = depth_data['depth_map']
        if depth_map is None:
            return

        # Early termination if no partners found
        if len(partner_image_ids) == 0:
            depth_data['consistency_map'] = np.zeros((depth_data['target_size'], depth_data['target_size']), dtype=np.float32)
            return

        consistency_map = np.zeros((depth_data['target_size'], depth_data['target_size']), dtype=np.float32)
        
        points_3d, valid_mask = depthmap_to_world_frame(depth_map, depth_data['camera_intrinsics'], depth_data['camera_pose'])
        for i, partner_id in enumerate(partner_image_ids):
            cmap_partner = self.compute_consistency_map(points_3d, valid_mask, partner_id)
            consistency_map += cmap_partner
            
        depth_data['consistency_map'] = consistency_map
        


    def compute_consistency_map(self, points_3d: np.ndarray, valid_mask: np.ndarray, partner_id: int) -> np.ndarray:
        """
        Compute consistency map by projecting valid 3D points to partner camera and comparing depths.
        
        Args:
            points_3d: HxWx3 array of 3D points in world coordinates
            valid_mask: HxW boolean mask indicating valid points
            partner_id: ID of partner image for consistency check
            
        Returns:
            consistency_map: HxW array indicating consistency (1.0 = consistent, 0.0 = inconsistent)
        """
        partner_depth_data = self.get_depth_data(partner_id)
        partner_depth_map = partner_depth_data['depth_map']
        if partner_depth_map is None:
            return np.zeros((self.target_size, self.target_size), dtype=np.float32)

        partner_intrinsics = partner_depth_data['camera_intrinsics']
        partner_pose = partner_depth_data['camera_pose']  # cam_from_world for partner
        
        # Initialize consistency map (default to inconsistent)
        consistency_map = np.zeros((self.target_size, self.target_size), dtype=np.float32)
        
        # Extract valid 3D points and their original coordinates
        valid_points_3d = points_3d[valid_mask]  # Shape: (N, 3)
        if len(valid_points_3d) == 0:
            return consistency_map
        
        # Get original pixel coordinates of valid points
        valid_coords = np.where(valid_mask)  # (y_coords, x_coords)
        valid_y, valid_x = valid_coords[0], valid_coords[1]
        
        # Transform 3D points to partner camera coordinates
        points_3d_homo = np.hstack([valid_points_3d, np.ones((len(valid_points_3d), 1))])
        cam_coords = (partner_pose @ points_3d_homo.T).T[:, :3]  # (N, 3)
        
        # Filter points behind camera
        valid_depth_mask = cam_coords[:, 2] > 0
        if not np.any(valid_depth_mask):
            return consistency_map
        
        # Keep only points with valid depth
        cam_coords = cam_coords[valid_depth_mask]
        original_y = valid_y[valid_depth_mask]
        original_x = valid_x[valid_depth_mask]
        
        # Project to image coordinates
        proj_coords = (partner_intrinsics @ cam_coords.T).T
        proj_coords = proj_coords / proj_coords[:, 2:3]  # Normalize by depth
        
        # Get pixel coordinates and depths
        partner_pixel_x = proj_coords[:, 0].astype(int)
        partner_pixel_y = proj_coords[:, 1].astype(int)
        projected_depths = cam_coords[:, 2]
        
        # Filter points within image bounds
        in_bounds_mask = (
            (partner_pixel_x >= 0) & (partner_pixel_x < self.target_size) &
            (partner_pixel_y >= 0) & (partner_pixel_y < self.target_size)
        )
        
        if not np.any(in_bounds_mask):
            return consistency_map
        
        # Keep only in-bounds points
        partner_pixel_x = partner_pixel_x[in_bounds_mask]
        partner_pixel_y = partner_pixel_y[in_bounds_mask]
        projected_depths = projected_depths[in_bounds_mask]
        original_y = original_y[in_bounds_mask]
        original_x = original_x[in_bounds_mask]
        
        # Get partner's depth values at projected locations
        partner_depths = partner_depth_map[partner_pixel_y, partner_pixel_x]
        
        # Compare depths where partner has valid measurements
        valid_partner_mask = partner_depths > 0
        if not np.any(valid_partner_mask):
            return consistency_map
        
        # Compute consistency for valid comparisons
        valid_partner_depths = partner_depths[valid_partner_mask]
        valid_projected_depths = projected_depths[valid_partner_mask]
        valid_original_y = original_y[valid_partner_mask]
        valid_original_x = original_x[valid_partner_mask]
        
        # Compute relative depth difference
        depth_diff = np.abs(valid_partner_depths - valid_projected_depths) / np.maximum(valid_projected_depths, 1e-6)
        
        # Mark as consistent if relative difference is below threshold
        is_consistent = depth_diff < self.fusion_consistency_threshold
        
        # Set consistency values in original image coordinates
        consistency_map[valid_original_y, valid_original_x] = is_consistent.astype(np.float32)
        
        return consistency_map

    def compute_consistency(self) -> None:
        self.parallel_executor.run_in_parallel_no_return(
            self.compute_consistency_image,
            self.active_image_ids,
            progress_desc="Computing consistency",
            max_workers=2  # Reduce workers to avoid CPU thrashing
        )


    def _save_single_result(self, image_id: int, tag: str = "") -> None:
        """Save all results for a single image."""
        self.save_depth_data(image_id)
        self.save_heatmap(image_id, what_to_save="all")
        self.save_cloud(image_id, file_name=f"{tag}cloud{image_id:06d}.ply")
        self.save_cloud(image_id, consistent_points=True, file_name=f"{tag}consistent_cloud{image_id:06d}.ply")
        self.save_cloud(image_id, use_prior_depth=True, file_name=f"{tag}prior_cloud{image_id:06d}.ply")

    def save_results(self) -> None:
        self.parallel_executor.run_in_parallel_no_return(
            self._save_single_result,
            self.active_image_ids,
            progress_desc="Saving results",
            max_workers=2
        )
