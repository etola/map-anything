import os
import numpy as np
from colmap_utils import ColmapReconstruction, build_image_id_mapping, compute_image_depthmap
from PIL import Image
import matplotlib.cm as cm
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import cv2

import open3d as o3d
import glob
from typing import List

from threedn_depth_data import ThreednDepthData
from geometric_utility import depthmap_to_world_frame, colorize_heatmap, save_point_cloud, compute_depthmap, uvd_to_world_frame



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
        'scaled_image'          : image_scaled in target_h x target_w,
        'depth_map'             : depth_map in target_h x target_w or None,
        'confidence_map'        : confidence_map in target_h x target_w or None,
        'point_mask'            : point mask in target_h x target_w or None, (stores with points should be considered)
        'fused_depth_map'       : fused depth map in target_h x target_w or None,
        'prior_depth_map'       : prior depth map in target_h x target_w or None,
        'consistency_map'       : consistency map of the estimated depth map with the partner images map in target_h x target_w or None,
        'depth_range'           : (min_depth, max_depth) tuple for consistent scaling, or None if no valid depths,
        'confidence_range'      : (min_confidence, max_confidence) tuple for consistent scaling, or None if no valid confidences,
        'camera_intrinsics'     : K_scaled,  # scaled intrinsics for target_h x target_w image
        'camera_pose'           : pose_4x4,        # Use 4x4 cam_from_world pose matrix for projecting world points to camera frame
        'original_intrinsics'   : K,       # original intrinsics for original image,
        'h'                     : height of the maps,
        'w'                     : width of the maps

    """
    
    def __init__(self, 
                 scene_folder: str,
                 target_h: int,
                 target_w: int,
                 output_folder: str
                 ) -> None:

        """
        Initialize the depth data class.
        
        Args:
            reconstruction: COLMAP reconstruction object
            scene_folder: Path to scene folder containing images
            target_h: Target image height
            target_w: Target image width
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
        self.target_h = target_h
        self.target_w = target_w

        self.output_folder = os.path.join(self.scene_folder, output_folder)
        os.makedirs(self.output_folder, exist_ok=True)

        self.depth_data_folder = os.path.join(self.output_folder, "depth_data")
        os.makedirs(self.depth_data_folder, exist_ok=True)

        self.cloud_folder = os.path.join(self.output_folder, "point_clouds")
        os.makedirs(self.cloud_folder, exist_ok=True)

        self.dmap_folder = os.path.join(self.output_folder, "dmaps")
        os.makedirs(self.dmap_folder, exist_ok=True)

        self.scene_depth_data = {}        # stores all the depth data for each image
        self.active_image_ids = self.reconstruction.get_all_image_ids()
        self.source_to_target_image_id_mapping = {}
        self._lock = threading.Lock()  # Thread lock for safe dictionary access
        self.parallel_executor = ParallelExecutor()  # Parallel execution helper

        # parameters for fusion
        self.fusion_max_partners = 8 # Reduced from 8 to 4 for faster processing
        self.fusion_min_points = 50
        self.fusion_parallax_sample_size = 50
        self.fusion_consistency_threshold = 0.05
        self.fusion_min_consistency_count = 3

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
        scale_x = self.target_w / original_width
        scale_y = self.target_h / original_height
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
            'point_mask': None,
            'prior_depth_map': None,
            'fused_depth_map': None,
            'consistency_map': None,
            'depth_range': None,
            'confidence_range': None,
            'camera_intrinsics': K_scaled,  # Use scaled intrinsics
            'camera_pose': pose_4x4,        # Use 4x4 cam_from_world pose matrix
            'original_intrinsics': K,       # Also save original intrinsics for reference,
            'target_w': self.target_w,
            'target_h': self.target_h,
            'partner_image_ids': self.find_similar_images_for_image(image_id, self.fusion_max_partners)
        }
        with self._lock:
            self.scene_depth_data[image_id] = depth_data

    def export_as_threedn_depth_data(self, image_id: int, max_image_size: int = 800) -> None:
        depth_data = self.get_depth_data(image_id)
        threedn_depth_data = ThreednDepthData()

        camera = self.reconstruction.get_image_camera(image_id)
        export_width, export_height = camera.width, camera.height

        if export_width > max_image_size or export_height > max_image_size:
            scale = max_image_size / max(export_width, export_height)
            export_width = int(export_width * scale)
            export_height = int(export_height * scale)

        print(f"Exporting dmap for image {image_id} with size {export_width}x{export_height}")

        dmap = cv2.resize(depth_data['depth_map'], (export_width, export_height), cv2.INTER_LINEAR)
        cmap = cv2.resize(depth_data['confidence_map'], (export_width, export_height), cv2.INTER_LINEAR)

        threedn_depth_data.image_name = depth_data['image_name']
        threedn_depth_data.image_size = (camera.width, camera.height)
        threedn_depth_data.depth_size = (export_width, export_height)
        threedn_depth_data.depth_range = np.min(dmap[dmap>0]), np.max(dmap[dmap>0])

        cam_from_world = depth_data['camera_pose']
        R = cam_from_world[:3, :3]
        t = cam_from_world[:3, 3]
        C = -R.T @ t

        threedn_depth_data.K = depth_data['camera_intrinsics']
        threedn_depth_data.R = R
        threedn_depth_data.C = C
        threedn_depth_data.flags = ThreednDepthData.HAS_DEPTH | ThreednDepthData.HAS_CONF
        threedn_depth_data.depthMap = dmap.flatten().tolist()
        threedn_depth_data.conf = cmap.flatten().tolist()

        partner_ids = depth_data['partner_image_ids'][:4]
        threedn_depth_data.neighbors = partner_ids

        threedn_depth_data.hsize = threedn_depth_data.headersize()

        threedn_depth_data.save(os.path.join(self.output_folder, f"depth{image_id:04d}.dmap"))

    def export_dmaps(self, max_image_size: int = 800, max_workers: int = 4) -> None:
        self.parallel_executor.run_in_parallel_no_return(
            self.export_as_threedn_depth_data,
            self.active_image_ids,
            progress_desc="Exporting dmaps",
            max_workers=max_workers
        )

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
        assert self.target_w == depth_data['target_w']
        assert self.target_h == depth_data['target_h']
        assert self.reconstruction.has_image(image_id)
        return depth_data

    def is_precomputed_depth_data_present(self) -> bool:
        depth_data_files = glob.glob(os.path.join(self.depth_data_folder, "*.npz"))
        return len(depth_data_files) > 0

    def initialize_from_folder(self):
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

        for img_id in self.active_image_ids:
            self.initialize_depth_data(img_id)
            self.initialize_prior_depth_data_from_reference(img_id)

        self.parallel_executor.run_in_parallel_no_return(
            self.initialize_prior_depth_data_from_reference,
            self.active_image_ids,
            progress_desc="Initializing Prior Depth Data from Reference",
            max_workers=4  # Reduce workers to avoid overwhelming system
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
        prior_depth_map, depth_range = compute_image_depthmap(self.reference_reconstruction, ref_image_id, depth_data['camera_intrinsics'], depth_data['camera_pose'], self.target_w, self.target_h, min_track_length=1)
        
        if prior_depth_map is None:
            print(f"Warning: No prior depth map found for image {image_id}")
        depth_data['prior_depth_map'] = prior_depth_map
        depth_data['depth_range'] = depth_range

    def initialize_scaled_image(self, image_id: int) -> None:
        depth_data = self.get_depth_data(image_id)
        if depth_data['scaled_image'] is not None:
            target_w = depth_data['target_w']
            target_h = depth_data['target_h']
            assert depth_data['scaled_image'].shape == (target_h, target_w, 3), f"Scaled image shape {depth_data['scaled_image'].shape} does not match target size {target_h}x{target_w}"
            return
        image_path = os.path.join(self.scene_folder, "images", depth_data['image_name'])
        assert os.path.exists(image_path), f"Image {image_id}: {image_path} does not exist"
        img = Image.open(image_path)
        if img.mode == "RGBA":
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(background, img)
        img = img.convert("RGB")
        # Convert PIL Image to numpy array for consistent indexing
        depth_data['scaled_image'] = np.array(img.resize((depth_data['target_h'], depth_data['target_w']), Image.Resampling.BICUBIC))

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

        if depth_data['point_mask'] is None:
            depth_data['point_mask'] = (depth_map > 0).astype(bool)
        else:
            depth_data['point_mask'] &= (depth_map > 0).astype(bool)

        target_w = depth_data['target_w']
        target_h = depth_data['target_h']
        assert depth_map.shape == (target_h, target_w), f"Depth map shape {depth_map.shape} does not match target size {target_h}x{target_w}"
        assert confidence_map.shape == (target_h, target_w), f"Confidence map shape {confidence_map.shape} does not match target size {target_h}x{target_w}"

    def save_depth_data(self, image_id: int) -> None:
        depth_data = self.get_depth_data(image_id)
        filepath = os.path.join(self.depth_data_folder, f"depth_{image_id:06d}.npz")
        np.savez_compressed(filepath, **depth_data)

    def save_current_state(self, max_workers: int = 4) -> None:
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

    def get_point_cloud(self, image_id: int, use_prior_depth: bool=False, use_fused_depth: bool=False, conf_threshold: float=0.0, consistency_threshold: int=0, get_color: bool=True) -> tuple:
        depth_data = self.get_depth_data(image_id)

        if use_prior_depth:
            depth_map = depth_data['prior_depth_map']
        elif use_fused_depth:
            depth_map = depth_data['fused_depth_map']
        else:
            depth_map = depth_data['depth_map']

        if depth_map is None:
            return None, None, None

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
            return None, None, None

        # Check if we have valid depth values
        if np.max(depth_map_filtered) == 0:
            print(f"Warning: No valid depth values after filtering: Image Id {image_id:06d}")
            return None, None, None
        
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

            best_partners = self.reconstruction.find_similar_images_for_image(
                reference_image_id, 
                min_points=10,  # Lower threshold for more flexibility
            )
            
            valid_partners = [pid for pid in best_partners if pid in self.active_image_ids]
            if len(valid_partners) == 0:
                print(f"No valid partners found for image {reference_image_id}")
                used_as_reference.add(reference_image_id)
                continue

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
            empty = np.zeros((depth_data['target_h'], depth_data['target_w'], 3), dtype=np.uint8)
            depth_rgb = colorize_heatmap(depth_data['depth_map'], data_range=depth_data['depth_range']) if depth_data['depth_map'] is not None else empty
            conf_rgb  = colorize_heatmap(depth_data['confidence_map'], data_range=depth_data['confidence_range']) if depth_data['confidence_map'] is not None else empty
            prior_rgb = colorize_heatmap(depth_data['prior_depth_map'], data_range=depth_data['depth_range']) if depth_data['prior_depth_map'] is not None else empty
            fusion_rgb = colorize_heatmap(depth_data['fused_depth_map'], data_range=depth_data['depth_range']) if depth_data['fused_depth_map'] is not None else empty
            consistency_rgb = colorize_heatmap(depth_data['consistency_map'], data_range=(0, self.fusion_max_partners-1)) if depth_data['consistency_map'] is not None else empty

            # generate a legend image
            legend_image = np.zeros((self.target_h, 50), dtype=np.float32)
            for i in range(self.fusion_max_partners):
                legend_image[i*self.target_h//self.fusion_max_partners:(i+1)*self.target_h//self.fusion_max_partners, :] = i
            legend_rgb = colorize_heatmap(legend_image, data_range=(0, self.fusion_max_partners-1))

            combined = np.concatenate([depth_data['scaled_image'], prior_rgb, depth_rgb, fusion_rgb, conf_rgb, consistency_rgb, legend_rgb], axis=1)
            Image.fromarray(combined).save(os.path.join(self.depth_data_folder, f"data_{image_id:06d}.png"))

    def save_cloud(self, image_id: int, use_prior_depth: bool=False, use_fused_depth: bool=False, consistent_points: bool=False, file_name: str=None) -> None:
        pts, colors, _ = self.get_point_cloud(image_id, use_prior_depth=use_prior_depth, use_fused_depth=use_fused_depth, consistency_threshold=self.fusion_min_consistency_count if consistent_points else 0)
        if pts is None:
            return
        if file_name is None:
            file_name = f"pointcloud_{image_id:06d}.ply"
        pc_path = os.path.join(self.cloud_folder, file_name)
        save_point_cloud(pts, colors, pc_path)

    def find_similar_images_for_image(self, ref_image_id: int, max_partners: int) -> List[int]:
        similar_image_ids = self.reconstruction.find_similar_images_for_image(
            image_id=ref_image_id,
            min_points=self.fusion_min_points
        )
        if len(similar_image_ids) == 0:
            return []
        valid_partners = [pid for pid in similar_image_ids if pid in self.active_image_ids]
        return valid_partners[:max_partners]

    def uvd_to_world_frame(self, image_id: int, uvd_map: np.ndarray) -> np.ndarray:
        """
        Convert uvd map to world frame.
        
        Args:
            image_id: ID of the image
            uvd_map: HxWx3 array containing [u, v, depth] coordinates
            
        Returns:
            xyz_map: HxWx3 array containing [x, y, z] world coordinates (0 for invalid points)
        """
        depth_data = self.get_depth_data(image_id)
        pose = depth_data['camera_pose']  # cam_from_world (4x4)
        intrinsics = depth_data['camera_intrinsics']  # 3x3
        return uvd_to_world_frame(uvd_map, intrinsics, pose)

    def fuse_for_image(self, image_id: int):
        depth_data = self.get_depth_data(image_id)
        pts3d, valid_mask = depthmap_to_world_frame(depth_data['depth_map'], depth_data['camera_intrinsics'], depth_data['camera_pose'])

        valid_mask = valid_mask & depth_data['point_mask']

        # project each valid point onto partner images and check if the projected point's depth is consistent with
        # the partner's depth map.
        accumulated_pts3d = np.zeros_like(pts3d, dtype=np.float32)
        accumulated_pts3d[valid_mask] = pts3d[valid_mask]
        accumulated_valid_mask = (valid_mask).astype(np.uint8)

        partner_image_ids = depth_data['partner_image_ids']
        partner_uvds = {}

        for partner_id in partner_image_ids:
            partner_data = self.get_depth_data(partner_id)
            partner_uvd = self.compute_consistency_map_depths(pts3d, valid_mask, partner_id)
            partner_valid_mask = valid_mask & (partner_uvd[:,:,2] > 0).astype(np.uint8)
            partner_uvd[partner_valid_mask,2] = 0
            partner_uvds[partner_id] = partner_uvd
            X = uvd_to_world_frame(partner_uvd, partner_data['camera_intrinsics'], partner_data['camera_pose'])
            accumulated_valid_mask += partner_valid_mask.astype(np.uint8)
            accumulated_pts3d[partner_valid_mask>0] += X[partner_valid_mask>0]

        # export 3d points that have accumulated valid mask > fusion_min_consistency_count
        consistency_mask = accumulated_valid_mask > self.fusion_min_consistency_count
        depth_data['consistency_mask'] = accumulated_valid_mask
        if not np.any(consistency_mask):
            return

        # average point by the number of accumulated valid masks per pixel
        fused_pts3d = np.zeros_like(pts3d, dtype=np.float32)
        fused_pts3d[consistency_mask] = accumulated_pts3d[consistency_mask] / accumulated_valid_mask[consistency_mask][:, None]
        fused_dmap = compute_depthmap(fused_pts3d[consistency_mask], depth_data['camera_intrinsics'], depth_data['camera_pose'], target_w=depth_data['target_w'], target_h=depth_data['target_h'])

        depth_data['fused_depth_map'] = fused_dmap


    def apply_fusion(self):

        # for image_id in self.active_image_ids:
        #     depth_data = self.get_depth_data(image_id)
        #     # filter point mask by confidence here if you want
        #     # depth_data['point_mask'] &= (depth_data['confidence_map'] > 0).astype(bool)

        for image_id in self.active_image_ids:
            self.fuse_for_image(image_id)

        # self.parallel_executor.run_in_parallel_no_return(
        #     self.fuse_for_image,
        #     self.active_image_ids,
        #     progress_desc="Fusing depth maps",
        #     max_workers=4
        # )

    def export_fused_point_cloud(self, stepping: int = 1, file_name: str = "fused.ply", use_parallel: bool = True):

        print(f"Exporting fused point cloud with stepping {stepping} and file name {file_name}")

        if use_parallel:
            return self._export_fused_point_cloud_parallel(stepping, file_name)
        else:
            return self._export_fused_point_cloud_sequential(stepping, file_name)
    
    def _export_fused_point_cloud_sequential(self, stepping: int = 1, file_name: str = "fused.ply"):
        pts_list = []
        colors_list = []

        # Initialize exported masks
        for image_id in self.active_image_ids:
            depth_data = self.get_depth_data(image_id)
            depth_data['exported_mask'] = np.zeros_like(depth_data['depth_map'], dtype=bool)

        # Create stepping mask once if needed (vectorized)
        stepping_mask = None
        if stepping > 1:
            stepping_mask = np.zeros_like(depth_data['depth_map'], dtype=bool)
            stepping_mask[::stepping, ::stepping] = True

        for image_id in self.active_image_ids:
            depth_data = self.get_depth_data(image_id)
            if depth_data['fused_depth_map'] is not None:
                dmap = depth_data['fused_depth_map']
                if dmap is None:
                    continue

                exported_mask = depth_data['exported_mask']
                image_pts3d, valid_mask = depthmap_to_world_frame(dmap, depth_data['camera_intrinsics'], depth_data['camera_pose'])
                
                # Remove already exported points
                valid_mask &= ~exported_mask

                # Mark points that have been exported to partner images (before applying stepping)
                for partner_id in depth_data['partner_image_ids']:
                    partner_data = self.get_depth_data(partner_id)
                    uvd = self.compute_consistency_map_depths(image_pts3d, valid_mask, partner_id)
                    uv = uvd[uvd[:,:,2] > 0, :2]
                    if len(uv) > 0:
                        uv = uv.astype(int)
                        partner_data['exported_mask'][uv[:, 1], uv[:, 0]] = True

                # Now apply stepping mask to choose which points to export from this image
                if stepping_mask is not None:
                    valid_mask &= stepping_mask

                # Extract points and colors
                pts3d = image_pts3d[valid_mask]
                if len(pts3d) > 0:
                    current_colors = np.ones((len(pts3d), 3), dtype=np.float32) * 0.5  # Gray color
                    if depth_data['scaled_image'] is not None:
                        simg_array = np.array(depth_data['scaled_image'], dtype=np.float32) / 255.0
                        current_colors = simg_array[valid_mask]
                    
                    pts_list.append(pts3d)
                    colors_list.append(current_colors)

        # Save results
        pts = np.vstack(pts_list)
        colors = np.vstack(colors_list)
        print(f"Exported {len(pts)} points to {file_name}")
        save_point_cloud(pts, colors, os.path.join(self.cloud_folder, file_name)) 
           
    def _export_fused_point_cloud_parallel(self, stepping: int = 1, file_name: str = "fused.ply"):
        from threading import Lock
        
        # Initialize exported masks
        for image_id in self.active_image_ids:
            depth_data = self.get_depth_data(image_id)
            depth_data['exported_mask'] = np.zeros_like(depth_data['depth_map'], dtype=bool)
        
        # Thread-safe data collection
        pts_list = []
        colors_list = []
        data_lock = Lock()
        mask_locks = {img_id: Lock() for img_id in self.active_image_ids}
        
        # Create stepping mask once if needed (vectorized)
        stepping_mask = None
        if stepping > 1:
            stepping_mask = np.zeros((self.target_h, self.target_w), dtype=bool)
            stepping_mask[::stepping, ::stepping] = True
        
        def process_image(image_id: int):
            """Process a single image in parallel"""
            depth_data = self.get_depth_data(image_id)
            if depth_data['fused_depth_map'] is None:
                return None
                
            dmap = depth_data['fused_depth_map']
            if dmap is None:
                return None
                
            # Get exported mask with lock
            with mask_locks[image_id]:
                exported_mask = depth_data['exported_mask'].copy()
                
            image_pts3d, valid_mask = depthmap_to_world_frame(dmap, depth_data['camera_intrinsics'], depth_data['camera_pose'])
            
            # Remove already exported points
            valid_mask &= ~exported_mask
                
            # Process partner images with locks to avoid race conditions (before applying stepping)
            for partner_id in depth_data['partner_image_ids']:
                if partner_id in mask_locks:  # Only process if partner is active
                    partner_data = self.get_depth_data(partner_id)
                    uvd = self.compute_consistency_map_depths(image_pts3d, valid_mask, partner_id)
                    uv = uvd[uvd[:,:,2] > 0, :2]
                    if len(uv) > 0:
                        uv = uv.astype(int)
                        # Thread-safe update of partner's exported_mask
                        with mask_locks[partner_id]:
                            partner_data['exported_mask'][uv[:, 1], uv[:, 0]] = True

            # Now apply stepping mask to choose which points to export from this image
            if stepping_mask is not None:
                valid_mask &= stepping_mask
                            
            # Extract points and colors
            pts3d = image_pts3d[valid_mask]
            if len(pts3d) > 0:
                current_colors = np.ones((len(pts3d), 3), dtype=np.float32) * 0.5
                if depth_data['scaled_image'] is not None:
                    simg_array = np.array(depth_data['scaled_image'], dtype=np.float32) / 255.0
                    current_colors = simg_array[valid_mask]
                
                # Thread-safe collection
                with data_lock:
                    pts_list.append(pts3d)
                    colors_list.append(current_colors)
            
            return len(pts3d) if len(pts3d) > 0 else 0
        
        # Run in parallel
        self.parallel_executor.run_in_parallel_no_return(
            process_image,
            self.active_image_ids,
            progress_desc="Exporting point cloud",
            max_workers=4
        )

        pts = np.vstack(pts_list)
        colors = np.vstack(colors_list)
        print(f"Exported {len(pts)} points to {file_name}")
        save_point_cloud(pts, colors, os.path.join(self.cloud_folder, file_name))


    def compute_consistency_map_depths(self, points_3d: np.ndarray, valid_mask: np.ndarray, partner_id: int) -> np.ndarray:
        """
        Compute consistency map by projecting valid 3D points to partner camera and comparing depths.
        
        Args:
            points_3d: HxWx3 array of 3D points in world coordinates
            valid_mask: HxW boolean mask indicating valid points
            partner_id: ID of partner image for consistency check
            
        Returns:
            consistency_map: HxWx3 array containing [u, v, depth] where u,v are partner image coordinates (all 0.0 = invalid)
        """
        partner_depth_data = self.get_depth_data(partner_id)
        partner_depth_map = partner_depth_data['depth_map']
        if partner_depth_map is None:
            return np.zeros((self.target_h, self.target_w, 3), dtype=np.float32)

        partner_intrinsics = partner_depth_data['camera_intrinsics']
        partner_pose = partner_depth_data['camera_pose']  # cam_from_world for partner
        
        # Initialize consistency map (default to invalid [u, v, depth])
        consistency_map = np.zeros((self.target_h, self.target_w, 3), dtype=np.float32)
        
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
            (partner_pixel_x >= 0) & (partner_pixel_x < self.target_w) &
            (partner_pixel_y >= 0) & (partner_pixel_y < self.target_h)
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
        valid_partner_pixel_x = partner_pixel_x[valid_partner_mask]
        valid_partner_pixel_y = partner_pixel_y[valid_partner_mask]
        
        # Compute relative depth difference
        depth_diff = np.abs(valid_partner_depths - valid_projected_depths) / np.maximum(valid_projected_depths, 1e-6)
        
        # Check consistency threshold
        is_consistent = depth_diff < self.fusion_consistency_threshold
        
        # Set [u, v, depth] for consistent points, [0, 0, 0] for inconsistent
        consistent_u = np.where(is_consistent, valid_partner_pixel_x, 0.0)
        consistent_v = np.where(is_consistent, valid_partner_pixel_y, 0.0)
        consistent_depths = np.where(is_consistent, valid_partner_depths, 0.0)
        
        consistency_map[valid_original_y, valid_original_x, 0] = consistent_u
        consistency_map[valid_original_y, valid_original_x, 1] = consistent_v
        consistency_map[valid_original_y, valid_original_x, 2] = consistent_depths
        
        return consistency_map

    def _save_single_result(self, image_id: int, tag: str = "") -> None:
        """Save all results for a single image."""
        self.save_depth_data(image_id)
        self.save_heatmap(image_id, what_to_save="all")
        self.save_cloud(image_id, file_name=f"{tag}cloud{image_id:06d}.ply")
        self.save_cloud(image_id, consistent_points=True, file_name=f"{tag}consistent_cloud{image_id:06d}.ply")
        self.save_cloud(image_id, use_prior_depth=True, file_name=f"{tag}prior_cloud{image_id:06d}.ply")
        self.save_cloud(image_id, use_fused_depth=True, file_name=f"{tag}fused_cloud{image_id:06d}.ply")

    def save_results(self) -> None:
        self.parallel_executor.run_in_parallel_no_return(
            self._save_single_result,
            self.active_image_ids,
            progress_desc="Saving results",
            max_workers=4
        )

    def transfer_fused_to_prior(self) -> None:
        for image_id in self.active_image_ids:
            depth_data = self.get_depth_data(image_id)
            depth_data['prior_depth_map'] = depth_data['fused_depth_map']

