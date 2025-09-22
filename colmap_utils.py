"""
COLMAP Utilities for MASt3R Densification
========================================

This module contains all COLMAP-related functionality including:
- Bounding box computation from sparse reconstruction
- Image pair selection algorithms
- Camera parameter extraction
- 3D point analysis

"""

import numpy as np
import random
import pycolmap
from typing import Optional, Dict, Set, Tuple, List
from tqdm import tqdm


class ColmapReconstruction:
    """
    Wrapper class for COLMAP reconstruction with caching for efficient repeated operations.
    
    This class caches expensive computations like image-to-3D point mappings to make
    repeated calls to pair selection and analysis functions much more efficient.
    """
    
    def __init__(self, reconstruction_path_or_object):
        """
        Initialize with either a path to reconstruction or existing reconstruction object.
        
        Args:
            reconstruction_path_or_object: Either string path to reconstruction directory
                                         or existing pycolmap.Reconstruction object
        """
        if isinstance(reconstruction_path_or_object, str):
            self.reconstruction = pycolmap.Reconstruction(reconstruction_path_or_object)
            print(f"Loaded reconstruction with {len(self.reconstruction.images)} images and {len(self.reconstruction.points3D)} 3D points")
        else:
            self.reconstruction = reconstruction_path_or_object
        
        # Cached mappings (lazy loaded)
        self._image_point3D_ids: Optional[Dict[int, Set[int]]] = None
        self._image_point3D_xy: Optional[Dict[int, Dict[int, np.ndarray]]] = None
        
        # Cached bounding box computation
        self._bbox_cache = {}
    
    def _ensure_image_point_maps(self):
        """Lazily build image-to-point mappings if not already cached."""
        if self._image_point3D_ids is None or self._image_point3D_xy is None:
            self._image_point3D_ids, self._image_point3D_xy = self._build_image_point_maps()
    
    def _build_image_point_maps(self) -> Tuple[Dict[int, Set[int]], Dict[int, Dict[int, np.ndarray]]]:
        """
        Build dictionaries mapping image IDs to their 3D points and 2D coordinates.
        
        Returns:
            image_point3D_ids: dict[image_id] -> set of point3D_ids
            image_point3D_xy: dict[image_id] -> dict[point3D_id] -> 2D coordinates
        """
        image_point3D_ids = {}
        image_point3D_xy = {}
        
        for image in self.reconstruction.images.values():
            image_point3D_xy[image.image_id] = {}
            for point2D in image.points2D:
                if point2D.has_point3D():
                    image_point3D_xy[image.image_id][point2D.point3D_id] = point2D.xy
                    if image_point3D_ids.get(image.image_id, None) is None:
                        image_point3D_ids[image.image_id] = set()
                    image_point3D_ids[image.image_id].add(point2D.point3D_id)
        
        return image_point3D_ids, image_point3D_xy
    
    def compute_feature_coverage(self, shared_points: Set[int], image_id: int) -> Tuple[float, float]:
        """Compute spatial coverage of features in an image."""
        self._ensure_image_point_maps()
        
        if image_id not in self._image_point3D_xy:
            return 0.0, 0.0
        
        # Check if we have valid shared points for this image
        valid_shared_points = []
        for point_id in shared_points:
            if point_id in self._image_point3D_xy[image_id]:
                valid_shared_points.append(point_id)
        
        if len(valid_shared_points) < 2:
            return 0.0, 0.0
        
        image = self.reconstruction.images[image_id]
        camera = self.reconstruction.cameras[image.camera_id]
        
        xs = [self._image_point3D_xy[image_id][point_id][0] for point_id in valid_shared_points]
        ys = [self._image_point3D_xy[image_id][point_id][1] for point_id in valid_shared_points]
        
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        x_coverage = x_range / camera.width
        y_coverage = y_range / camera.height
        
        return x_coverage, y_coverage
    
    def compute_average_parallax(self, shared_points: Set[int], img1_id: int, img2_id: int, sample_size: int = 100) -> float:
        """Compute average parallax angle between two images for shared 3D points."""
        if len(shared_points) == 0:
            return 0.0
            
        parallax_angles = []
        sample_size = min(sample_size, len(shared_points))
        
        if sample_size == 0:
            return 0.0
        
        try:
            for point_id in random.sample(list(shared_points), sample_size):
                if point_id in self.reconstruction.points3D:
                    point = self.reconstruction.points3D[point_id]
                    parallax_angle = compute_parallax_angle(point.xyz, img1_id, img2_id, self)
                    parallax_angles.append(parallax_angle)
        except Exception as e:
            # If any error in parallax computation, return a default value
            return 0.1  # Small positive value to indicate some parallax
        
        if len(parallax_angles) == 0:
            return 0.0
            
        return np.mean(parallax_angles)
    
   
    def _find_best_partner_for_image(self, image_id: int, min_points: int = 100, 
                                   parallax_sample_size: int = 100) -> List[int]:
        """
        Internal function to find the best partners for a given image.
        
        Args:
            image_id: ID of the image to find a partner for
            min_points: Minimum number of shared 3D points required
            parallax_sample_size: Number of points to sample for parallax computation
        
        Returns:
            List of partner image IDs that satisfy parallax requirements, or [-1] if no good match found
        """
        # Ensure image point mappings are built
        self._ensure_image_point_maps()
        
        # Check if the current image has any 3D points
        if self._image_point3D_ids is None or image_id not in self._image_point3D_ids:
            return [-1]
        
        # Find other images that share at least min_points points
        other_images = [other_image for other_image in self.reconstruction.images.values() 
                       if other_image.image_id != image_id]
        match_candidates = []
        
        for other_image in other_images:
            if other_image.image_id not in self._image_point3D_ids:
                continue
            # Get the points that the two images share
            shared_points = self._image_point3D_ids[image_id] & self._image_point3D_ids[other_image.image_id]
            if len(shared_points) > 0:  # Only add if there are shared points
                match_candidates.append(MatchCandidate(other_image.image_id, shared_points))
        
        # Sort by number of shared points
        match_candidates.sort(key=lambda x: len(x.shared_points), reverse=True)
        
        # Skip if no good candidates found
        if len(match_candidates) == 0 or len(match_candidates[0].shared_points) < min_points:
            return [-1]
        
        # Filter out match candidates that don't have enough matches
        match_candidates = [candidate for candidate in match_candidates 
                          if len(candidate.shared_points) >= min_points]

        # Compute feature coverage for each candidate
        for match_candidate in match_candidates:
            x_cov, y_cov = self.compute_feature_coverage(match_candidate.shared_points, image_id)
            match_candidate.x_coverage = x_cov
            match_candidate.y_coverage = y_cov

        # Order match candidates by average xy coverage
        match_candidates.sort(key=lambda x: (x.x_coverage + x.y_coverage) / 2, reverse=True)

        # Compute average parallax angle for each candidate
        for match_candidate in match_candidates:
            match_candidate.avg_parallax_angle = self.compute_average_parallax(
                match_candidate.shared_points, image_id, match_candidate.image_id, 
                parallax_sample_size)

        if len(match_candidates) == 0:
            return [-1]
        
        # Collect all candidates with good parallax
        good_parallax_candidates = []
        for candidate in match_candidates:
            if candidate.avg_parallax_angle > 0.1:
                good_parallax_candidates.append(candidate.image_id)
        
        # If no good parallax candidates, return the first candidate as fallback
        if len(good_parallax_candidates) == 0:
            good_parallax_candidates = [match_candidates[0].image_id]
        
        return good_parallax_candidates
    
    def get_best_pairs(self, min_points: int = 100, parallax_sample_size: int = 100, min_feature_coverage: float = 0.6, pairs_per_image: int = 1) -> Dict[int, List[int]]:
        """
        Find the best matching image pairs for densification.
        Returns a dictionary where each key is an image_id and value is a list of partner image_ids.
        """
        
        # Initialize empty pair map (image ids)
        pairs = {}
        for image in self.reconstruction.images.values():
            pairs[image.image_id] = []
        
        self._ensure_image_point_maps()
        
        # Process each image with progress bar
        image_list = list(self.reconstruction.images.values())
        for image in tqdm(image_list, desc="Selecting image pairs", unit="img"):
            # Find the best partners for this image using the extracted function
            best_partner_ids = self._find_best_partner_for_image(
                image_id=image.image_id,
                min_points=min_points,
                parallax_sample_size=parallax_sample_size
            )
            
            # Filter out -1 (no match found) and take up to pairs_per_image partners
            valid_partners = [pid for pid in best_partner_ids if pid != -1]
            pairs[image.image_id] = valid_partners[:pairs_per_image]

        return pairs
    
    def get_summary(self) -> Dict:
        """Get summary statistics about the reconstruction."""
        total_observations = sum(len(point3D.track.elements) for point3D in self.reconstruction.points3D.values())
        avg_track_length = total_observations / len(self.reconstruction.points3D) if self.reconstruction.points3D else 0
        
        return {
            'num_images': len(self.reconstruction.images),
            'num_points_3d': len(self.reconstruction.points3D),
            'num_cameras': len(self.reconstruction.cameras),
            'total_observations': total_observations,
            'avg_track_length': avg_track_length
        }
    
    def compute_robust_bounding_box(self, min_visibility: int = 3, padding_factor: float = 0.1, verbose: bool = True) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Compute a robust bounding box from COLMAP 3D points with good visibility.
        Results are cached to avoid recomputation for the same parameters.
        
        Args:
            min_visibility: Minimum number of images a point must be visible in
            padding_factor: Additional padding as fraction of bounding box size
        
        Returns:
            bbox_min, bbox_max: 3D coordinates of bounding box corners
        """
        # Create cache key from parameters
        cache_key = (min_visibility, padding_factor)
        
        # Return cached result if available
        if cache_key in self._bbox_cache:
            bbox_min, bbox_max = self._bbox_cache[cache_key]
            if bbox_min is not None and bbox_max is not None and verbose:
                print(f"Using cached bounding box (min_visibility={min_visibility}, padding={padding_factor})")
                print(f"  Min: [{bbox_min[0]:.3f}, {bbox_min[1]:.3f}, {bbox_min[2]:.3f}]")
                print(f"  Max: [{bbox_max[0]:.3f}, {bbox_max[1]:.3f}, {bbox_max[2]:.3f}]")
            return bbox_min, bbox_max
        
        robust_points = []
        
        # Collect 3D points with sufficient visibility
        for point_id, point3D in self.reconstruction.points3D.items():
            if len(point3D.track.elements) >= min_visibility:
                robust_points.append(point3D.xyz)
        
        if len(robust_points) == 0:
            if verbose:
                print(f"Warning: No points found with visibility >= {min_visibility}")
            # Fallback to all points
            robust_points = [point3D.xyz for point3D in self.reconstruction.points3D.values()]
        
        if len(robust_points) == 0:
            if verbose:
                print("Warning: No 3D points found in reconstruction")
            # Cache the failure result
            self._bbox_cache[cache_key] = (None, None)
            return None, None
        
        robust_points = np.array(robust_points)
        
        # Compute percentile-based bounding box to be robust to outliers
        bbox_min = np.percentile(robust_points, 5, axis=0)   # 5th percentile
        bbox_max = np.percentile(robust_points, 95, axis=0)  # 95th percentile
        
        # Add padding
        bbox_size = bbox_max - bbox_min
        padding = bbox_size * padding_factor
        bbox_min -= padding
        bbox_max += padding
        
        # Cache the result
        self._bbox_cache[cache_key] = (bbox_min.copy(), bbox_max.copy())
        
        if verbose:
            print(f"Computed robust bounding box from {len(robust_points)} points (min_visibility={min_visibility})")
            print(f"  Min: [{bbox_min[0]:.3f}, {bbox_min[1]:.3f}, {bbox_min[2]:.3f}]")
            print(f"  Max: [{bbox_max[0]:.3f}, {bbox_max[1]:.3f}, {bbox_max[2]:.3f}]")
            print(f"  Size: [{bbox_size[0]:.3f}, {bbox_size[1]:.3f}, {bbox_size[2]:.3f}]")
        
        return bbox_min, bbox_max
    

    # Accessor methods for clean API
    def has_image(self, image_id: int) -> bool:
        """Check if an image ID exists in the reconstruction."""
        return image_id in self.reconstruction.images
    
    def get_image(self, image_id: int):
        """Get image object by ID."""
        if not self.has_image(image_id):
            raise KeyError(f"Image ID {image_id} not found in reconstruction")
        return self.reconstruction.images[image_id]
    
    def get_image_name(self, image_id: int) -> str:
        """Get image filename by ID."""
        return self.get_image(image_id).name
    
    def get_image_camera(self, image_id: int):
        """Get camera object for an image."""
        image = self.get_image(image_id)
        return self.reconstruction.cameras[image.camera_id]
    
    def get_image_cam_from_world(self, image_id: int):
        """Get camera pose (cam_from_world) for an image."""
        return self.get_image(image_id).cam_from_world
    
    def get_all_image_ids(self) -> List[int]:
        """Get list of all image IDs in the reconstruction."""
        return list(self.reconstruction.images.keys())
    
    def get_num_images(self) -> int:
        """Get total number of images."""
        return len(self.reconstruction.images)
    
    def get_camera_projection_matrix(self, image_id: int) -> np.ndarray:
        """Get camera projection matrix (K @ [R|t])."""
        camera = self.get_image_camera(image_id)
        K = camera.calibration_matrix()
        cam_from_world = self.get_image_cam_from_world(image_id)
        return K @ cam_from_world.matrix()
    
    def get_camera_center(self, image_id: int) -> np.ndarray:
        """Get camera center in world coordinates."""
        cam = self.get_image_cam_from_world(image_id)
        R = cam.rotation.matrix()
        t = cam.translation
        return -R.T @ t
    
    def compute_baseline(self, img1_id: int, img2_id: int) -> float:
        """Compute baseline distance between two cameras."""
        C1 = self.get_camera_center(img1_id)
        C2 = self.get_camera_center(img2_id)
        return np.linalg.norm(C1 - C2)
    
    def get_camera_calibration_matrix(self, image_id: int) -> np.ndarray:
        """Get camera calibration matrix K for an image."""
        return self.get_image_camera(image_id).calibration_matrix()
    
    def get_camera_distortion_params(self, image_id: int):
        """Extract camera distortion parameters as a dictionary and array."""
        camera = self.get_image_camera(image_id)
        
        # Parse parameter info
        dist_keys = camera.params_info.split(', ')
        dist_dict = {}
        for i, key in enumerate(dist_keys):
            dist_dict[key] = camera.params[i]
        
        # Create standard distortion coefficient array
        dist_coeffs = np.array([
            dist_dict.get('k1', 0), dist_dict.get('k2', 0), 
            dist_dict.get('p1', 0), dist_dict.get('p2', 0), 
            dist_dict.get('k3', 0), dist_dict.get('k4', 0), 
            dist_dict.get('k5', 0), dist_dict.get('k6', 0)
        ])
        
        return dist_dict, dist_coeffs
    
    def get_visible_3d_points(self, image_id: int, min_track_length: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get 3D points visible in the specified image with optional track length filtering.
        
        Args:
            image_id: COLMAP image ID
            min_track_length: Minimum track length for 3D points (default: 0, no filtering)
            
        Returns:
            points_3d: (N, 3) array of 3D world coordinates
            points_2d: (N, 2) array of 2D image coordinates  
            point_ids: (N,) array of 3D point IDs
        """
        if not self.has_image(image_id):
            raise KeyError(f"Image ID {image_id} not found in reconstruction")
        
        # Ensure mappings are built
        self._ensure_image_point_maps()
        
        # Check if image has any 3D points
        if image_id not in self._image_point3D_ids:
            return np.array([]).reshape(0, 3), np.array([]).reshape(0, 2), np.array([])
        
        # Get all 3D point IDs visible in this image
        visible_point_ids = self._image_point3D_ids[image_id]
        
        if not visible_point_ids:
            return np.array([]).reshape(0, 3), np.array([]).reshape(0, 2), np.array([])
        
        # Filter by track length if specified
        if min_track_length > 0:
            filtered_point_ids = []
            for point_id in visible_point_ids:
                point3d = self.reconstruction.points3D[point_id]
                if len(point3d.track.elements) >= min_track_length:
                    filtered_point_ids.append(point_id)
            visible_point_ids = filtered_point_ids
        
        if not visible_point_ids:
            return np.array([]).reshape(0, 3), np.array([]).reshape(0, 2), np.array([])
        
        # Extract 3D coordinates, 2D projections, and point IDs
        points_3d = []
        points_2d = []
        point_ids = []
        
        for point_id in visible_point_ids:
            point3d = self.reconstruction.points3D[point_id]
            point2d_xy = self._image_point3D_xy[image_id][point_id]
            
            points_3d.append(point3d.xyz)
            points_2d.append(point2d_xy)
            point_ids.append(point_id)
        
        return np.array(points_3d), np.array(points_2d), np.array(point_ids)
    
    def get_3d_points_stats(self, image_id: int, min_track_length: int = 0) -> Dict:
        """
        Get statistics about 3D points visible in an image.
        
        Args:
            image_id: COLMAP image ID
            min_track_length: Minimum track length for filtering
            
        Returns:
            Dictionary with statistics about visible 3D points
        """
        points_3d, points_2d, point_ids = self.get_visible_3d_points(image_id, min_track_length)
        
        if len(points_3d) == 0:
            return {
                'num_points': 0,
                'track_lengths': [],
                'avg_track_length': 0,
                'min_track_length': 0,
                'max_track_length': 0,
                'coverage_x': 0.0,
                'coverage_y': 0.0
            }
        
        # Compute track length statistics
        track_lengths = []
        for point_id in point_ids:
            point3d = self.reconstruction.points3D[point_id]
            track_lengths.append(len(point3d.track.elements))
        
        # Compute spatial coverage
        camera = self.get_image_camera(image_id)
        x_coords = points_2d[:, 0]
        y_coords = points_2d[:, 1]
        
        x_range = (x_coords.max() - x_coords.min()) if len(x_coords) > 1 else 0
        y_range = (y_coords.max() - y_coords.min()) if len(y_coords) > 1 else 0
        
        coverage_x = x_range / camera.width
        coverage_y = y_range / camera.height
        
        return {
            'num_points': len(points_3d),
            'track_lengths': track_lengths,
            'avg_track_length': np.mean(track_lengths),
            'min_track_length': min(track_lengths),
            'max_track_length': max(track_lengths),
            'coverage_x': coverage_x,
            'coverage_y': coverage_y
        }





def compute_parallax_angle(point_3d, img1_id, img2_id, reconstruction: ColmapReconstruction):
    """Compute parallax angle for a 3D point viewed from two cameras."""
    C1 = reconstruction.get_camera_center(img1_id)
    C2 = reconstruction.get_camera_center(img2_id)
    
    u = C1 - point_3d
    v = C2 - point_3d
    
    # Compute angle between viewing rays
    cos_angle = np.clip(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)), -1.0, 1.0)
    return np.arccos(cos_angle)


class MatchCandidate:
    """Helper class for storing image matching candidates with their metrics."""
    def __init__(self, image_id, shared_points):
        self.image_id = image_id
        self.shared_points = shared_points
        self.avg_parallax_angle = 0
        self.x_coverage = 0
        self.y_coverage = 0


def load_reconstruction(reconstruction_path):
    """
    Load COLMAP reconstruction and return ColmapReconstruction wrapper.
    """
    try:
        return ColmapReconstruction(reconstruction_path)
    except Exception as e:
        raise ValueError(f"Failed to load COLMAP reconstruction from {reconstruction_path}: {e}")


def find_image_match(source_reconstruction, target_reconstruction, source_image_id, pose_tolerance_rotation=0.1, pose_tolerance_translation=0.1, verbose=False):
    """
    Find the match of a single image from source reconstruction in target reconstruction.
    Searches for potential matches using multiple criteria and returns the one with the best camera pose match.
    
    Args:
        source_reconstruction: ColmapReconstruction object (source/calibration reconstruction)
        target_reconstruction: ColmapReconstruction object (target/reference reconstruction)
        source_image_id: Image ID from source reconstruction to find match for
        pose_tolerance_rotation: Maximum allowed rotation angle difference (default: 0.1 radians ≈ 5.7 degrees)
        pose_tolerance_translation: Maximum allowed difference in translation vectors (default: 0.1 meters)
        verbose: Whether to print matching information
        
    Returns:
        dict: {
            'target_image_id': target image ID if found (None if no match),
            'image_name': name of the image,
            'match_found': boolean indicating if a match was found,
            'pose_status': status string describing pose comparison,
            'rotation_angle': rotation angle difference in radians (None if no match),
            'translation_diff': translation difference in meters (None if no match),
            'poses_compatible': boolean indicating if poses are within tolerance,
            'match_method': string describing how the match was found
        }
    """
    import os
    
    # Check if source image exists
    if not source_reconstruction.has_image(source_image_id):
        if verbose:
            print(f"Source image ID {source_image_id} not found in source reconstruction")
        return {
            'target_image_id': None,
            'image_name': None,
            'match_found': False,
            'pose_status': "Source image not found",
            'rotation_angle': None,
            'translation_diff': None,
            'poses_compatible': False,
            'match_method': None
        }
    
    # Get image name and extract filename from source reconstruction
    source_img_name = source_reconstruction.get_image_name(source_image_id)
    source_filename = os.path.basename(source_img_name)  # Extract just the filename without path
    
    if verbose:
        print(f"Searching for match for '{source_img_name}' (filename: '{source_filename}', ID: {source_image_id})")
    
    # Get source camera pose for comparison
    source_pose = source_reconstruction.get_image_cam_from_world(source_image_id)
    source_R = source_pose.rotation.matrix()
    source_t = source_pose.translation
    
    # Collect all potential candidate matches
    candidates = []
    
    # Method 1: Exact name match
    # Method 2: Filename match (ignoring path)
    # Method 3: Same image ID
    for target_img_id in target_reconstruction.get_all_image_ids():
        target_img_name = target_reconstruction.get_image_name(target_img_id)
        target_filename = os.path.basename(target_img_name)
        
        match_method = None
        
        # Check for exact name match
        if source_img_name == target_img_name:
            match_method = "exact_name"
        # Check for filename match (ignoring path differences)
        elif source_filename == target_filename:
            match_method = "filename"
        # Check for same image ID
        elif source_image_id == target_img_id:
            match_method = "same_id"
        
        if match_method:
            candidates.append((target_img_id, target_img_name, match_method))
    
    if not candidates:
        if verbose:
            print(f"  No candidates found (no exact name, filename, or ID matches)")
        return {
            'target_image_id': None,
            'image_name': source_img_name,
            'match_found': False,
            'pose_status': "No potential matches found",
            'rotation_angle': None,
            'translation_diff': None,
            'poses_compatible': False,
            'match_method': None
        }
    
    if verbose:
        print(f"  Found {len(candidates)} potential candidates:")
        for target_id, target_name, method in candidates:
            print(f"    {target_name} (ID: {target_id}) - matched by {method}")
    
    # Evaluate pose similarity for all candidates
    best_candidate = None
    best_pose_score = float('inf')  # Lower is better (rotation_angle + normalized_translation_diff)
    best_rotation_angle = None
    best_translation_diff = None
    best_match_method = None
    
    candidate_results = []
    
    for target_img_id, target_img_name, match_method in candidates:
        # Get target camera pose
        target_pose = target_reconstruction.get_image_cam_from_world(target_img_id)
        target_R = target_pose.rotation.matrix()
        target_t = target_pose.translation
        
        # Compare rotation matrices using relative rotation R1^T * R2
        R_rel = source_R.T @ target_R
        trace_R = np.trace(R_rel)
        cos_angle = (trace_R - 1) / 2
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Clamp for numerical stability
        rotation_angle = np.arccos(cos_angle)  # Rotation angle in radians
        
        # Compare translation vectors using L2 norm
        translation_diff = np.linalg.norm(source_t - target_t)
        
        # Combine rotation and translation for overall pose score
        # Normalize translation to roughly same scale as rotation (assuming ~1m translation ≈ 1 radian)
        pose_score = rotation_angle + translation_diff
        
        candidate_results.append({
            'target_id': target_img_id,
            'target_name': target_img_name,
            'match_method': match_method,
            'rotation_angle': rotation_angle,
            'translation_diff': translation_diff,
            'pose_score': pose_score
        })
        
        if verbose:
            rotation_degrees = np.degrees(rotation_angle)
            print(f"    {target_img_name}: rot={rotation_angle:.4f} rad ({rotation_degrees:.1f}°), trans={translation_diff:.4f} m, score={pose_score:.4f}")
        
        # Track the best candidate
        if pose_score < best_pose_score:
            best_pose_score = pose_score
            best_candidate = target_img_id
            best_rotation_angle = rotation_angle
            best_translation_diff = translation_diff
            best_match_method = match_method
    
    # Determine pose status and compatibility for the best candidate
    poses_compatible = (best_rotation_angle < pose_tolerance_rotation and 
                       best_translation_diff < pose_tolerance_translation)
    
    if best_rotation_angle < 1e-6 and best_translation_diff < 1e-6:
        pose_status = "✓ Identical"
    elif poses_compatible:
        pose_status = f"⚠ Close (rot: {best_rotation_angle:.6f} rad, trans: {best_translation_diff:.6f} m)"
    else:
        rotation_degrees = np.degrees(best_rotation_angle)
        pose_status = f"❌ Different (rot: {best_rotation_angle:.3f} rad ({rotation_degrees:.1f}°), trans: {best_translation_diff:.3f} m)"
    
    if verbose:
        best_target_name = next(r['target_name'] for r in candidate_results if r['target_id'] == best_candidate)
        print(f"  → Best match: {best_target_name} (ID: {best_candidate}) via {best_match_method}")
        print(f"  → Pose: {pose_status}")
        print(f"  → Compatible: {'Yes' if poses_compatible else 'No'}")
        
        if not poses_compatible:
            # Show camera centers for debugging
            source_center = source_reconstruction.get_camera_center(source_image_id)
            target_center = target_reconstruction.get_camera_center(best_candidate)
            center_diff = np.linalg.norm(source_center - target_center)
            
            print(f"  → Source camera center: [{source_center[0]:.3f}, {source_center[1]:.3f}, {source_center[2]:.3f}]")
            print(f"  → Target camera center: [{target_center[0]:.3f}, {target_center[1]:.3f}, {target_center[2]:.3f}]")
            print(f"  → Camera center distance: {center_diff:.3f} meters")
    
    return {
        'target_image_id': best_candidate,
        'image_name': source_img_name,
        'match_found': True,
        'pose_status': pose_status,
        'rotation_angle': best_rotation_angle,
        'translation_diff': best_translation_diff,
        'poses_compatible': poses_compatible,
        'match_method': best_match_method
    } 