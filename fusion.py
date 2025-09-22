#!/usr/bin/env python3
"""
Robust Point Cloud Fusion from NPZ Depth Maps

This module implements a consistency-based fusion system that:
1. Loads NPZ files containing depth maps and camera parameters
2. Finds partner images using COLMAP reconstruction
3. Projects 3D points between images and checks depth consistency
4. Generates a robust merged point cloud with color assignment
"""

import os
import numpy as np
import torch
import cv2
from typing import List, Tuple, Dict, Optional
import trimesh
from scipy.spatial.distance import cdist
import argparse
from tqdm import tqdm

# Import COLMAP utilities
from colmap_utils import ColmapReconstruction, find_image_match


class DepthMapFusion:
    """
    Robust point cloud fusion from NPZ depth maps using consistency checking.
    """
    
    def __init__(self, depth_data_folder: str, reconstruction: ColmapReconstruction, scene_folder: str,
                 consistency_threshold: float = 0.05, min_consistency_count: int = 2,
                 max_partners: int = 5, verbose: bool = True):
        """
        Initialize the fusion system.
        
        Args:
            depth_data_folder: Path to folder containing depth data files
            reconstruction: COLMAP reconstruction object
            scene_folder: Path to scene folder containing images
            consistency_threshold: Depth consistency threshold (0.05 = 5%)
            min_consistency_count: Minimum number of consistent views required
            max_partners: Maximum number of partner images to consider
            verbose: Enable verbose logging
        """
        self.depth_data_folder = depth_data_folder
        self.reconstruction = reconstruction
        self.scene_folder = scene_folder
        self.consistency_threshold = consistency_threshold
        self.min_consistency_count = min_consistency_count
        self.max_partners = max_partners
        self.verbose = verbose
        
        # Load all depth data files and organize by image_id
        self.depth_data_files = self._load_depth_data_files()
        self.depth_data_by_id = {}  # Dictionary keyed by image_id
        self.processing_masks = {}
        
        # Load all depth data upfront
        self._load_all_depth_data()
        
        print(f"Initialized fusion with {len(self.depth_data_by_id)} depth data files")
    
    def _load_depth_data_files(self) -> List[str]:
        """Load all depth data files from the folder."""
        import glob
        depth_data_files = glob.glob(os.path.join(self.depth_data_folder, "*.npz"))
        return sorted(depth_data_files)
    
    def _load_all_depth_data(self):
        """Load all depth data and organize by image_id."""
        print("Loading all depth data files...")
        
        for depth_data_file in self.depth_data_files:
            try:
                data = np.load(depth_data_file)
                image_id = int(data['image_id'])
                
                # Extract and preprocess data
                depth_map = data['depth_map']
                confidence_map = data['confidence_map']
                camera_intrinsics = data['camera_intrinsics']
                camera_pose = data['camera_pose']
                image_name = str(data['image_name'])
                
                # Check for prior depth map
                prior_depth_map = None
                if 'prior_depth_map' in data:
                    prior_depth_map = data['prior_depth_map']
                
                # Squeeze extra dimensions if present
                if depth_map.ndim == 4:
                    depth_map = depth_map.squeeze(0).squeeze(-1)
                if confidence_map.ndim == 4:
                    confidence_map = confidence_map.squeeze(0).squeeze(-1)
                if prior_depth_map is not None and prior_depth_map.ndim == 4:
                    prior_depth_map = prior_depth_map.squeeze(0).squeeze(-1)
                
                # Initialize processing mask (don't mark prior pixels as processed yet)
                processing_mask = np.zeros(depth_map.shape, dtype=bool)
                
                # Load corresponding image for color assignment
                image_path = self._find_image_path(image_id)
                image = None
                if image_path and os.path.exists(image_path):
                    image = cv2.imread(image_path)
                    if image is not None:
                        # Resize image to match depth map dimensions
                        image = cv2.resize(image, (depth_map.shape[1], depth_map.shape[0]))
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                depth_data = {
                    'depth_map': depth_map,
                    'confidence_map': confidence_map,
                    'prior_depth_map': prior_depth_map,
                    'camera_intrinsics': camera_intrinsics,
                    'camera_pose': camera_pose,
                    'image_id': image_id,
                    'image_name': image_name,
                    'processing_mask': processing_mask,
                    'image': image,
                    'depth_data_file': depth_data_file
                }
                
                self.depth_data_by_id[image_id] = depth_data
                
                if self.verbose:
                    print(f"Loaded depth data for image {image_id} ({image_name})")
                    
            except Exception as e:
                print(f"Error loading {depth_data_file}: {e}")
                continue
    
    def _get_depth_data_by_id(self, image_id: int) -> Optional[Dict]:
        """Get depth data by image_id."""
        return self.depth_data_by_id.get(image_id, None)
    
    def _find_image_path(self, image_id: int) -> Optional[str]:
        """Find the path to the original image using image_id."""
        try:
            # Get image name from COLMAP reconstruction using image_id
            image_name = self.reconstruction.get_image_name(image_id)
            
            # Construct path relative to scene folder
            image_path = os.path.join(self.scene_folder, "images", image_name)
            
            if os.path.exists(image_path):
                return image_path
            else:
                if self.verbose:
                    print(f"Warning: Image not found at {image_path}")
                return None
                
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not find image path for ID {image_id}: {e}")
            return None
    
    def _find_partner_images(self, reference_image_id: int) -> List[int]:
        """Find partner images for the reference image using COLMAP reconstruction."""
        try:
            # Use the COLMAP reconstruction method to find best partners
            partner_image_ids = self.reconstruction._find_best_partner_for_image(
                image_id=reference_image_id,
                min_points=100,
                parallax_sample_size=100
            )
            
            # Check if no good match was found
            if partner_image_ids == [-1]:
                if self.verbose:
                    print(f"No good partner images found for image {reference_image_id}")
                return []
            
            # Filter to only include images that have depth data and limit to max_partners
            available_partners = []
            for partner_id in partner_image_ids:
                if partner_id in self.depth_data_by_id:
                    available_partners.append(partner_id)
                    if len(available_partners) >= self.max_partners:
                        break
            
            if self.verbose:
                print(f"Found {len(available_partners)} partner images for image {reference_image_id}")
            
            return available_partners
            
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not find partners for image {reference_image_id}: {e}")
            return []
    
    def _project_3d_to_image(self, points_3d: np.ndarray, camera_intrinsics: np.ndarray, 
                            camera_pose: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project 3D points to image coordinates.
        
        Args:
            points_3d: (N, 3) 3D points in world coordinates
            camera_intrinsics: (3, 3) camera intrinsics matrix
            camera_pose: (4, 4) camera-to-world transformation matrix
            
        Returns:
            Tuple of (projected_points, depths) where projected_points are (N, 2) pixel coordinates
        """
        if len(points_3d) == 0:
            return np.array([]).reshape(0, 2), np.array([])
        
        # Convert to homogeneous coordinates
        points_homo = np.hstack([points_3d, np.ones((len(points_3d), 1))])
        
        # Transform to camera coordinates
        world_to_cam = np.linalg.inv(camera_pose)
        points_cam = (world_to_cam @ points_homo.T).T
        
        # Extract depths
        depths = points_cam[:, 2]
        
        # Project to image plane
        points_2d = points_cam[:, :2] / points_cam[:, 2:3]
        
        # Apply intrinsics
        projected_points = (camera_intrinsics[:2, :2] @ points_2d.T + camera_intrinsics[:2, 2:3]).T
        
        return projected_points, depths
    
    def _check_depth_consistency(self, points_3d: np.ndarray, reference_image_id: int, 
                                partner_image_id: int) -> np.ndarray:
        """
        Check depth consistency between reference and partner images.
        
        Args:
            points_3d: (N, 3) 3D points
            reference_image_id: ID of reference image
            partner_image_id: ID of partner image
            
        Returns:
            Boolean array indicating which points are consistent
        """
        if len(points_3d) == 0:
            return np.array([], dtype=bool)
        
        # Get partner data from dictionary
        partner_data = self._get_depth_data_by_id(partner_image_id)
        if partner_data is None:
            return np.zeros(len(points_3d), dtype=bool)
        
        partner_depth = partner_data['depth_map']
        partner_intrinsics = partner_data['camera_intrinsics']
        partner_pose = partner_data['camera_pose']
        
        # Project 3D points to partner image
        projected_points, projected_depths = self._project_3d_to_image(
            points_3d, partner_intrinsics, partner_pose
        )
        
        # Check if projections are within image bounds
        h, w = partner_depth.shape
        valid_proj = ((projected_points[:, 0] >= 0) & (projected_points[:, 0] < w) &
                     (projected_points[:, 1] >= 0) & (projected_points[:, 1] < h) &
                     (projected_depths > 0))
        
        if not np.any(valid_proj):
            return np.zeros(len(points_3d), dtype=bool)
        
        # Sample depths from partner depth map
        valid_indices = np.where(valid_proj)[0]
        sampled_depths = np.zeros(len(valid_indices))
        
        for i, idx in enumerate(valid_indices):
            x, y = projected_points[idx].astype(int)
            sampled_depths[i] = partner_depth[y, x]
        
        # Compute consistency
        depth_ratios = np.abs(projected_depths[valid_proj] - sampled_depths) / projected_depths[valid_proj]
        consistent = depth_ratios < self.consistency_threshold
        
        # Create full consistency array
        consistency_array = np.zeros(len(points_3d), dtype=bool)
        consistency_array[valid_indices] = consistent
        
        return consistency_array
    
    def _generate_points_from_depth(self, image_id: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate 3D points from depth map, excluding already processed pixels.
        
        Args:
            image_id: ID of the image
            
        Returns:
            Tuple of (points_3d, colors, valid_mask)
        """
        data = self._get_depth_data_by_id(image_id)
        if data is None:
            return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3), np.array([])
        
        depth_map = data['depth_map']
        prior_depth_map = data.get('prior_depth_map', None)
        processing_mask = data['processing_mask']
        camera_intrinsics = data['camera_intrinsics']
        camera_pose = data['camera_pose']
        image = data['image']
        
        # Only use predicted depth map for unprocessed pixels
        # Prior depths are handled separately in _generate_points_from_prior_depth
        valid_depth_mask = (depth_map > 0) & (~processing_mask)
        active_depth_map = depth_map
        if self.verbose:
            print(f"Using predicted depth map for {np.sum(valid_depth_mask)} pixels")
        
        if not np.any(valid_depth_mask):
            return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3), np.array([])
        
        # Get valid pixel coordinates
        y_coords, x_coords = np.where(valid_depth_mask)
        depths = active_depth_map[valid_depth_mask]
        
        # Convert to 3D points
        fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
        cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]
        
        # Camera coordinates
        x_cam = (x_coords - cx) * depths / fx
        y_cam = (y_coords - cy) * depths / fy
        z_cam = depths
        
        points_cam = np.column_stack([x_cam, y_cam, z_cam])
        
        # Transform to world coordinates
        points_homo = np.hstack([points_cam, np.ones((len(points_cam), 1))])
        points_3d = (camera_pose @ points_homo.T).T[:, :3]
        
        # Extract colors
        colors = np.ones((len(points_3d), 3), dtype=np.float32) * 0.5  # Default gray
        
        if image is not None:
            colors = image[valid_depth_mask] / 255.0
        
        return points_3d, colors, valid_depth_mask
    
    def _generate_points_from_prior_depth(self, image_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate 3D points from prior depth map if available.
        
        Args:
            image_id: ID of the image
            
        Returns:
            Tuple of (points_3d, colors)
        """
        data = self._get_depth_data_by_id(image_id)
        if data is None:
            return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3)
        
        prior_depth_map = data.get('prior_depth_map', None)
        if prior_depth_map is None or not np.any(prior_depth_map > 0):
            return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3)
        
        camera_intrinsics = data['camera_intrinsics']
        camera_pose = data['camera_pose']
        image = data['image']
        
        # Get valid prior depth pixels
        prior_valid_mask = prior_depth_map > 0
        
        if not np.any(prior_valid_mask):
            return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3)
        
        # Get valid pixel coordinates
        y_coords, x_coords = np.where(prior_valid_mask)
        depths = prior_depth_map[prior_valid_mask]
        
        # Convert to 3D points
        fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
        cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]
        
        # Camera coordinates
        x_cam = (x_coords - cx) * depths / fx
        y_cam = (y_coords - cy) * depths / fy
        z_cam = depths
        
        # Stack to get camera coordinates
        points_cam = np.column_stack([x_cam, y_cam, z_cam])
        
        # Transform to world coordinates
        points_cam_homo = np.column_stack([points_cam, np.ones(len(points_cam))])
        points_3d = (camera_pose @ points_cam_homo.T).T[:, :3]
        
        # Get colors from image
        colors = np.ones((len(points_3d), 3), dtype=np.float32) * 0.5  # Default gray
        if image is not None:
            colors = image[prior_valid_mask] / 255.0
        
        # Mark these pixels as processed
        data['processing_mask'][prior_valid_mask] = True
        if self.verbose:
            print(f"Marked {np.sum(prior_valid_mask)} prior depth pixels as processed")
        
        return points_3d, colors
    
    def _update_processing_masks(self, image_id: int, valid_mask: np.ndarray):
        """Update processing mask for the given image ID."""
        data = self._get_depth_data_by_id(image_id)
        if data is not None:
            data['processing_mask'] |= valid_mask
    
    def fuse_point_clouds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Main fusion function that generates a robust point cloud.
        
        Returns:
            Tuple of (points_3d, colors) - final fused point cloud
        """
        print("Starting robust point cloud fusion...")
        
        all_points = []
        all_colors = []
        
        # Create progress bar
        progress_bar = tqdm(
            self.depth_data_by_id.items(), 
            desc="Fusing point clouds", 
            unit="image",
            disable=False  # Always show progress bar
        )
        
        for i, (image_id, ref_data) in enumerate(progress_bar):
            # Update progress bar description
            progress_bar.set_description(f"Processing {ref_data['image_name']}")
            
            if self.verbose:
                print(f"\nProcessing {i+1}/{len(self.depth_data_by_id)}: {ref_data['image_name']} (ID: {image_id})")
            
            # First, generate points from prior depths if available
            prior_points_3d, prior_colors = self._generate_points_from_prior_depth(image_id)
            if len(prior_points_3d) > 0:
                all_points.append(prior_points_3d)
                all_colors.append(prior_colors)
                if self.verbose:
                    print(f"Added {len(prior_points_3d)} points from prior depths for {ref_data['image_name']}")
            
            # Find partner images
            partner_image_ids = self._find_partner_images(image_id)
            
            if len(partner_image_ids) == 0:
                if self.verbose:
                    print(f"No partner images found for {ref_data['image_name']}")
                continue
            
            # Generate 3D points from remaining unprocessed pixels
            points_3d, colors, valid_mask = self._generate_points_from_depth(image_id)
            
            if len(points_3d) == 0:
                if self.verbose:
                    print(f"No valid points generated from remaining pixels in {ref_data['image_name']}")
                continue
            
            # Check consistency with partner images
            consistency_scores = np.zeros(len(points_3d), dtype=int)
            
            for partner_id in partner_image_ids:
                consistency = self._check_depth_consistency(points_3d, image_id, partner_id)
                consistency_scores += consistency.astype(int)
            
            # Filter points based on consistency
            consistent_mask = consistency_scores >= self.min_consistency_count
            
            if np.any(consistent_mask):
                consistent_points = points_3d[consistent_mask]
                consistent_colors = colors[consistent_mask]
                
                all_points.append(consistent_points)
                all_colors.append(consistent_colors)
                
                # Update processing masks for all partner images
                for partner_id in partner_image_ids:
                    partner_data = self._get_depth_data_by_id(partner_id)
                    if partner_data is not None:
                        # Project consistent points to partner image and mark as processed
                        partner_intrinsics = partner_data['camera_intrinsics']
                        partner_pose = partner_data['camera_pose']
                        
                        projected_points, _ = self._project_3d_to_image(
                            consistent_points, partner_intrinsics, partner_pose
                        )
                        
                        # Mark projected pixels as processed
                        h, w = partner_data['depth_map'].shape
                        valid_proj = ((projected_points[:, 0] >= 0) & (projected_points[:, 0] < w) &
                                     (projected_points[:, 1] >= 0) & (projected_points[:, 1] < h))
                        
                        if np.any(valid_proj):
                            proj_coords = projected_points[valid_proj].astype(int)
                            partner_data['processing_mask'][proj_coords[:, 1], proj_coords[:, 0]] = True
                
                if self.verbose:
                    print(f"Added {len(consistent_points)} consistent points "
                          f"(consistency: {np.mean(consistency_scores):.2f})")
            else:
                if self.verbose:
                    print(f"No consistent points found for {ref_data['image_name']}")
        
        # Close progress bar
        progress_bar.close()
        
        # Combine all point clouds
        if all_points:
            final_points = np.vstack(all_points)
            final_colors = np.vstack(all_colors)
            
            print(f"\nFusion complete: {len(final_points)} points from {len(all_points)} images")
            return final_points, final_colors
        else:
            print("No points generated from fusion")
            return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3)


def main():
    """Main function for running the fusion system."""
    parser = argparse.ArgumentParser(description="Robust Point Cloud Fusion from NPZ Files")
    parser.add_argument("-s", "--scene_folder", required=True, help="Path to scene folder containing images")
    parser.add_argument("-o", "--output", type=str, help="Output PLY file path", default="fused.ply")
    parser.add_argument("-d", "--depth_data", help="Path to folder containing depth data files (default: scene_folder/depth_maps)")
    parser.add_argument("-t", "--consistency_threshold", type=float, default=0.05, 
                       help="Depth consistency threshold (default: 0.05 = 5%%)")
    parser.add_argument("-m", "--min_consistency_count", type=int, default=2,
                       help="Minimum number of consistent views required")
    parser.add_argument("-p", "--max_partners", type=int, default=5,
                       help="Maximum number of partner images to consider")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set default paths based on scene_folder
    if args.depth_data is None:
        args.depth_data = os.path.join(args.scene_folder, "depth_maps")
    colmap_reconstruction = os.path.join(args.scene_folder, "sparse")
    
    # Check if paths exist
    if not os.path.exists(args.depth_data):
        print(f"Error: Depth data folder not found: {args.depth_data}")
        return
    if not os.path.exists(colmap_reconstruction):
        print(f"Error: COLMAP reconstruction not found: {colmap_reconstruction}")
        return
    
    print(f"Using depth data folder: {args.depth_data}")
    print(f"Using COLMAP reconstruction: {colmap_reconstruction}")
    print(f"Using scene folder: {args.scene_folder}")
    
    # Load COLMAP reconstruction
    print("Loading COLMAP reconstruction...")
    reconstruction = ColmapReconstruction(colmap_reconstruction)
    
    # Initialize fusion system
    fusion = DepthMapFusion(
        depth_data_folder=args.depth_data,
        reconstruction=reconstruction,
        scene_folder=args.scene_folder,
        consistency_threshold=args.consistency_threshold,
        min_consistency_count=args.min_consistency_count,
        max_partners=args.max_partners,
        verbose=args.verbose
    )
    
    # Run fusion
    points, colors = fusion.fuse_point_clouds()
    
    if len(points) > 0:
        # Save fused point cloud
        pc = trimesh.PointCloud(vertices=points, colors=(colors * 255).astype(np.uint8))
        out_file = os.path.join(args.scene_folder, args.output)
        pc.export(out_file)
        print(f"Saved fused point cloud to: {out_file}")
    else:
        print("No points to save")


if __name__ == "__main__":
    main()
