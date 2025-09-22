#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Script to generate a point cloud using MapAnything model with COLMAP calibrations and images.

This script takes a scene folder containing:
- scene_folder/images: directory with images
- scene_folder/sparse: COLMAP reconstruction data

It uses the model to predict depth and generate a dense point cloud based on the COLMAP calibrations.
"""

import argparse
import os
import sys

# Add parent directory to path to import mapanything modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import torch
import trimesh
import open3d as o3d
import tempfile
import shutil
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from torchvision import transforms as tvf
from scipy.spatial.distance import cdist

from mapanything.models import MapAnything
from mapanything.utils.geometry import depthmap_to_world_frame
from mapanything.utils.image import rgb
from mapanything.utils.misc import seed_everything
from colmap_utils import ColmapReconstruction, find_image_match
from uniception.models.encoders.image_normalizations import IMAGE_NORMALIZATION_DICT

# Configure CUDA settings
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate point cloud using MapAnything with COLMAP calibrations"
    )
    parser.add_argument(
        "-sf", "--scene_folder",
        type=str,
        required=True,
        help="Scene folder containing 'images' and 'sparse' subdirectories",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default=None,
        help="Output folder for results (default: scene_folder/output/)",
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--memory_efficient_inference",
        action="store_true",
        default=False,
        help="Use memory efficient inference for reconstruction (trades off speed)",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=518,
        help="Resolution for MapAnything model inference (default: 518)",
    )
    parser.add_argument(
        "--conf_threshold",
        type=float,
        default=0.0,
        help="Confidence threshold for depth filtering (default: 0.0)",
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=1000000,
        help="Maximum number of points in output point cloud (default: 1000000)",
    )
    parser.add_argument(
        "--apache",
        action="store_true",
        help="Use Apache 2.0 licensed model (facebook/map-anything-apache)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Number of images to process in each batch to manage memory usage (default: 8)",
    )
    parser.add_argument(
        "--smart_batching",
        action="store_true",
        default=False,
        help="Use COLMAP reconstruction quality for intelligent batch formation (default: True)",
    )
    parser.add_argument(
        "--sequential_batching",
        action="store_true",
        help="Use simple sequential batching instead of smart batching",
    )
    parser.add_argument(
        "--reference_reconstruction",
        type=str,
        default=None,
        help="Path to reference COLMAP reconstruction for prior depth information (default: None)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose output and save colorized prior and predicted depth maps",
    )
    parser.add_argument(
        "--validate_prior_only",
        action="store_true",
        default=False,
        help="Only validate prior depthmap generation without running MapAnything predictions",
    )
    return parser.parse_args()


def load_images_from_colmap(reconstruction, images_dir, target_size=518, data_norm_type=None, image_ids_subset=None):
    """
    Load images based on COLMAP reconstruction and preprocess them.
    
    Args:
        reconstruction: ColmapReconstruction object
        images_dir: Directory containing images
        target_size: Target size for image preprocessing
        data_norm_type: Image normalization type
        image_ids_subset: Optional list of specific image IDs to load (for batching)
        
    Returns:
        tuple: (images_tensor, image_ids, image_paths)
    """
    if image_ids_subset is not None:
        print(f"Loading {len(image_ids_subset)} images from batch...")
        all_image_ids = image_ids_subset
    else:
        print(f"Loading images from COLMAP reconstruction...")
        all_image_ids = reconstruction.get_all_image_ids()
    
    # Get image IDs and paths from COLMAP reconstruction
    image_ids = []
    image_paths = []
    
    for image_id in all_image_ids:
        image_name = reconstruction.get_image_name(image_id)
        image_path = os.path.join(images_dir, image_name)
        
        if not os.path.exists(image_path):
            print(f"Warning: Image {image_path} not found, skipping...")
            continue
            
        image_ids.append(image_id)
        image_paths.append(image_path)
    
    if len(image_paths) == 0:
        raise ValueError(f"No valid images found in {images_dir}")
    
    print(f"Loading {len(image_paths)} images...")
    
    # Set up image normalization
    if data_norm_type is None:
        img_transform = tvf.ToTensor()
    elif data_norm_type in IMAGE_NORMALIZATION_DICT.keys():
        img_norm = IMAGE_NORMALIZATION_DICT[data_norm_type]
        img_transform = tvf.Compose([
            tvf.ToTensor(), 
            tvf.Normalize(mean=img_norm.mean, std=img_norm.std)
        ])
    else:
        raise ValueError(f"Unknown normalization type: {data_norm_type}")
    
    # Load and preprocess images
    images = []
    for image_path in image_paths:
        # Load image
        img = Image.open(image_path)
        
        # Handle alpha channel
        if img.mode == "RGBA":
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(background, img)
        
        # Convert to RGB
        img = img.convert("RGB")
        
        # Resize to target size
        img = img.resize((target_size, target_size), Image.Resampling.BICUBIC)
        
        # Apply transforms
        img_tensor = img_transform(img)
        images.append(img_tensor)
    
    # Stack images
    images_tensor = torch.stack(images)
    
    return images_tensor, image_ids, image_paths


def project_3d_points_to_depth(points_3d, intrinsics, cam_from_world, target_size):
    """
    Project 3D points to depth map in camera coordinate system.
    
    Args:
        points_3d: (N, 3) array of 3D world coordinates
        intrinsics: (3, 3) scaled intrinsics matrix
        cam_from_world: camera transformation matrix (3x4 or 4x4)
        target_size: target image size for depth map
        
    Returns:
        depth_map: (target_size, target_size) numpy array with depth values
    """
    if len(points_3d) == 0:
        return np.zeros((target_size, target_size), dtype=np.float32)
    
    # Convert 3D points to homogeneous coordinates
    points_3d_homo = np.hstack([points_3d, np.ones((len(points_3d), 1))])
    
    # Transform to camera coordinates
    if cam_from_world.shape == (3, 4):
        # 3x4 transformation matrix
        cam_coords = (cam_from_world @ points_3d_homo.T).T
    else:
        # 4x4 transformation matrix
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
        for x, y, d in zip(pixel_x, pixel_y, pixel_depths):
            if depth_map[y, x] == 0 or d < depth_map[y, x]:
                depth_map[y, x] = d
    
    return depth_map


def colorize_heatmap(data_map, colormap='plasma', save_path=None, data_range=None, title_prefix="Heatmap"):
    """
    Colorize a data map (depth, confidence, etc.) for visualization and optionally save it.
    
    Args:
        data_map: (H, W) numpy array with data values
        colormap: matplotlib colormap name (default: 'plasma')
        save_path: optional path to save the colorized image
        data_range: optional tuple (min_value, max_value) for consistent scaling across multiple maps
        title_prefix: prefix for the plot title (default: "Heatmap")
        
    Returns:
        colorized_image: (H, W, 3) RGB array of colorized data map
        actual_data_range: tuple (min_value, max_value) of actual data range used
    """
    # Handle case where data map is all zeros
    if np.max(data_map) == 0:
        # Create a black image for zero values
        colorized = np.zeros((data_map.shape[0], data_map.shape[1], 3), dtype=np.uint8)
        if save_path:
            Image.fromarray(colorized).save(save_path)
        return colorized, (0.0, 0.0)
    
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
            
        actual_range = (min_value, max_value)
    else:
        normalized_data = np.zeros_like(data_map)
        actual_range = (0.0, 0.0)
    
    # Apply colormap
    cmap = cm.get_cmap(colormap)
    colorized = cmap(normalized_data)
    
    # Set invalid pixels to black
    colorized[~valid_mask] = [0, 0, 0, 1]
    
    # Convert to 8-bit RGB
    colorized_rgb = (colorized[:, :, :3] * 255).astype(np.uint8)
    
    # Save if path provided
    if save_path:
        # Create figure with depth information
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Original data map
        im1 = ax1.imshow(data_map, cmap='gray', vmin=actual_range[0], vmax=actual_range[1])
        ax1.set_title(f'{title_prefix} (Raw)\nMin: {actual_range[0]:.2f}, Max: {actual_range[1]:.2f}' if np.any(valid_mask) else f'{title_prefix} (Raw)\nNo valid values')
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        
        # Colorized data map
        ax2.imshow(colorized_rgb)
        ax2.set_title(f'{title_prefix} (Colorized)\n{np.sum(valid_mask)} valid pixels')
        ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved colorized heatmap to: {save_path}")
    
    return colorized_rgb, actual_range


def compute_prior_depthmap(reference_reconstruction, image_id, scaled_intrinsics, pose_matrix, target_size, min_track_length=2, verbose=False, output_folder=None):
    """
    Compute prior depth map from reference COLMAP reconstruction for a specific image.
    This depth map will be provided to the MapAnything model as prior information via the 'depth_z' key.
    
    Args:
        reference_reconstruction: ColmapReconstruction object containing prior 3D points
        image_id: COLMAP image ID to compute depth map for
        scaled_intrinsics: (3, 3) scaled camera intrinsics matrix
        pose_matrix: (3, 4) camera pose matrix (cam_from_world)
        target_size: target image size for depth map
        min_track_length: minimum track length for 3D points to include
        verbose: if True, save colorized depth maps to output folder
        output_folder: folder to save colorized depth maps when verbose=True
        
    Returns:
        tuple: (prior_depth, depth_range) where:
            - prior_depth: (target_size, target_size) numpy array with depth values, or None if no points
            - depth_range: (min_depth, max_depth) tuple for consistent scaling, or None if no valid depths
    """
    try:
        # Get visible 3D points from reference reconstruction
        points_3d, points_2d, point_ids = reference_reconstruction.get_visible_3d_points(
            image_id, min_track_length=min_track_length
        )
        
        if len(points_3d) == 0:
            print(f"Warning: No visible 3D points found in reference reconstruction for image {image_id}")
            return None, None
        
        print(f"Computing prior depth from {len(points_3d)} 3D points for image {image_id}")
        
        # Project 3D points to depth map
        depth_map = project_3d_points_to_depth(
            points_3d, scaled_intrinsics, pose_matrix, target_size
        )
        
        # Check if depth map has valid depths
        valid_depths = np.sum(depth_map > 0)
        if valid_depths == 0:
            print(f"Warning: No valid depths after projection for image {image_id}")
            return None, None
        
        print(f"Generated prior depth map with {valid_depths} valid pixels for image {image_id}")
        
        # Get depth range for consistent scaling
        valid_mask = depth_map > 0
        depth_range = None
        if np.any(valid_mask):
            min_depth = np.min(depth_map[valid_mask])
            max_depth = np.max(depth_map[valid_mask])
            depth_range = (min_depth, max_depth)
        
        # Note: Prior depth maps will be saved after inference with consistent scaling
        if verbose and output_folder is not None:
            print(f"Prior depth computed for image {image_id}, will be saved after inference")
        
        return depth_map, depth_range
        
    except Exception as e:
        print(f"Error computing prior depth for image {image_id}: {e}")
        return None, None


def filter_images_with_reference_points(reconstruction, reference_reconstruction, image_ids, verbose=False):
    """
    Filter images to only include those with reference points.
    
    Args:
        reconstruction: ColmapReconstruction object (calibration)
        reference_reconstruction: ColmapReconstruction object (reference)
        image_ids: List of image IDs to filter
        verbose: If True, print detailed filtering information
        
    Returns:
        tuple: (valid_image_ids, image_name_mapping)
    """
    if reference_reconstruction is None:
        return image_ids, {}
    
    print(f"\n=== Filtering Images with Reference Points ===")
    
    # Build image name mapping using single-image matching
    image_name_mapping = {}
    matched_count = 0
    compatible_count = 0
    
    for img_id in image_ids:
        match_result = find_image_match(
            source_reconstruction=reconstruction,
            target_reconstruction=reference_reconstruction,
            source_image_id=img_id,
            pose_tolerance_rotation=1e-3,  # Very strict tolerance for identical poses
            pose_tolerance_translation=1e-3,  # Very strict tolerance for identical poses
            verbose=False
        )
        
        if match_result['match_found'] and match_result['poses_compatible']:
            image_name_mapping[img_id] = match_result['target_image_id']
            matched_count += 1
            compatible_count += 1
            if verbose:
                print(f"  ✓ {match_result['image_name']}: Compatible match (ID: {img_id} → {match_result['target_image_id']}) via {match_result['match_method']}")
        else:
            image_name_mapping[img_id] = None
            if match_result['match_found']:
                if verbose:
                    print(f"  ❌ {match_result['image_name']}: Pose mismatch (ID: {img_id} → {match_result['target_image_id']}) via {match_result['match_method']} - {match_result['pose_status']}")
            else:
                if verbose:
                    print(f"  ❌ {match_result['image_name'] or f'ID:{img_id}'}: No match found - {match_result['pose_status']}")
    
    print(f"Matching Summary: {compatible_count}/{matched_count}/{len(image_ids)} (compatible/matched/total)")
    
    # Filter to only include images with reference points
    valid_image_ids = []
    
    for img_id in image_ids:
        if img_id in image_name_mapping and image_name_mapping[img_id] is not None:
            # Check if reference points exist for this image
            ref_image_id = image_name_mapping[img_id]
            ref_points_3d, ref_points_2d, ref_point_ids = reference_reconstruction.get_visible_3d_points(
                ref_image_id, min_track_length=2
            )
            
            if len(ref_points_3d) > 0:
                valid_image_ids.append(img_id)
                print(f"  ✓ Including image {reconstruction.get_image_name(img_id)} (ID:{img_id}) - {len(ref_points_3d)} reference points")
            else:
                print(f"  ❌ Excluding image {reconstruction.get_image_name(img_id)} (ID:{img_id}) - no reference points found")
        else:
            print(f"  ❌ Excluding image {reconstruction.get_image_name(img_id)} (ID:{img_id}) - no compatible match in reference reconstruction")
    
    print(f"\nFiltered to {len(valid_image_ids)}/{len(image_ids)} images with reference points")
    return valid_image_ids, image_name_mapping


def run_mapanything_inference(model, images, reconstruction, image_ids, target_size, dtype, memory_efficient_inference=False, reference_reconstruction=None, verbose=False, output_folder=None, image_name_mapping=None):
    """
    Run MapAnything model inference on images with COLMAP camera parameters.
    
    Args:
        model: MapAnything model
        images: Tensor of images [N, 3, H, W]
        reconstruction: ColmapReconstruction object
        image_ids: List of COLMAP image IDs corresponding to the images
        target_size: Target image size used for preprocessing
        dtype: Data type for inference
        memory_efficient_inference: Whether to use memory efficient inference
        reference_reconstruction: Optional ColmapReconstruction object for prior depth information
        verbose: If True, save colorized prior and predicted depth maps to output folder
        output_folder: Output folder for saving visualizations when verbose=True
        
    Returns:
        List of prediction dictionaries
        
    Note:
        When reference_reconstruction is provided, prior depth maps are generated from the 3D points
        and passed to the model via the 'depth_z' key to provide depth priors for better reconstruction.
        The prior depth maps are also saved alongside predicted depth maps when verbose=True.
        Images without reference points are removed from the batch.
    """
    print("Running MapAnything inference with COLMAP camera parameters...")
    
    # Check if reference reconstruction is provided for prior depth information
    if reference_reconstruction is not None:
        print(f"Using reference reconstruction with {reference_reconstruction.get_num_images()} images and {len(reference_reconstruction.reconstruction.points3D)} 3D points for prior depth information")
        print("Note: Prior depth will be provided to the model to improve reconstruction quality")
        
        # Use provided image name mapping or create empty one
        if image_name_mapping is None:
            image_name_mapping = {}
        
    else:
        print("No reference reconstruction provided - running without prior depth information")
        # Create empty mapping when no reference reconstruction
        image_name_mapping = {}
    
    # Keep track of depth ranges for consistent scaling
    all_depth_ranges = []
    
    # Keep track of prior depths for each view
    all_prior_depths = []
    
    # Prepare views for model
    views = []
    for view_idx in range(images.shape[0]):
        image_id = image_ids[view_idx]
        
        # Extract camera intrinsics and scale for target size
        camera = reconstruction.get_image_camera(image_id)
        original_width, original_height = camera.width, camera.height
        
        # Get intrinsics matrix and scale it for the target resolution
        K = reconstruction.get_camera_calibration_matrix(image_id)
        
        # Scale intrinsics for resized image
        scale_x = target_size / original_width
        scale_y = target_size / original_height
        K_scaled = K.copy()
        K_scaled[0, :] *= scale_x  # Scale fx and cx
        K_scaled[1, :] *= scale_y  # Scale fy and cy
        
        # Get camera pose (world to camera transformation)
        cam_from_world = reconstruction.get_image_cam_from_world(image_id)
        pose_matrix = cam_from_world.matrix()  # 3x4 transformation matrix
        
        # Convert to 4x4 homogeneous matrix
        pose_4x4 = np.eye(4)
        pose_4x4[:3, :] = pose_matrix
        
        # Convert from cam_from_world to cam2world (world_from_cam) for MapAnything
        # MapAnything expects OpenCV cam2world convention: camera coordinates -> world coordinates
        pose_4x4 = np.linalg.inv(pose_4x4)
        
        # Compute prior depth map if reference reconstruction is provided
        prior_depth = None
        depth_range = None
        if reference_reconstruction is not None:
            # Since we've already filtered to only include images with valid reference points,
            # we know this image has a match
            ref_image_id = image_name_mapping[image_id]
            img_name = reconstruction.get_image_name(image_id)
            print(f"Computing prior depth for image {img_name} (Cal_ID:{image_id} -> Ref_ID:{ref_image_id})...")
            
            # Since the poses are identical for matching images, use the calibration image's pose matrix
            prior_depth, depth_range = compute_prior_depthmap(
                reference_reconstruction=reference_reconstruction,
                image_id=ref_image_id,  # Use reference reconstruction's image ID
                scaled_intrinsics=K_scaled,
                pose_matrix=pose_matrix,  # Use calibration image's pose matrix (should be same as reference)
                target_size=target_size,
                verbose=verbose,
                output_folder=output_folder
            )
            # Collect depth range for consistent scaling
            if depth_range is not None:
                all_depth_ranges.append(depth_range)
            
            # Store prior depth for this view
            all_prior_depths.append(prior_depth)
            
            # Compute and save point cloud from prior depth map for comparison
            if prior_depth is not None and np.sum(prior_depth > 0) > 0:
                print(f"Computing point cloud from prior depth map for {img_name}...")
                
                # Create a dummy image for color information
                dummy_image = np.ones((target_size, target_size, 3), dtype=np.float32) * 0.5
                
                # Convert depth map to point cloud using ground truth parameters
                # Convert cam_from_world to cam2world for the function
                pose_4x4 = np.eye(4)
                pose_4x4[:3, :] = pose_matrix
                pose_4x4 = np.linalg.inv(pose_4x4)  # Convert cam_from_world to cam2world
                
                prior_points_from_depth, prior_colors_from_depth = compute_pointcloud_from_depthmap_groundtruth(
                    prior_depth, dummy_image, K_scaled, pose_4x4,
                    conf_threshold=0.0, max_points=None
                )
                
                if len(prior_points_from_depth) > 0:
                    print(f"Generated {len(prior_points_from_depth)} points from prior depth map")
                    # Save the point cloud from prior depth map
                    if output_folder is not None and verbose:
                        save_single_pointcloud(prior_points_from_depth, prior_colors_from_depth, image_id, img_name, 
                                             "depth_prior", output_folder)
                        print(f"Saved prior depth point cloud for {img_name}")
                else:
                    print(f"Warning: No points generated from prior depth map for {img_name}")
            else:
                print(f"Warning: No valid prior depth map for {img_name}")
        
        view = {
            "img": images[view_idx][None],  # Add batch dimension
            "data_norm_type": [model.encoder.data_norm_type],
            # Provide COLMAP camera parameters for metric reconstruction
            "intrinsics": torch.tensor(K_scaled, dtype=torch.float32).unsqueeze(0),  # Scaled intrinsics
            "camera_poses": torch.tensor(pose_4x4, dtype=torch.float32).unsqueeze(0),  # Camera-to-world pose (OpenCV convention)
            "is_metric_scale": torch.ones(1, dtype=torch.bool),  # Enable metric scale (COLMAP provides this) - should be bool
        }
        
        # Add prior depth to view if available
        if prior_depth is not None:
            # Convert to torch tensor and ensure proper dtype
            z_depth_tensor = torch.tensor(prior_depth, dtype=torch.float32)
            
            # Make sure the tensor is contiguous and has proper properties
            z_depth_tensor = z_depth_tensor.contiguous()
            
            # Check for any NaN or infinite values and replace them
            if torch.isnan(z_depth_tensor).any():
                print(f"Warning: Found NaN values in prior depth for image {image_id}, setting to 0")
                z_depth_tensor = torch.nan_to_num(z_depth_tensor, nan=0.0)
            
            if torch.isinf(z_depth_tensor).any():
                print(f"Warning: Found infinite values in prior depth for image {image_id}, setting to 0")
                z_depth_tensor = torch.nan_to_num(z_depth_tensor, posinf=0.0, neginf=0.0)
            
            # Ensure all values are finite and non-negative
            z_depth_tensor = torch.clamp(z_depth_tensor, min=0.0, max=1e6)
            
            # The model expects z-depth in shape [1, H, W, 1] under 'depth_z' key
            view["depth_z"] = z_depth_tensor.unsqueeze(0).unsqueeze(-1)
            
            # Print some debug info
            valid_pixels = (z_depth_tensor > 0).sum().item()
            total_pixels = z_depth_tensor.numel()
            print(f"Added prior depth to view for image {image_id}: {valid_pixels}/{total_pixels} valid pixels")
            print(f"  Z-depth range: {z_depth_tensor[z_depth_tensor > 0].min():.3f} - {z_depth_tensor[z_depth_tensor > 0].max():.3f}" if valid_pixels > 0 else "  No valid depths")
        else:
            print(f"No prior depth computed for image {image_id}")
            # Add None for this view since no prior depth was computed
            all_prior_depths.append(None)
        views.append(view)
    
    # Summary of views being passed to model
    views_with_prior = sum(1 for view in views if "depth_z" in view)
    total_views = len(views)
    print(f"Passing {total_views} views to model, {views_with_prior} with prior depth information")
    
    # Run inference
    with torch.amp.autocast("cuda", dtype=dtype):
        predictions = model.infer(
            views, memory_efficient_inference=memory_efficient_inference
        )
    
    # Save depth maps if verbose mode is enabled
    if verbose and output_folder is not None:
        print("Saving depth maps...")
        depth_maps_folder = os.path.join(output_folder, "depth_maps")
        os.makedirs(depth_maps_folder, exist_ok=True)
        
        # Determine consistent depth range for all maps (prior + predicted)
        min_depths = []
        max_depths = []
        
        # Include prior depth ranges if available
        if all_depth_ranges:
            min_depths.extend([r[0] for r in all_depth_ranges])
            max_depths.extend([r[1] for r in all_depth_ranges])
        
        # Include predicted depth ranges
        for pred_idx, pred in enumerate(predictions):
            pred_depth = pred["depth_z"][0].squeeze(-1).cpu().numpy()  # (H, W)
            pred_valid_mask = pred_depth > 0
            if np.any(pred_valid_mask):
                pred_min = np.min(pred_depth[pred_valid_mask])
                pred_max = np.max(pred_depth[pred_valid_mask])
                min_depths.append(pred_min)
                max_depths.append(pred_max)
        
        # Determine consistent depth range
        consistent_depth_range = None
        if min_depths and max_depths:
            consistent_depth_range = (min(min_depths), max(max_depths))
            print(f"Using consistent depth range for all visualizations: [{consistent_depth_range[0]:.2f}, {consistent_depth_range[1]:.2f}]m")
        
        # Save prior depth maps with consistent scaling if they exist
        if reference_reconstruction is not None:
            for view_idx, image_id in enumerate(image_ids):
                # Since we've already filtered to only include images with valid reference points,
                # we know this image has a match
                ref_image_id = image_name_mapping[image_id]
                img_name = reconstruction.get_image_name(image_id)
                
                # Get the calibration image's pose matrix (same as used in inference)
                camera = reconstruction.get_image_camera(image_id)
                K = reconstruction.get_camera_calibration_matrix(image_id)
                
                # Scale intrinsics for resized image
                scale_x = target_size / camera.width
                scale_y = target_size / camera.height
                K_scaled = K.copy()
                K_scaled[0, :] *= scale_x
                K_scaled[1, :] *= scale_y
                
                # Get camera pose (world to camera transformation)
                cam_from_world = reconstruction.get_image_cam_from_world(image_id)
                pose_matrix = cam_from_world.matrix()  # 3x4 transformation matrix
                
                # Re-compute prior depth for consistent scaling using same logic as inference
                prior_depth, _ = compute_prior_depthmap(
                    reference_reconstruction=reference_reconstruction,
                    image_id=ref_image_id,  # Use reference reconstruction's image ID
                    scaled_intrinsics=K_scaled,
                    pose_matrix=pose_matrix,  # Use calibration image's pose matrix (should be same as reference)
                    target_size=target_size,
                    verbose=False,  # Don't save in the function, we'll save here
                    output_folder=None
                )
                if prior_depth is not None:
                    # Save with consistent scaling (use calibration image_id for filename)
                    save_path = os.path.join(depth_maps_folder, f"prior_depth_{image_id}_{img_name}.png")
                    _, _ = colorize_heatmap(
                        prior_depth, 
                        colormap='plasma', 
                        save_path=save_path, 
                        data_range=consistent_depth_range,
                        title_prefix=f"Prior Depth Map ({img_name}) - Cal_ID:{image_id} → Ref_ID:{ref_image_id}"
                    )
                    print(f"Saved prior depth map: {save_path}")
                else:
                    print(f"Warning: No prior depth computed for {img_name} (Cal_ID:{image_id} → Ref_ID:{ref_image_id})")
        
        # Save predicted depth maps (only for images with reference points)
        for pred_idx, pred in enumerate(predictions):
            image_id = image_ids[pred_idx]
            img_name = reconstruction.get_image_name(image_id)
            pred_depth = pred["depth_z"][0].squeeze(-1).cpu().numpy()  # (H, W)
            
            # Only save predicted depth maps for images that have reference points
            if reference_reconstruction is None or (image_id in image_name_mapping and image_name_mapping[image_id] is not None):
                # Save colorized predicted depth map
                save_path = os.path.join(depth_maps_folder, f"predicted_depth_{image_id}_{img_name}.png")
                _, _ = colorize_heatmap(
                    pred_depth, 
                    colormap='plasma', 
                    save_path=save_path, 
                    data_range=consistent_depth_range,
                    title_prefix=f"Predicted Depth Map ({img_name}) - Cal_ID:{image_id}"
                )
                print(f"Saved predicted depth map: {save_path}")
                
                # Save colorized confidence map
                pred_confidence = pred["conf"][0].squeeze(-1).cpu().numpy()  # (H, W)
                conf_save_path = os.path.join(depth_maps_folder, f"predicted_confidence_{image_id}_{img_name}.png")
                _, _ = colorize_heatmap(
                    pred_confidence, 
                    colormap='viridis', 
                    save_path=conf_save_path, 
                    data_range=None,  # Use automatic range for confidence (0-1)
                    title_prefix=f"Predicted Confidence Map ({img_name}) - Cal_ID:{image_id}"
                )
                print(f"Saved predicted confidence map: {conf_save_path}")
            else:
                print(f"Skipping predicted depth map for {img_name} (ID:{image_id}) - no reference points")
        
        print(f"Saved depth maps to {depth_maps_folder}")
    
    return predictions, all_prior_depths


def compute_pointcloud_from_depthmap_groundtruth(depthmap, image, groundtruth_intrinsics, groundtruth_pose, conf_threshold=0.0, max_points=None):
    """
    Compute point cloud from depth map using ground truth pose and intrinsics.
    This function confirms the depth map to point cloud computation is accurate.
    
    Args:
        depthmap: (H, W) numpy array with depth values
        image: (H, W, 3) numpy array with RGB values
        groundtruth_intrinsics: (3, 3) ground truth camera intrinsics matrix
        groundtruth_pose: (4, 4) ground truth camera pose matrix (cam2world)
        conf_threshold: Confidence threshold for filtering points (not used for ground truth)
        max_points: Maximum number of points to keep (None for no limit)
        
    Returns:
        tuple: (points_3d, colors) - numpy arrays
    """
    print("Computing point cloud from depth map using ground truth pose and intrinsics...")
    
    if depthmap is None or np.max(depthmap) == 0:
        print("Warning: No valid depth values in depth map")
        return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3)
    
    # Convert to torch tensors
    depthmap_torch = torch.tensor(depthmap, dtype=torch.float32)
    intrinsics_torch = torch.tensor(groundtruth_intrinsics, dtype=torch.float32)
    pose_torch = torch.tensor(groundtruth_pose, dtype=torch.float32)
    
    # Compute 3D points from depth using ground truth parameters
    pts3d, valid_mask = depthmap_to_world_frame(
        depthmap_torch, intrinsics_torch, pose_torch
    )
    
    # Convert to numpy
    pts3d_np = pts3d.cpu().numpy()
    valid_mask_np = valid_mask.cpu().numpy()
    
    # Extract valid points and colors
    if valid_mask_np.any():
        valid_pts = pts3d_np[valid_mask_np]
        valid_colors = image[valid_mask_np]  # (N, 3)
        
        # Ensure colors are in [0, 1] range
        if valid_colors.max() > 1.0:
            valid_colors = valid_colors / 255.0
        
        # Only subsample if max_points is specified and we have more points than the limit
        if max_points is not None and len(valid_pts) > max_points:
            print(f"Subsampling from {len(valid_pts)} to {max_points} points...")
            indices = np.random.choice(len(valid_pts), max_points, replace=False)
            valid_pts = valid_pts[indices]
            valid_colors = valid_colors[indices]
        
        print(f"Generated point cloud with {len(valid_pts)} points using ground truth parameters")
        return valid_pts, valid_colors
    else:
        print("Warning: No valid points found in depth map")
        return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3)


def get_groundtruth_camera_parameters(reconstruction, image_id, target_size, device="cuda"):
    """
    Get ground truth camera parameters for an image.
    
    Args:
        reconstruction: ColmapReconstruction object
        image_id: COLMAP image ID
        target_size: Target image size for scaling
        device: Device to place tensors on
        
    Returns:
        tuple: (scaled_intrinsics, pose_matrix) as torch tensors
    """
    # Get camera info
    camera = reconstruction.get_image_camera(image_id)
    original_width, original_height = camera.width, camera.height
    
    # Get ground truth intrinsics and scale for target resolution
    K = reconstruction.get_camera_calibration_matrix(image_id)
    
    # Scale intrinsics for resized image
    scale_x = target_size / original_width
    scale_y = target_size / original_height
    K_scaled = K.copy()
    K_scaled[0, :] *= scale_x  # Scale fx and cx
    K_scaled[1, :] *= scale_y  # Scale fy and cy
    
    # Get ground truth camera pose (world to camera transformation)
    cam_from_world = reconstruction.get_image_cam_from_world(image_id)
    pose_matrix = cam_from_world.matrix()  # 3x4 transformation matrix
    
    # Convert to 4x4 homogeneous matrix
    pose_4x4 = np.eye(4)
    pose_4x4[:3, :] = pose_matrix
    
    # Convert from cam_from_world to cam2world (world_from_cam) for MapAnything
    # MapAnything expects OpenCV cam2world convention: camera coordinates -> world coordinates
    pose_4x4 = np.linalg.inv(pose_4x4)
    
    # Convert to torch tensors and move to device
    intrinsics_torch = torch.tensor(K_scaled, dtype=torch.float32, device=device)
    camera_pose_torch = torch.tensor(pose_4x4, dtype=torch.float32, device=device)
    
    return intrinsics_torch, camera_pose_torch


def extract_points_from_single_prediction(pred, image_id, reconstruction, conf_threshold=0.0, use_groundtruth=True, device="cuda"):
    """
    Extract 3D points from a single prediction.
    
    Args:
        pred: Single prediction dictionary
        image_id: COLMAP image ID
        reconstruction: ColmapReconstruction object
        conf_threshold: Confidence threshold for filtering points
        use_groundtruth: If True, use ground truth parameters
        device: Device to place tensors on
        
    Returns:
        tuple: (points_3d, colors) - numpy arrays, or (None, None) if no valid points
    """
    # Extract prediction data
    depthmap_torch = pred["depth_z"][0].squeeze(-1)  # (H, W)
    conf_torch = pred["conf"][0]  # (H, W)
    
    # Ensure depth map is on the correct device
    if depthmap_torch.device != torch.device(device):
        depthmap_torch = depthmap_torch.to(device)
    
    if use_groundtruth:
        # Get ground truth camera parameters
        intrinsics_torch, camera_pose_torch = get_groundtruth_camera_parameters(
            reconstruction, image_id, depthmap_torch.shape[0], device
        )
        print(f"Using ground truth parameters for image {image_id}")
    else:
        # Use predicted parameters (original behavior)
        intrinsics_torch = pred["intrinsics"][0].to(device)  # (3, 3)
        camera_pose_torch = pred["camera_poses"][0].to(device)  # (4, 4)
        print(f"Using predicted parameters for image {image_id}")
    
    # Compute 3D points from depth
    pts3d, valid_mask = depthmap_to_world_frame(
        depthmap_torch, intrinsics_torch, camera_pose_torch
    )
    
    # Get mask from predictions
    mask = pred["mask"][0].squeeze(-1).cpu().numpy().astype(bool)
    mask = mask & valid_mask.cpu().numpy()  # Combine with valid depth mask
    
    # Apply confidence threshold
    if conf_threshold > 0:
        conf_mask = conf_torch.cpu().numpy() >= conf_threshold
        mask = mask & conf_mask
    
    # Extract valid points and colors
    if mask.any():
        valid_pts = pts3d.cpu().numpy()[mask]
        
        # Get colors from denormalized image
        img_no_norm = pred["img_no_norm"][0].cpu().numpy()  # (H, W, 3)
        valid_colors = img_no_norm[mask]  # (N, 3)
        
        return valid_pts, valid_colors
    else:
        return None, None


def extract_point_cloud_from_predictions(predictions, reconstruction, image_ids, conf_threshold=0.0, max_points=1000000, use_groundtruth=True, device="cuda"):
    """
    Extract point cloud from model predictions using ground truth pose and intrinsics.
    
    Args:
        predictions: List of prediction dictionaries
        reconstruction: ColmapReconstruction object for ground truth data
        image_ids: List of COLMAP image IDs corresponding to the predictions
        conf_threshold: Confidence threshold for filtering points
        max_points: Maximum number of points to keep
        use_groundtruth: If True, use ground truth pose and intrinsics instead of predicted ones
        device: Device to place tensors on
        
    Returns:
        tuple: (points_3d, colors) - numpy arrays
    """
    print("Extracting point cloud from predictions using ground truth pose and intrinsics...")
    
    all_points = []
    all_colors = []
    
    for view_idx, pred in enumerate(predictions):
        image_id = image_ids[view_idx]
        
        # Extract points from single prediction
        points, colors = extract_points_from_single_prediction(
            pred, image_id, reconstruction, conf_threshold, use_groundtruth, device
        )
        
        if points is not None:
            all_points.append(points)
            all_colors.append(colors)
    
    if len(all_points) == 0:
        print("Warning: No valid points found in predictions")
        return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3)
    
    # Concatenate all points and colors
    points_3d = np.concatenate(all_points, axis=0)
    colors = np.concatenate(all_colors, axis=0)
    
    # Ensure colors are in [0, 1] range
    if colors.max() > 1.0:
        colors = colors / 255.0
    
    # Randomly subsample if too many points
    if len(points_3d) > max_points:
        print(f"Subsampling from {len(points_3d)} to {max_points} points...")
        indices = np.random.choice(len(points_3d), max_points, replace=False)
        points_3d = points_3d[indices]
        colors = colors[indices]
    
    print(f"Generated point cloud with {len(points_3d)} points")
    return points_3d, colors


def build_image_name_mapping(reconstruction, reference_reconstruction, image_ids):
    """
    Build mapping between source and target image IDs for reference reconstruction.
    
    Args:
        reconstruction: Source ColmapReconstruction object
        reference_reconstruction: Target ColmapReconstruction object
        image_ids: List of image IDs to map
        
    Returns:
        dict: Mapping from source image_id to target image_id (or None if no match)
    """
    image_name_mapping = {}
    if reference_reconstruction is not None:
        for img_id in image_ids:
            match_result = find_image_match(
                source_reconstruction=reconstruction,
                target_reconstruction=reference_reconstruction,
                source_image_id=img_id,
                pose_tolerance_rotation=0.1,
                pose_tolerance_translation=0.1,
                verbose=False
            )
            if match_result['match_found']:
                image_name_mapping[img_id] = match_result['target_image_id']
            else:
                image_name_mapping[img_id] = None
    return image_name_mapping


def save_single_pointcloud(points, colors, image_id, img_name, pc_type, output_folder):
    """
    Save a single point cloud to file.
    
    Args:
        points: 3D points array
        colors: Colors array
        image_id: COLMAP image ID
        img_name: Image name
        pc_type: Type of point cloud ('predicted' or 'prior')
        output_folder: Output folder path
        
    Returns:
        str: Path to saved point cloud file, or None if no points
    """
    if len(points) == 0:
        return None
    
    individual_pc_folder = os.path.join(output_folder, "individual_pointclouds")
    os.makedirs(individual_pc_folder, exist_ok=True)
    
    pc = trimesh.PointCloud(vertices=points, colors=(colors * 255).astype(np.uint8))
    pc_path = os.path.join(individual_pc_folder, f"{pc_type}_{image_id}_{img_name}.ply")
    pc.export(pc_path)
    print(f"Saved {pc_type} point cloud: {pc_path} ({len(points)} points)")
    return pc_path


def save_depth_maps_as_npz(predictions, reconstruction, image_ids, output_folder, device, target_size, prior_depths=None):
    """
    Save predicted depth maps as NPZ files for later post-processing.
    Overwrites predicted depths with prior depths where they are valid.
    
    Args:
        predictions: List of prediction dictionaries from model inference
        reconstruction: ColmapReconstruction object
        image_ids: List of image IDs corresponding to predictions
        output_folder: Output directory for saving files
        device: Device for tensor operations
        target_size: Target image size used for preprocessing (e.g., args.resolution)
        prior_depths: Optional list of prior depth maps corresponding to predictions
    """
    print("Saving predicted depth maps as NPZ files...")
    
    # Create depth maps directory
    depth_maps_dir = os.path.join(output_folder, "depth_maps")
    os.makedirs(depth_maps_dir, exist_ok=True)
    
    for i, (prediction, image_id) in enumerate(zip(predictions, image_ids)):
        if prediction is None:
            continue
            
        # Get image name
        image_name = reconstruction.get_image_name(image_id)
        safe_name = image_name.replace('.jpg', '').replace('.png', '').replace('.jpeg', '')
        
        # Extract depth map and confidence
        depth_map = prediction['depth_z'].cpu().numpy()  # (H, W) or (1, H, W, 1)
        confidence_map = prediction['conf'].cpu().numpy()  # (H, W) or (1, H, W, 1)
        
        # Squeeze extra dimensions if present
        if depth_map.ndim == 4:
            depth_map = depth_map.squeeze(0).squeeze(-1)  # Remove batch and channel dims
        if confidence_map.ndim == 4:
            confidence_map = confidence_map.squeeze(0).squeeze(-1)  # Remove batch and channel dims
        
        # Print confidence map statistics
        conf_min = np.min(confidence_map)
        conf_max = np.max(confidence_map)
        conf_median = np.median(confidence_map)
        conf_mean = np.mean(confidence_map)
        print(f"Confidence map stats for {image_name}: min={conf_min:.3f}, max={conf_max:.3f}, median={conf_median:.3f}, mean={conf_mean:.3f}")
        
        # Store prior depth map separately (don't override predicted depths)
        prior_depth_map = None
        if prior_depths is not None and i < len(prior_depths) and prior_depths[i] is not None:
            prior_depth_map = prior_depths[i]
            if prior_depth_map.shape == depth_map.shape:
                valid_prior_pixels = np.sum(prior_depth_map > 0)
                print(f"Found {valid_prior_pixels} valid prior depth pixels for {image_name}")
            else:
                print(f"Warning: Prior depth shape {prior_depth_map.shape} doesn't match predicted depth shape {depth_map.shape} for {image_name}")
                prior_depth_map = None
        
        # Get camera parameters for this image
        K = reconstruction.get_camera_calibration_matrix(image_id)  # (3, 3)
        
        # Get camera pose (world to camera transformation)
        cam_from_world = reconstruction.get_image_cam_from_world(image_id)
        pose_matrix = cam_from_world.matrix()  # 3x4 transformation matrix
        
        # Convert to 4x4 homogeneous matrix
        pose_4x4 = np.eye(4)
        pose_4x4[:3, :] = pose_matrix
        
        # Convert from cam_from_world to cam2world (world_from_cam) for consistency
        pose_4x4 = np.linalg.inv(pose_4x4)
        
        # Scale intrinsics to match the image scaling used during depth computation
        # Get original image dimensions
        camera = reconstruction.get_image_camera(image_id)
        original_width = camera.width
        original_height = camera.height
        
        # Use the target size passed as parameter
        
        # Compute scaling factors
        scale_x = target_size / original_width
        scale_y = target_size / original_height
        
        # Scale intrinsics
        K_scaled = K.copy()
        K_scaled[0, :] *= scale_x  # Scale fx and cx
        K_scaled[1, :] *= scale_y  # Scale fy and cy
        
        # Create filename
        filename = f"depth_{image_id}_{safe_name}.npz"
        filepath = os.path.join(depth_maps_dir, filename)
        
        # Prepare data to save
        save_data = {
            'depth_map': depth_map,
            'confidence_map': confidence_map,
            'camera_intrinsics': K_scaled,  # Use scaled intrinsics
            'camera_pose': pose_4x4,  # Use 4x4 cam2world pose matrix
            'image_id': image_id,
            'image_name': image_name,
            'original_intrinsics': K  # Also save original intrinsics for reference
        }
        
        # Add prior depth map if available
        if prior_depth_map is not None:
            save_data['prior_depth_map'] = prior_depth_map
            print(f"  Added prior depth map to NPZ file for {image_name}")
        
        # Save as NPZ file
        np.savez_compressed(filepath, **save_data)
        
        print(f"Saved depth map: {filepath}")
    
    print(f"Saved {len([p for p in predictions if p is not None])} depth maps to {depth_maps_dir}")


def load_npz_and_generate_pointcloud(npz_filepath, conf_threshold=0.0, max_points=None):
    """
    Load a depth map from NPZ file and generate a point cloud.
    
    Args:
        npz_filepath: Path to the NPZ file containing depth map and camera parameters
        conf_threshold: Confidence threshold for filtering points (default: 0.0)
        max_points: Maximum number of points to keep (None for no limit)
        
    Returns:
        tuple: (points_3d, colors, metadata) - numpy arrays and metadata dict
    """
    print(f"Loading depth map from: {npz_filepath}")
    
    # Load NPZ file
    data = np.load(npz_filepath)
    
    # Extract data
    depth_map = data['depth_map']  # (H, W) or (1, H, W, 1)
    confidence_map = data['confidence_map']  # (H, W) or (1, H, W, 1)
    camera_intrinsics = data['camera_intrinsics']  # (3, 3) - scaled intrinsics
    camera_pose = data['camera_pose']  # (4, 4) - cam2world pose
    image_id = data['image_id']
    image_name = data['image_name']
    
    # Squeeze extra dimensions if present
    if depth_map.ndim == 4:
        depth_map = depth_map.squeeze(0).squeeze(-1)  # Remove batch and channel dims
    if confidence_map.ndim == 4:
        confidence_map = confidence_map.squeeze(0).squeeze(-1)  # Remove batch and channel dims
    
    print(f"Loaded depth map for {image_name} (ID: {image_id})")
    print(f"Depth map shape: {depth_map.shape}")
    print(f"Depth range: {np.min(depth_map[depth_map > 0]):.3f} - {np.max(depth_map[depth_map > 0]):.3f}")
    print(f"Confidence range: {np.min(confidence_map):.3f} - {np.max(confidence_map):.3f}")
    
    # Filter by confidence threshold
    if conf_threshold > 0.0:
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
    
    # Convert to torch tensors
    depthmap_torch = torch.tensor(depth_map_filtered, dtype=torch.float32)
    intrinsics_torch = torch.tensor(camera_intrinsics, dtype=torch.float32)
    pose_torch = torch.tensor(camera_pose, dtype=torch.float32)
    
    # Compute 3D points from depth using the saved camera parameters
    result = depthmap_to_world_frame(
        depthmap_torch, intrinsics_torch, pose_torch
    )
    
    # Handle different return formats
    if len(result) == 2:
        pts3d, valid_mask = result
    else:
        print(f"Warning: depthmap_to_world_frame returned {len(result)} values, expected 2")
        pts3d, valid_mask = result[0], result[1]
    
    # Convert to numpy
    pts3d_np = pts3d.cpu().numpy()
    valid_mask_np = valid_mask.cpu().numpy()
    
    # Extract valid points
    if valid_mask_np.any():
        valid_pts = pts3d_np[valid_mask_np]
        
        # Create dummy colors (you could load actual image colors if needed)
        colors = np.ones((len(valid_pts), 3), dtype=np.float32) * 0.5  # Gray color
        
        # Subsample if max_points is specified
        if max_points is not None and len(valid_pts) > max_points:
            print(f"Subsampling from {len(valid_pts)} to {max_points} points...")
            indices = np.random.choice(len(valid_pts), max_points, replace=False)
            valid_pts = valid_pts[indices]
            colors = colors[indices]
        
        print(f"Generated point cloud with {len(valid_pts)} points")
        
        # Prepare metadata
        metadata = {
            'image_id': int(image_id),
            'image_name': str(image_name),
            'depth_range': (float(np.min(depth_map[depth_map > 0])), float(np.max(depth_map[depth_map > 0]))),
            'confidence_range': (float(np.min(confidence_map)), float(np.max(confidence_map))),
            'camera_intrinsics': camera_intrinsics,
            'camera_pose': camera_pose,
            'conf_threshold': conf_threshold,
            'max_points': max_points
        }
        
        return valid_pts, colors, metadata
    else:
        print("Warning: No valid points found in depth map")
        return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3), {}


def load_multiple_npz_and_generate_pointclouds(npz_folder, conf_threshold=0.0, max_points_per_file=None, max_total_points=None):
    """
    Load multiple NPZ files from a folder and generate point clouds.
    
    Args:
        npz_folder: Path to folder containing NPZ files
        conf_threshold: Confidence threshold for filtering points (default: 0.0)
        max_points_per_file: Maximum points per file (None for no limit)
        max_total_points: Maximum total points across all files (None for no limit)
        
    Returns:
        tuple: (all_points, all_colors, metadata_list) - combined point clouds and metadata
    """
    import glob
    
    print(f"Loading NPZ files from: {npz_folder}")
    
    # Find all NPZ files
    npz_files = glob.glob(os.path.join(npz_folder, "*.npz"))
    if not npz_files:
        print(f"No NPZ files found in {npz_folder}")
        return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3), []
    
    print(f"Found {len(npz_files)} NPZ files")
    
    all_points = []
    all_colors = []
    metadata_list = []
    
    for i, npz_file in enumerate(npz_files):
        print(f"\nProcessing file {i+1}/{len(npz_files)}: {os.path.basename(npz_file)}")
        
        try:
            points, colors, metadata = load_npz_and_generate_pointcloud(
                npz_file, conf_threshold=conf_threshold, max_points=max_points_per_file
            )
            
            if len(points) > 0:
                all_points.append(points)
                all_colors.append(colors)
                metadata_list.append(metadata)
                print(f"Added {len(points)} points from {metadata['image_name']}")
            else:
                print(f"Skipped {os.path.basename(npz_file)} - no valid points")
                
        except Exception as e:
            print(f"Error processing {npz_file}: {e}")
            continue
    
    if not all_points:
        print("No valid point clouds generated from any NPZ files")
        return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3), []
    
    # Combine all point clouds
    combined_points = np.vstack(all_points)
    combined_colors = np.vstack(all_colors)
    
    print(f"\nCombined point cloud: {len(combined_points)} points from {len(all_points)} files")
    
    # Apply total point limit if specified
    if max_total_points is not None and len(combined_points) > max_total_points:
        print(f"Subsampling from {len(combined_points)} to {max_total_points} total points...")
        indices = np.random.choice(len(combined_points), max_total_points, replace=False)
        combined_points = combined_points[indices]
        combined_colors = combined_colors[indices]
    
    print(f"Final point cloud: {len(combined_points)} points")
    
    return combined_points, combined_colors, metadata_list


def process_single_image_pointclouds(pred, image_id, reconstruction, reference_reconstruction, 
                                    image_name_mapping, max_points_per_image, device="cuda"):
    """
    Process point clouds for a single image (both predicted and prior if available).
    
    Args:
        pred: Single prediction dictionary
        image_id: COLMAP image ID
        reconstruction: ColmapReconstruction object
        reference_reconstruction: Reference reconstruction object
        image_name_mapping: Mapping from source to target image IDs
        max_points_per_image: Maximum points per image
        device: Device to place tensors on
        
    Returns:
        tuple: (predicted_points, predicted_colors, prior_points, prior_colors)
    """
    img_name = reconstruction.get_image_name(image_id)
    
    # Get ground truth parameters for this image
    target_size = pred["depth_z"][0].squeeze(-1).shape[0]
    intrinsics_torch, pose_torch = get_groundtruth_camera_parameters(
        reconstruction, image_id, target_size, device
    )
    
    # Convert to numpy for ground truth function
    K_scaled = intrinsics_torch.cpu().numpy()
    pose_4x4 = pose_torch.cpu().numpy()
    
    # Get predicted depth map and image
    pred_depth = pred["depth_z"][0].squeeze(-1).cpu().numpy()
    pred_image = pred["img_no_norm"][0].cpu().numpy()
    
    # Compute predicted point cloud using ground truth parameters
    pred_points, pred_colors = compute_pointcloud_from_depthmap_groundtruth(
        pred_depth, pred_image, K_scaled, pose_4x4, 
        conf_threshold=0.0, max_points=max_points_per_image
    )
    
    # Compute prior point cloud if reference reconstruction is available
    prior_points, prior_colors = None, None
    if reference_reconstruction is not None:
        # Check if image has a match in reference reconstruction
        if image_id in image_name_mapping and image_name_mapping[image_id] is not None:
            ref_image_id = image_name_mapping[image_id]
            
            # Get the pose from the reference reconstruction
            ref_cam_from_world = reference_reconstruction.get_image_cam_from_world(ref_image_id)
            ref_pose_matrix = ref_cam_from_world.matrix()
            ref_pose_4x4 = np.eye(4)
            ref_pose_4x4[:3, :] = ref_pose_matrix
            ref_pose_4x4 = np.linalg.inv(ref_pose_4x4)
            
            # Compute prior depth map
            prior_depth, _ = compute_prior_depthmap(
                reference_reconstruction=reference_reconstruction,
                image_id=ref_image_id,
                scaled_intrinsics=K_scaled,
                pose_matrix=ref_pose_matrix,
                target_size=target_size,
                verbose=False,
                output_folder=None
            )
            
            if prior_depth is not None:
                # Compute prior point cloud using ground truth parameters
                prior_points, prior_colors = compute_pointcloud_from_depthmap_groundtruth(
                    prior_depth, pred_image, K_scaled, pose_4x4,
                    conf_threshold=0.0, max_points=max_points_per_image
                )
            else:
                print(f"Warning: No prior depth computed for image {img_name} (ID:{image_id})")
        else:
            print(f"Warning: Image {img_name} (ID:{image_id}) not found in reference reconstruction")
    
    return pred_points, pred_colors, prior_points, prior_colors


def validate_prior_depthmaps(reconstruction, reference_reconstruction, image_ids, 
                            target_size, output_folder):
    """
    Validate prior depthmap generation before running full predictions.
    This function tests the prior depthmap generation process and compares with reference points.
    
    Args:
        reconstruction: ColmapReconstruction object (calibration reconstruction)
        reference_reconstruction: ColmapReconstruction object (reference with 3D points)
        image_ids: List of image IDs to test
        target_size: Target image size for depth maps
        output_folder: Output folder for saving results
        
    Returns:
        bool: True if validation passes, False if significant issues found
    """
    print("\n" + "="*60)
    print("🔍 VALIDATING PRIOR DEPTHMAP GENERATION")
    print("="*60)
    
    if reference_reconstruction is None:
        print("❌ No reference reconstruction provided - cannot validate prior depthmaps")
        return False
    
    # Build image name mapping
    print("Building image name mapping between reconstructions...")
    image_name_mapping = build_image_name_mapping(
        reconstruction, reference_reconstruction, image_ids
    )
    
    # Find images with matches
    matched_images = [(img_id, ref_id) for img_id, ref_id in image_name_mapping.items() 
                     if ref_id is not None]
    
    if len(matched_images) == 0:
        print("❌ No matching images found between reconstructions")
        return False
    
    print(f"Found {len(matched_images)} matching images")
    
    # Test all matching images
    test_images = matched_images
    print(f"Testing all {len(test_images)} matching images for prior depthmap validation")
    
    validation_results = []
    overall_success = True
    
    for cal_image_id, ref_image_id in test_images:
        img_name = reconstruction.get_image_name(cal_image_id)
        print(f"\n--- Testing Image: {img_name} (Cal_ID:{cal_image_id} → Ref_ID:{ref_image_id}) ---")
        
        try:
            # Run comparison
            comparison_results = compare_prior_depthmap_with_reference_points(
                reconstruction, reference_reconstruction, cal_image_id, ref_image_id,
                target_size, output_folder, verbose=True
            )
            
            if comparison_results is None:
                print(f"❌ Failed to generate comparison for {img_name}")
                validation_results.append({
                    'image_id': cal_image_id,
                    'image_name': img_name,
                    'success': False,
                    'error': 'Comparison failed'
                })
                overall_success = False
                continue
            
            # Check if results are acceptable
            success = True
            issues = []
            critical_failure = False
            
            # Check for critical failures that indicate bugs
            if 'distance_statistics' in comparison_results:
                distance_stats = comparison_results['distance_statistics']
                mean_distance = distance_stats.get('mean_distance', float('inf'))
                
                # Critical failure thresholds - these indicate bugs that must be fixed
                if mean_distance > 0.1:  # 10cm threshold for critical failure
                    critical_failure = True
                    success = False
                    issues.append(f"CRITICAL: Large mean distance: {mean_distance:.6f}m (>10cm)")
                    print(f"\n🚨 CRITICAL VALIDATION FAILURE DETECTED!")
                    print(f"   Image: {img_name}")
                    print(f"   Mean distance: {mean_distance:.6f}m")
                    print(f"   This indicates a bug in the depth map generation or point cloud conversion.")
                    print(f"   Validation stopped - fix the bug before continuing.")
                    break
            
            # Check other metrics for warnings
            if 'overlap_percentage' in comparison_results:
                overlap = comparison_results['overlap_percentage']
                mean_dist = comparison_results['mean_min_distance']
                
                if overlap < 30:  # Very low overlap threshold
                    success = False
                    issues.append(f"Very low overlap: {overlap:.1f}%")
                
                if mean_dist > 2.0:  # Very large distance threshold
                    success = False
                    issues.append(f"Large mean distance: {mean_dist:.3f}m")
                
                if overlap < 50:  # Warning threshold
                    issues.append(f"Low overlap: {overlap:.1f}%")
                
                if mean_dist > 1.0:  # Warning threshold
                    issues.append(f"Large distance: {mean_dist:.3f}m")
            
            validation_results.append({
                'image_id': cal_image_id,
                'image_name': img_name,
                'success': success,
                'comparison_results': comparison_results,
                'issues': issues,
                'critical_failure': critical_failure
            })
            
            if critical_failure:
                overall_success = False
                print(f"❌ CRITICAL VALIDATION FAILURE for {img_name}: {', '.join(issues)}")
                break  # Stop validation immediately
            elif not success:
                overall_success = False
                print(f"❌ Validation failed for {img_name}: {', '.join(issues)}")
            else:
                print(f"✅ Validation passed for {img_name}")
                if issues:
                    print(f"   Warnings: {', '.join(issues)}")
        
        except Exception as e:
            print(f"❌ Error testing {img_name}: {e}")
            validation_results.append({
                'image_id': cal_image_id,
                'image_name': img_name,
                'success': False,
                'error': str(e)
            })
            overall_success = False
    
    # Summary
    print(f"\n" + "="*60)
    print("📊 VALIDATION SUMMARY")
    print("="*60)
    
    successful_tests = sum(1 for r in validation_results if r['success'])
    total_tests = len(validation_results)
    critical_failures = sum(1 for r in validation_results if r.get('critical_failure', False))
    
    print(f"Successful tests: {successful_tests}/{total_tests}")
    if critical_failures > 0:
        print(f"🚨 CRITICAL FAILURES: {critical_failures}")
    
    if critical_failures > 0:
        print("\n🚨 CRITICAL VALIDATION FAILURE DETECTED!")
        print("   A bug has been identified in the depth map generation or point cloud conversion.")
        print("   The validation was stopped to prevent further issues.")
        print("   You MUST fix this bug before continuing with any predictions.")
        print("   Check the comparison results above for details on the specific issue.")
    elif overall_success:
        print("✅ All prior depthmap validations passed!")
        print("   The prior depthmap generation process appears to be working correctly.")
        print("   You can proceed with full MapAnything predictions.")
    else:
        print("❌ Some prior depthmap validations failed!")
        print("   Issues detected in the prior depthmap generation process.")
        print("   Please review the comparison results and fix issues before proceeding.")
        
        # Print detailed results
        print("\nDetailed Results:")
        for result in validation_results:
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            print(f"  {status} {result['image_name']} (ID:{result['image_id']})")
            if 'issues' in result and result['issues']:
                for issue in result['issues']:
                    print(f"    - {issue}")
            if 'error' in result:
                print(f"    - Error: {result['error']}")
    
    return overall_success


def compute_point_to_point_distances(points1, points2):
    """
    Compute closest point-to-point distances between two point sets.
    
    Args:
        points1: First set of 3D points (N1, 3)
        points2: Second set of 3D points (N2, 3)
        
    Returns:
        dict: Distance statistics
    """
    if len(points1) == 0 or len(points2) == 0:
        return {
            'mean_distance': float('inf'),
            'median_distance': float('inf'),
            'max_distance': float('inf'),
            'min_distance': float('inf'),
            'std_distance': float('inf')
        }
    
    # Compute distances between all point pairs
    distances = cdist(points1, points2)
    
    # Find minimum distances for each point in points1
    min_distances = np.min(distances, axis=1)
    
    return {
        'mean_distance': np.mean(min_distances),
        'median_distance': np.median(min_distances),
        'max_distance': np.max(min_distances),
        'min_distance': np.min(min_distances),
        'std_distance': np.std(min_distances)
    }


def compare_prior_depthmap_with_reference_points(reconstruction, reference_reconstruction, 
                                                image_id, ref_image_id, target_size, 
                                                output_folder=None, verbose=False):
    """
    Generate a prior depthmap from reference 3D points, convert it back to 3D points,
    and compare with the original reference 3D points.
    
    Args:
        reconstruction: ColmapReconstruction object (calibration reconstruction)
        reference_reconstruction: ColmapReconstruction object (reference with 3D points)
        image_id: Image ID in calibration reconstruction
        ref_image_id: Image ID in reference reconstruction
        target_size: Target image size for depth map
        output_folder: Optional output folder for saving comparison results
        verbose: If True, print detailed comparison information
        
    Returns:
        dict: Comparison results including distance statistics
    """
    print(f"\n=== Comparing Prior Depthmap with Reference Points ===")
    print(f"Calibration Image ID: {image_id}")
    print(f"Reference Image ID: {ref_image_id}")
    
    # Get ground truth parameters for calibration image
    camera = reconstruction.get_image_camera(image_id)
    original_width, original_height = camera.width, camera.height
    
    # Get ground truth intrinsics and scale for target resolution
    K = reconstruction.get_camera_calibration_matrix(image_id)
    scale_x = target_size / original_width
    scale_y = target_size / original_height
    K_scaled = K.copy()
    K_scaled[0, :] *= scale_x
    K_scaled[1, :] *= scale_y
    
    # Get ground truth camera pose from calibration reconstruction
    cal_cam_from_world = reconstruction.get_image_cam_from_world(image_id)
    cal_pose_matrix = cal_cam_from_world.matrix()  # This is cam_from_world (3x4)
    
    # Get reference 3D points for this image
    ref_points_3d, ref_points_2d, ref_point_ids = reference_reconstruction.get_visible_3d_points(
        ref_image_id, min_track_length=2
    )
    
    if len(ref_points_3d) == 0:
        print("Error: No reference 3D points found")
        return None
    
    print(f"Found {len(ref_points_3d)} reference 3D points")
    
    # Step 1: Generate prior depth map using the same function as the normal pipeline
    # Since the poses are identical for matching images, we can use the calibration image's pose matrix
    prior_depth, depth_range = compute_prior_depthmap(
        reference_reconstruction=reference_reconstruction,
        image_id=ref_image_id,
        scaled_intrinsics=K_scaled,
        pose_matrix=cal_pose_matrix,  # Use calibration image's pose matrix (should be same as reference)
        target_size=target_size,
        verbose=False,
        output_folder=None
    )
    
    if prior_depth is None or np.sum(prior_depth > 0) == 0:
        print("Error: Could not generate prior depth map")
        return None
    
    print(f"Generated prior depth map with {np.sum(prior_depth > 0)} valid pixels")
    print(f"Depth range: {np.min(prior_depth[prior_depth > 0]):.3f} - {np.max(prior_depth[prior_depth > 0]):.3f}")
    
    # Step 2: Convert prior depth map back to 3D points using ground truth pose/intrinsics
    # Create a dummy image for color information
    dummy_image = np.ones((target_size, target_size, 3), dtype=np.float32) * 0.5
    
    # Convert depth map to point cloud using ground truth parameters
    # Convert cam_from_world to cam2world for the function
    pose_4x4 = np.eye(4)
    pose_4x4[:3, :] = cal_pose_matrix
    pose_4x4 = np.linalg.inv(pose_4x4)  # Convert cam_from_world to cam2world
    
    prior_points, prior_colors = compute_pointcloud_from_depthmap_groundtruth(
        prior_depth, dummy_image, K_scaled, pose_4x4,
        conf_threshold=0.0, max_points=None
    )
    
    if len(prior_points) == 0:
        print("Error: Could not generate point cloud from prior depth map")
        return None
    
    print(f"Generated {len(prior_points)} points from prior depth map")
    
    # Step 3: Compare prior_points with original reference 3D points
    print(f"Comparing {len(prior_points)} prior points with {len(ref_points_3d)} reference points...")
    
    # Compute point-to-point distances
    distance_stats = compute_point_to_point_distances(prior_points, ref_points_3d)
    
    print(f"Distance Analysis:")
    print(f"  Mean distance: {distance_stats['mean_distance']:.6f}m")
    print(f"  Median distance: {distance_stats['median_distance']:.6f}m")
    print(f"  Max distance: {distance_stats['max_distance']:.6f}m")
    print(f"  Min distance: {distance_stats['min_distance']:.6f}m")
    print(f"  Std distance: {distance_stats['std_distance']:.6f}m")
    
    # Check if distances are small (should be very close to zero for perfect reconstruction)
    tolerance = 0.01  # 1cm tolerance
    if distance_stats['mean_distance'] < tolerance:
        print("✅ EXCELLENT: Mean distance is very small - depth map conversion is working correctly!")
    elif distance_stats['mean_distance'] < 0.1:  # 10cm tolerance
        print("✅ GOOD: Mean distance is small - depth map conversion is working well")
    else:
        print("⚠️  WARNING: Mean distance is large - potential issues in depth map conversion")
        print("   Possible issues:")
        print("   - Incorrect intrinsics scaling")
        print("   - Depth map projection errors")
        print("   - Point cloud conversion errors")
    
    # Prepare results
    comparison_results = {
        'prior_points_count': len(prior_points),
        'reference_points_count': len(ref_points_3d),
        'depth_map_valid_pixels': np.sum(prior_depth > 0),
        'calibration_image_id': image_id,
        'reference_image_id': ref_image_id,
        'distance_statistics': distance_stats
    }
    
    # Save comparison results if output folder provided
    if output_folder is not None:
        comparison_folder = os.path.join(output_folder, "prior_comparison")
        os.makedirs(comparison_folder, exist_ok=True)
        
        # Save prior point cloud (from depth map)
        if len(prior_points) > 0:
            prior_pc = trimesh.PointCloud(vertices=prior_points, colors=(prior_colors * 255).astype(np.uint8))
            prior_pc_path = os.path.join(comparison_folder, f"prior_points_{image_id}_{ref_image_id}.ply")
            prior_pc.export(prior_pc_path)
            print(f"Saved prior point cloud: {prior_pc_path}")
        
        # Save reference points (original 3D points)
        if len(ref_points_3d) > 0:
            ref_colors = np.ones((len(ref_points_3d), 3)) * [1, 0, 0]  # Red for reference
            ref_pc = trimesh.PointCloud(vertices=ref_points_3d, colors=(ref_colors * 255).astype(np.uint8))
            ref_pc_path = os.path.join(comparison_folder, f"reference_points_{image_id}_{ref_image_id}.ply")
            ref_pc.export(ref_pc_path)
            print(f"Saved reference points: {ref_pc_path}")
        
        # Save combined point cloud for visualization
        if len(prior_points) > 0 and len(ref_points_3d) > 0:
            combined_points = np.vstack([prior_points, ref_points_3d])
            combined_colors = np.vstack([
                prior_colors,
                np.ones((len(ref_points_3d), 3)) * [1, 0, 0]  # Red for reference
            ])
            combined_pc = trimesh.PointCloud(vertices=combined_points, colors=(combined_colors * 255).astype(np.uint8))
            combined_pc_path = os.path.join(comparison_folder, f"combined_{image_id}_{ref_image_id}.ply")
            combined_pc.export(combined_pc_path)
            print(f"Saved combined point cloud: {combined_pc_path}")
    
    return comparison_results


def save_individual_pointclouds(batch_predictions, batch_image_ids, reconstruction, 
                               reference_reconstruction, output_folder, max_points, device="cuda"):
    """
    Save individual prior and predicted point clouds for each image in the batch.
    
    Args:
        batch_predictions: List of prediction dictionaries
        batch_image_ids: List of COLMAP image IDs
        reconstruction: ColmapReconstruction object
        reference_reconstruction: Reference reconstruction object
        output_folder: Output folder path
        max_points: Maximum total points (will be divided among images)
        device: Device to place tensors on
    """
    # Build image name mapping for reference reconstruction if available
    image_name_mapping = build_image_name_mapping(
        reconstruction, reference_reconstruction, batch_image_ids
    )
    
    max_points_per_image = max_points // len(batch_image_ids)
    
    for view_idx, pred in enumerate(batch_predictions):
        image_id = batch_image_ids[view_idx]
        img_name = reconstruction.get_image_name(image_id)
        
        # Process point clouds for this image
        pred_points, pred_colors, prior_points, prior_colors = process_single_image_pointclouds(
            pred, image_id, reconstruction, reference_reconstruction, 
            image_name_mapping, max_points_per_image, device
        )
        
        # Save predicted point cloud
        if pred_points is not None and len(pred_points) > 0:
            save_single_pointcloud(pred_points, pred_colors, image_id, img_name, 
                                 "predicted", output_folder)
        else:
            print(f"Warning: No predicted points for image {img_name} (ID:{image_id})")
        
        # Save prior point cloud if available and run comparison
        if prior_points is not None and len(prior_points) > 0:
            save_single_pointcloud(prior_points, prior_colors, image_id, img_name, 
                                 "prior", output_folder)
        elif reference_reconstruction is not None:
            print(f"Warning: No prior points for image {img_name} (ID:{image_id})")


def split_into_batches_smart(reconstruction, image_ids, batch_size):
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
    batches = []
    used_as_reference = set()  # Images that have been used as reference images
    
    # Sort image IDs for consistent processing order
    remaining_candidates = sorted(image_ids)
    
    print(f"Creating smart batches with geometric relationships...")
    
    while len(used_as_reference) < len(image_ids):
        # Find next unused reference image
        reference_image = None
        for img_id in remaining_candidates:
            if img_id not in used_as_reference:
                reference_image = img_id
                break
        
        if reference_image is None:
            break
            
        # Find best partner images for this reference
        try:
            # Ensure image point mappings are built
            reconstruction._ensure_image_point_maps()
            
            best_partners = reconstruction._find_best_partner_for_image(
                reference_image, 
                min_points=50,  # Lower threshold for more flexibility
                parallax_sample_size=50
            )
            
            # Filter out invalid partners (-1) and create batch
            valid_partners = [pid for pid in best_partners if pid != -1 and pid in image_ids]
            
            # Start batch with reference image
            batch = [reference_image]
            
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
            
            batches.append(batch)
            
            # Mark ALL images in this batch as used (cannot be reference images anymore)
            for img_id in batch:
                used_as_reference.add(img_id)
            
            print(f"Batch {len(batches)}: Reference {reference_image} with {len(batch)-1} partners: {batch[1:]}")
            print(f"  Marked {len(batch)} images as used: {batch}")
            
        except Exception as e:
            print(f"Error: Could not find partners for image {reference_image}: {e}")
            # Mark this image as used and continue
            used_as_reference.add(reference_image)
            continue
    
    # Check if any images were missed (shouldn't happen with the new logic)
    all_batched_images = set()
    for batch in batches:
        all_batched_images.update(batch)
    
    remaining_unprocessed = [img_id for img_id in image_ids if img_id not in all_batched_images]
    
    if remaining_unprocessed:
        print(f"Warning: {len(remaining_unprocessed)} images not included in any batch: {remaining_unprocessed}")
        # Add them as a final batch
        batches.append(remaining_unprocessed)
        print(f"Final batch {len(batches)}: Remaining images {remaining_unprocessed}")
    
    print(f"Created {len(batches)} smart batches with geometric relationships")
    print(f"Total images processed: {len(all_batched_images)}/{len(image_ids)}")
    return batches


def split_into_batches(image_ids, batch_size):
    """
    Simple sequential split into batches (fallback method).
    
    Args:
        image_ids: List of image IDs
        batch_size: Maximum number of images per batch
        
    Returns:
        List of batches, where each batch is a list of image IDs
    """
    batches = []
    for i in range(0, len(image_ids), batch_size):
        batch = image_ids[i:i + batch_size]
        batches.append(batch)
    return batches


def save_batch_pointcloud(points_3d, colors, batch_idx, temp_dir):
    """
    Save a batch point cloud to a temporary file.
    
    Args:
        points_3d: 3D points array
        colors: Colors array
        batch_idx: Batch index for filename
        temp_dir: Temporary directory path
        
    Returns:
        Path to saved point cloud file
    """
    if len(points_3d) == 0:
        return None
    
    # Convert colors to uint8 for PLY format
    colors_uint8 = (colors * 255).astype(np.uint8)
    
    # Create and save point cloud
    point_cloud = trimesh.PointCloud(vertices=points_3d, colors=colors_uint8)
    
    batch_filename = f"batch_{batch_idx:03d}.ply"
    batch_path = os.path.join(temp_dir, batch_filename)
    point_cloud.export(batch_path)
    
    print(f"Saved batch {batch_idx} with {len(points_3d)} points to {batch_filename}")
    return batch_path


def merge_pointclouds_with_open3d(batch_files, output_path, max_points=None):
    """
    Merge multiple point cloud files into a single point cloud using Open3D.
    
    Args:
        batch_files: List of paths to batch point cloud files
        output_path: Path for the final merged point cloud
        max_points: Optional maximum number of points to keep after merging
    """
    print(f"Merging {len(batch_files)} point cloud batches...")
    
    merged_points = []
    merged_colors = []
    
    for batch_file in batch_files:
        if batch_file is None:
            continue
            
        # Load point cloud with Open3D
        pcd = o3d.io.read_point_cloud(batch_file)
        
        if len(pcd.points) == 0:
            continue
            
        # Extract points and colors
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        
        merged_points.append(points)
        merged_colors.append(colors)
        
        print(f"Loaded {len(points)} points from {os.path.basename(batch_file)}")
    
    if len(merged_points) == 0:
        raise ValueError("No valid point clouds found to merge")
    
    # Concatenate all points and colors
    all_points = np.concatenate(merged_points, axis=0)
    all_colors = np.concatenate(merged_colors, axis=0)
    
    print(f"Total points before subsampling: {len(all_points)}")
    
    # Subsample if necessary
    if max_points is not None and len(all_points) > max_points:
        print(f"Subsampling from {len(all_points)} to {max_points} points...")
        indices = np.random.choice(len(all_points), max_points, replace=False)
        all_points = all_points[indices]
        all_colors = all_colors[indices]
    
    # Create final point cloud
    final_pcd = o3d.geometry.PointCloud()
    final_pcd.points = o3d.utility.Vector3dVector(all_points)
    final_pcd.colors = o3d.utility.Vector3dVector(all_colors)
    
    # Save merged point cloud
    o3d.io.write_point_cloud(output_path, final_pcd)
    
    print(f"Successfully saved merged point cloud with {len(all_points)} points to {output_path}")
    return len(all_points)


def main():
    """Main function."""
    args = parse_args()
    
    # Print configuration
    print("Arguments:", vars(args))
    
    # Set seed for reproducibility
    seed_everything(args.seed)
    
    # Validate scene folder structure
    if not os.path.isdir(args.scene_folder):
        raise ValueError(f"Scene folder {args.scene_folder} does not exist")
    
    images_dir = os.path.join(args.scene_folder, "images")
    sparse_dir = os.path.join(args.scene_folder, "sparse")
    
    if not os.path.isdir(images_dir):
        raise ValueError(f"Images directory {images_dir} does not exist")
    
    if not os.path.isdir(sparse_dir):
        raise ValueError(f"Sparse directory {sparse_dir} does not exist")
    
    # Set output folder
    if args.output_folder is None:
        args.output_folder = os.path.join(args.scene_folder, "output")
    else:
        args.output_folder = os.path.join(args.scene_folder, args.output_folder)
    
    # Create output folder
    os.makedirs(args.output_folder, exist_ok=True)
    print(f"Output folder: {args.output_folder}")
    
    # Set final point cloud path
    final_pointcloud_path = os.path.join(args.output_folder, "final_pointcloud.ply")
    
    # Set device and dtype
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Use float32 for better compatibility with linear algebra operations
    dtype = torch.float32
    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")
    
    # Load COLMAP reconstruction
    print(f"Loading COLMAP reconstruction from {sparse_dir}...")
    reconstruction = ColmapReconstruction(sparse_dir)
    print(f"Loaded reconstruction with {reconstruction.get_num_images()} images")
    
    # Load reference reconstruction if provided
    reference_reconstruction = None
    if args.reference_reconstruction is not None:
        if not os.path.isdir(args.reference_reconstruction):
            raise ValueError(f"Reference reconstruction path {args.reference_reconstruction} does not exist")
        print(f"Loading reference COLMAP reconstruction from {args.reference_reconstruction}...")
        reference_reconstruction = ColmapReconstruction(args.reference_reconstruction)
        print(f"Loaded reference reconstruction with {reference_reconstruction.get_num_images()} images and {len(reference_reconstruction.reconstruction.points3D)} 3D points")
        
        # Enable verbose depth map saving if verbose flag is set
        if args.verbose:
            depth_maps_folder = os.path.join(args.output_folder, "depth_maps")
            print(f"Verbose mode enabled: Prior and predicted depth maps will be saved to {depth_maps_folder}")
    elif args.verbose and args.reference_reconstruction is None:
        depth_maps_folder = os.path.join(args.output_folder, "depth_maps")
        print(f"Verbose mode enabled: Only predicted depth maps will be saved to {depth_maps_folder}")
        print("Note: No reference reconstruction provided, so no prior depth information will be available to the model.")
    
    # Initialize model
    if args.apache:
        model_name = "facebook/map-anything-apache"
        print("Loading Apache 2.0 licensed MapAnything model...")
    else:
        model_name = "facebook/map-anything"
        print("Loading CC-BY-NC 4.0 licensed MapAnything model...")
    
    model = MapAnything.from_pretrained(model_name).to(device)
    model.eval()
    
    # Get all image IDs and filter to only include those with reference points
    all_image_ids = reconstruction.get_all_image_ids()
    print(f"Total images in reconstruction: {len(all_image_ids)}")
    
    # Filter images to only include those with reference points
    if reference_reconstruction is not None:
        print("\n=== Filtering Images with Reference Points ===")
        valid_image_ids, global_image_name_mapping = filter_images_with_reference_points(
            reconstruction, reference_reconstruction, all_image_ids, verbose=args.verbose
        )
        
        if len(valid_image_ids) == 0:
            print("❌ No valid images found with reference points. Cannot run inference.")
            return
        
        print(f"\n📊 Global filtering: {len(valid_image_ids)}/{len(all_image_ids)} images with reference points")
        print(f"🔍 DEBUG: Valid image IDs: {valid_image_ids[:10]}...")  # Show first 10
        all_image_ids = valid_image_ids  # Use only valid images
    else:
        global_image_name_mapping = {}
    
    # Validate prior depthmaps only if explicitly requested
    if args.validate_prior_only:
        if reference_reconstruction is None:
            print("\n❌ Error: --validate_prior_only requires --reference_reconstruction to be provided")
            return
            
        print("\n🔍 Running prior depthmap validation...")
        validation_success = validate_prior_depthmaps(
            reconstruction, reference_reconstruction, all_image_ids, 
            args.resolution, args.output_folder
        )
        
        if validation_success:
            print("\n✅ Prior depthmap validation completed successfully!")
            print("   You can now run the full prediction pipeline.")
            return
        else:
            print("\n❌ Prior depthmap validation failed!")
            print("   Please fix the issues before running full predictions.")
            return
    
    # Choose batching strategy
    use_smart_batching = args.smart_batching and not args.sequential_batching
    
    print(f"🔍 DEBUG: About to batch {len(all_image_ids)} images")
    print(f"🔍 DEBUG: First 5 image IDs: {all_image_ids[:5]}")
    print(f"🔍 DEBUG: Expected 102 images with reference points")
    
    if use_smart_batching:
        print("Using smart batching based on COLMAP reconstruction quality...")
        batches = split_into_batches_smart(reconstruction, all_image_ids, args.batch_size)
        print(f"Processing {len(batches)} smart batches with max batch size {args.batch_size}")
    else:
        print("Using sequential batching...")
        batches = split_into_batches(all_image_ids, args.batch_size)
        print(f"Processing {len(batches)} sequential batches with batch size {args.batch_size}")
    
    # Debug: Print total images to be processed
    total_images_to_process = sum(len(batch) for batch in batches)
    print(f"🔍 DEBUG: Total images to be processed: {total_images_to_process}")
    print(f"🔍 DEBUG: All image IDs length: {len(all_image_ids)}")
    if reference_reconstruction is not None:
        print(f"🔍 DEBUG: Images with reference points: {len(all_image_ids)} (after filtering)")
    
    try:
        batch_files = []
        
        # Process each batch
        for batch_idx, batch_image_ids in enumerate(batches):
            print(f"\n--- Processing batch {batch_idx + 1}/{len(batches)} ---")
            print(f"🔍 DEBUG: Batch {batch_idx + 1} has {len(batch_image_ids)} images")
            
            # Load images for this batch (all images are already filtered)
            batch_images, batch_image_ids_loaded, _ = load_images_from_colmap(
                reconstruction, images_dir, args.resolution, 
                model.encoder.data_norm_type, batch_image_ids
            )
            batch_images = batch_images.to(device)
            print(f"🔍 DEBUG: Loaded {len(batch_image_ids_loaded)} images for batch {batch_idx + 1}")
            
            # Run model inference on batch
            with torch.no_grad():
                batch_predictions, batch_prior_depths = run_mapanything_inference(
                    model, batch_images, reconstruction, batch_image_ids_loaded, 
                    args.resolution, dtype, args.memory_efficient_inference, reference_reconstruction,
                    args.verbose, args.output_folder, global_image_name_mapping
                )

            # Save predicted depth maps as NPZ files for later post-processing
            save_depth_maps_as_npz(
                batch_predictions, reconstruction, batch_image_ids_loaded, 
                args.output_folder, device=device, target_size=args.resolution, prior_depths=batch_prior_depths
            )

            if args.verbose:

                # Extract point cloud from batch predictions using ground truth parameters
                batch_points_3d, batch_colors = extract_point_cloud_from_predictions(
                    batch_predictions, reconstruction, batch_image_ids_loaded, 
                    args.conf_threshold, args.max_points, use_groundtruth=True, device=device
                )
          
                # Save batch point cloud
                if len(batch_points_3d) > 0:
                    batch_file = save_batch_pointcloud(
                        batch_points_3d, batch_colors, batch_idx, args.output_folder
                    )
                    batch_files.append(batch_file)
                else:
                    print(f"Warning: Batch {batch_idx} produced no points")
                    batch_files.append(None)
            
                # Save individual prior and predicted point clouds for each image in the batch
                if args.verbose and args.output_folder is not None:
                    save_individual_pointclouds(
                        batch_predictions, batch_image_ids_loaded, reconstruction, 
                        reference_reconstruction, args.output_folder, args.max_points, device
                    )
            
                del batch_points_3d, batch_colors

            # Clear GPU memory
            del batch_images, batch_predictions
            torch.cuda.empty_cache()

        if args.verbose:
            # Filter out None batch files
            valid_batch_files = [f for f in batch_files if f is not None]
            if len(valid_batch_files) == 0:
                print("Error: No valid point clouds generated from any batch. Try lowering the confidence threshold.")
                return
            # Merge all batch point clouds
            print(f"\n--- Merging {len(valid_batch_files)} batch point clouds ---")
            total_points = merge_pointclouds_with_open3d(
                valid_batch_files, final_pointcloud_path, args.max_points
            )
            
            print(f"Successfully created final point cloud with {total_points} points at {final_pointcloud_path}")

        # Generate point cloud from NPZ files for verification
        print("\n=== Generating Point Cloud from NPZ Files for Verification ===")
        depth_maps_folder = os.path.join(args.output_folder, "depth_maps")
        
        if os.path.exists(depth_maps_folder):
            print(f"Loading NPZ files from: {depth_maps_folder}")
            
            # Load all NPZ files and generate point cloud
            npz_points, npz_colors, npz_metadata = load_multiple_npz_and_generate_pointclouds(
                depth_maps_folder, 
                conf_threshold=0.0,  # No confidence filtering for verification
                max_points_per_file=100000,  # Limit per file to avoid memory issues
                max_total_points=10000000  # Total limit
            )
            
            if len(npz_points) > 0:
                # Save NPZ-generated point cloud
                npz_cloud_path = os.path.join(args.output_folder, "npz_cloud.ply")
                
                # Create point cloud using trimesh
                npz_pc = trimesh.PointCloud(vertices=npz_points, colors=(npz_colors * 255).astype(np.uint8))
                npz_pc.export(npz_cloud_path)
                
                print(f"✅ Saved NPZ-generated point cloud: {npz_cloud_path} ({len(npz_points)} points)")
                print(f"   Generated from {len(npz_metadata)} NPZ files")
            else:
                print("❌ No valid point clouds generated from NPZ files")
        else:
            print(f"❌ Depth maps folder not found: {depth_maps_folder}")

    finally:
        print("Processing complete!")


if __name__ == "__main__":
    main()
