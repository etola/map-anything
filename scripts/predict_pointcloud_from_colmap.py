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
from colmap_utils import ColmapReconstruction, build_image_id_mapping
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


def compute_prior_depthmap(reference_reconstruction, image_id, scaled_intrinsics, pose_matrix, target_size, min_track_length):
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




class DepthData:
    """
    Depth data class for storing depth related data.
    """
    
    def __init__(self, 
                 scene_folder: str,
                 reconstruction: ColmapReconstruction, 
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
        self.reconstruction = reconstruction
        self.scene_folder = scene_folder
        self.reference_reconstruction = None
        self.target_size = target_size
        self.output_folder = os.path.join(output_folder, "depth_maps")
        os.makedirs(output_folder, exist_ok=True)

        # stores all the depth data for each image
        self.scene_depth_data = {}

        self.active_image_ids = reconstruction.get_all_image_ids()
        self.source_to_target_image_id_mapping = {}

    def get_depth_data(self, image_id: int) -> dict:
        if image_id not in self.scene_depth_data:
            self.initialize_depth_data(image_id)
        return self.scene_depth_data[image_id]

    def initialize_from_folder(self):
        import glob
        depth_data_files = glob.glob(os.path.join(self.output_folder, "*.npz"))
        for df in depth_data_files:
            data = np.load(df)
            image_id = int(data['image_id'])
            self.scene_depth_data[image_id] = data

    def initialize_with_reference(self, reference_reconstruction: ColmapReconstruction) -> None:

        self.reference_reconstruction = reference_reconstruction
        if self.reference_reconstruction is None:
            raise ValueError("Reference reconstruction is required to initialize with reference")

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
            self.initialize_prior_depth_data(img_id)
            self.initialize_scaled_image(img_id)

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

    def initialize_prior_depth_data(self, image_id: int) -> None:
        if self.reference_reconstruction is None:
            return
        depth_data = self.get_depth_data(image_id)
        ref_image_id = self.source_to_target_image_id_mapping[image_id]
        assert ref_image_id is not None, f"No target image id found for image {image_id}"
        cam_from_world = depth_data['camera_pose'][:3, :]
        prior_depth_map, depth_range = compute_prior_depthmap(self.reference_reconstruction, ref_image_id, depth_data['camera_intrinsics'], cam_from_world, self.target_size, min_track_length=1)
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
        filepath = os.path.join(self.output_folder, f"depth_{image_id:06d}.npz")
        np.savez_compressed(filepath, **depth_data)

    def save(self) -> None:
        for img_id in self.scene_depth_data:
            self.save_depth_data(img_id)

    def load_prior_depth_data(self, image_id: int) -> None:
        filepath = os.path.join(self.output_folder, f"depth_{image_id:06d}.npz")
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

        # depth_range = depth_data['depth_range']
        # confidence_range = depth_data['confidence_range']
        # print(f"Loaded depth map for {image_name} (ID: {image_id})")
        # print(f"Depth map shape: {depth_map.shape}")
        # print(f"Depth range: {depth_range[0]:.3f} - {depth_range[1]:.3f}")
        # print(f"Confidence range: {confidence_range[0]:.3f} - {confidence_range[1]:.3f}")
        
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
        
        cam_to_world = np.linalg.inv(cam_from_world)

        # Convert to torch tensors
        depthmap_torch   = torch.tensor(depth_map_filtered, dtype=torch.float32)
        intrinsics_torch = torch.tensor(camera_intrinsics, dtype=torch.float32)
        pose_torch       = torch.tensor(cam_to_world, dtype=torch.float32)
        
        # Compute 3D points from depth using the saved camera parameters
        result = depthmap_to_world_frame(depthmap_torch, intrinsics_torch, pose_torch)
        
        # Handle different return formats
        if len(result) == 2:
            pts3d, valid_mask = result
        else:
            print(f"Warning: depthmap_to_world_frame returned {len(result)} values, expected 2")
            pts3d, valid_mask = result[0], result[1]
        
        # Convert to numpy
        pts3d_np = pts3d.cpu().numpy()
        valid_mask_np = valid_mask.cpu().numpy()

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

    def initialize_batches_geometric(self, batch_size: int) -> None:
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

    def initialize_batches_sequential(self, batch_size: int) -> None:
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

            # Clear GPU memory
            del batch_images, batch_predictions
            torch.cuda.empty_cache()

    finally:
        print("Processing complete!")


if __name__ == "__main__":
    main()
