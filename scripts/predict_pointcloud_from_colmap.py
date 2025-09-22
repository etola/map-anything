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
        default=True,
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


def colorize_depth_map(depth_map, colormap='plasma', save_path=None, depth_range=None, title_prefix="Depth Map"):
    """
    Colorize a depth map for visualization and optionally save it.
    
    Args:
        depth_map: (H, W) numpy array with depth values
        colormap: matplotlib colormap name (default: 'plasma')
        save_path: optional path to save the colorized image
        depth_range: optional tuple (min_depth, max_depth) for consistent scaling across multiple maps
        title_prefix: prefix for the plot title (default: "Depth Map")
        
    Returns:
        colorized_image: (H, W, 3) RGB array of colorized depth map
        actual_depth_range: tuple (min_depth, max_depth) of actual depth range used
    """
    # Handle case where depth map is all zeros
    if np.max(depth_map) == 0:
        # Create a black image for zero depth
        colorized = np.zeros((depth_map.shape[0], depth_map.shape[1], 3), dtype=np.uint8)
        if save_path:
            Image.fromarray(colorized).save(save_path)
        return colorized, (0.0, 0.0)
    
    # Determine depth range for normalization
    valid_mask = depth_map > 0
    if np.any(valid_mask):
        actual_min_depth = np.min(depth_map[valid_mask])
        actual_max_depth = np.max(depth_map[valid_mask])
        
        # Use provided depth range if available, otherwise use actual range
        if depth_range is not None:
            min_depth, max_depth = depth_range
            # Ensure the provided range encompasses the actual data
            min_depth = min(min_depth, actual_min_depth)
            max_depth = max(max_depth, actual_max_depth)
        else:
            min_depth, max_depth = actual_min_depth, actual_max_depth
        
        # Create normalized depth map
        normalized_depth = np.zeros_like(depth_map)
        if max_depth > min_depth:
            normalized_depth[valid_mask] = np.clip((depth_map[valid_mask] - min_depth) / (max_depth - min_depth), 0, 1)
        else:
            normalized_depth[valid_mask] = 1.0
            
        actual_range = (min_depth, max_depth)
    else:
        normalized_depth = np.zeros_like(depth_map)
        actual_range = (0.0, 0.0)
    
    # Apply colormap
    cmap = cm.get_cmap(colormap)
    colorized = cmap(normalized_depth)
    
    # Set invalid pixels to black
    colorized[~valid_mask] = [0, 0, 0, 1]
    
    # Convert to 8-bit RGB
    colorized_rgb = (colorized[:, :, :3] * 255).astype(np.uint8)
    
    # Save if path provided
    if save_path:
        # Create figure with depth information
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Original depth map
        im1 = ax1.imshow(depth_map, cmap='gray', vmin=actual_range[0], vmax=actual_range[1])
        ax1.set_title(f'{title_prefix} (Raw)\nMin: {actual_range[0]:.2f}m, Max: {actual_range[1]:.2f}m' if np.any(valid_mask) else f'{title_prefix} (Raw)\nNo valid depths')
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        
        # Colorized depth map
        ax2.imshow(colorized_rgb)
        ax2.set_title(f'{title_prefix} (Colorized)\n{np.sum(valid_mask)} valid pixels')
        ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved colorized depth map to: {save_path}")
    
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


def run_mapanything_inference(model, images, reconstruction, image_ids, target_size, dtype, memory_efficient_inference=False, reference_reconstruction=None, verbose=False, output_folder=None):
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
    """
    print("Running MapAnything inference with COLMAP camera parameters...")
    
    # Check if reference reconstruction is provided for prior depth information
    if reference_reconstruction is not None:
        print(f"Using reference reconstruction with {reference_reconstruction.get_num_images()} images and {len(reference_reconstruction.reconstruction.points3D)} 3D points for prior depth information")
        print("Note: Prior depth will be provided to the model to improve reconstruction quality")
        
        # Build image name mapping using single-image matching
        print("\n=== Matching Images Between Reconstructions ===")
        image_name_mapping = {}
        matched_count = 0
        compatible_count = 0
        
        for img_id in image_ids:
            match_result = find_image_match(
                source_reconstruction=reconstruction,
                target_reconstruction=reference_reconstruction,
                source_image_id=img_id,
                pose_tolerance_rotation=0.1,  # 0.1 radians ≈ 5.7 degrees
                pose_tolerance_translation=0.1,  # 0.1 meters = 10 cm
                verbose=False  # Pass through the verbose flag
            )
            
            if match_result['match_found']:
                image_name_mapping[img_id] = match_result['target_image_id']
                matched_count += 1
                if match_result['poses_compatible']:
                    compatible_count += 1
                    if not verbose:  # Concise output when not verbose
                        print(f"  ✓ {match_result['image_name']}: Compatible match (ID: {img_id} → {match_result['target_image_id']}) via {match_result['match_method']}")
                else:
                    if not verbose:  # Concise output when not verbose
                        print(f"  ⚠ {match_result['image_name']}: Pose mismatch (ID: {img_id} → {match_result['target_image_id']}) via {match_result['match_method']} - {match_result['pose_status']}")
            else:
                image_name_mapping[img_id] = None
                if not verbose:  # Concise output when not verbose
                    print(f"  ❌ {match_result['image_name'] or f'ID:{img_id}'}: No match found - {match_result['pose_status']}")
        
        print(f"\nMatching Summary: {compatible_count}/{matched_count}/{len(image_ids)} (compatible/matched/total)")
        print("=" * 50)
    else:
        print("No reference reconstruction provided - running without prior depth information")
        # Create empty mapping when no reference reconstruction
        image_name_mapping = {}
    
    # Keep track of depth ranges for consistent scaling
    all_depth_ranges = []
    
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
            # Check if the image has a match in reference reconstruction by name
            if image_id in image_name_mapping and image_name_mapping[image_id] is not None:
                ref_image_id = image_name_mapping[image_id]
                img_name = reconstruction.get_image_name(image_id)
                print(f"Computing prior depth for image {img_name} (Cal_ID:{image_id} -> Ref_ID:{ref_image_id})...")
                
                # Get the pose from the reference reconstruction for the matched image
                ref_cam_from_world = reference_reconstruction.get_image_cam_from_world(ref_image_id)
                ref_pose_matrix = ref_cam_from_world.matrix()  # 3x4 cam_from_world matrix from reference reconstruction
                
                prior_depth, depth_range = compute_prior_depthmap(
                    reference_reconstruction=reference_reconstruction,
                    image_id=ref_image_id,  # Use reference reconstruction's image ID
                    scaled_intrinsics=K_scaled,
                    pose_matrix=ref_pose_matrix,  # 3x4 cam_from_world matrix from reference reconstruction
                    target_size=target_size,
                    verbose=verbose,
                    output_folder=output_folder
                )
                # Collect depth range for consistent scaling
                if depth_range is not None:
                    all_depth_ranges.append(depth_range)
            else:
                img_name = reconstruction.get_image_name(image_id)
                print(f"Warning: Image {img_name} (ID:{image_id}) not found in reference reconstruction, skipping prior depth")
        
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
                # Check if image has a match in reference reconstruction by name
                if image_id in image_name_mapping and image_name_mapping[image_id] is not None:
                    ref_image_id = image_name_mapping[image_id]
                    img_name = reconstruction.get_image_name(image_id)
                    
                    # Get the correct pose from reference reconstruction
                    ref_cam_from_world = reference_reconstruction.get_image_cam_from_world(ref_image_id)
                    ref_pose_matrix = ref_cam_from_world.matrix()
                    
                    # Re-compute prior depth for consistent scaling
                    prior_depth, _ = compute_prior_depthmap(
                        reference_reconstruction=reference_reconstruction,
                        image_id=ref_image_id,  # Use reference reconstruction's image ID
                        scaled_intrinsics=views[view_idx]["intrinsics"][0].cpu().numpy(),
                        pose_matrix=ref_pose_matrix,  # Use reference reconstruction's pose matrix
                        target_size=target_size,
                        verbose=False,  # Don't save in the function, we'll save here
                        output_folder=None
                    )
                    if prior_depth is not None:
                        # Save with consistent scaling (use calibration image_id for filename)
                        save_path = os.path.join(depth_maps_folder, f"prior_depth_{image_id}_{img_name}.png")
                        _, _ = colorize_depth_map(
                            prior_depth, 
                            colormap='plasma', 
                            save_path=save_path, 
                            depth_range=consistent_depth_range,
                            title_prefix=f"Prior Depth Map ({img_name}) - Cal_ID:{image_id} → Ref_ID:{ref_image_id}"
                        )
                        print(f"Saved prior depth map: {save_path}")
                    else:
                        print(f"Warning: No prior depth computed for {img_name} (Cal_ID:{image_id} → Ref_ID:{ref_image_id})")
        
        # Save predicted depth maps
        for pred_idx, pred in enumerate(predictions):
            image_id = image_ids[pred_idx]
            img_name = reconstruction.get_image_name(image_id)
            pred_depth = pred["depth_z"][0].squeeze(-1).cpu().numpy()  # (H, W)
            
            # Save colorized predicted depth map
            save_path = os.path.join(depth_maps_folder, f"predicted_depth_{image_id}_{img_name}.png")
            _, _ = colorize_depth_map(
                pred_depth, 
                colormap='plasma', 
                save_path=save_path, 
                depth_range=consistent_depth_range,
                title_prefix=f"Predicted Depth Map ({img_name}) - Cal_ID:{image_id}"
            )
            print(f"Saved predicted depth map: {save_path}")
        
        print(f"Saved depth maps to {depth_maps_folder}")
    
    return predictions


def extract_point_cloud_from_predictions(predictions, conf_threshold=0.0, max_points=1000000):
    """
    Extract point cloud from model predictions.
    
    Args:
        predictions: List of prediction dictionaries
        conf_threshold: Confidence threshold for filtering points
        max_points: Maximum number of points to keep
        
    Returns:
        tuple: (points_3d, colors) - numpy arrays
    """
    print("Extracting point cloud from predictions...")
    
    all_points = []
    all_colors = []
    
    for view_idx, pred in enumerate(predictions):
        # Extract prediction data
        depthmap_torch = pred["depth_z"][0].squeeze(-1)  # (H, W)
        intrinsics_torch = pred["intrinsics"][0]  # (3, 3)
        camera_pose_torch = pred["camera_poses"][0]  # (4, 4)
        conf_torch = pred["conf"][0]  # (H, W)
        
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
            
            all_points.append(valid_pts)
            all_colors.append(valid_colors)
    
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
    
    # Get all image IDs and split into smart batches using COLMAP relationships
    all_image_ids = reconstruction.get_all_image_ids()
    print(f"Total images in reconstruction: {len(all_image_ids)}")
    
    # Choose batching strategy
    use_smart_batching = args.smart_batching and not args.sequential_batching
    
    if use_smart_batching:
        print("Using smart batching based on COLMAP reconstruction quality...")
        batches = split_into_batches_smart(reconstruction, all_image_ids, args.batch_size)
        print(f"Processing {len(batches)} smart batches with max batch size {args.batch_size}")
    else:
        print("Using sequential batching...")
        batches = split_into_batches(all_image_ids, args.batch_size)
        print(f"Processing {len(batches)} sequential batches with batch size {args.batch_size}")
    
    try:
        batch_files = []
        
        # Process each batch
        for batch_idx, batch_image_ids in enumerate(batches):
            print(f"\n--- Processing batch {batch_idx + 1}/{len(batches)} ---")
            
            # Load images for this batch
            batch_images, batch_image_ids_loaded, _ = load_images_from_colmap(
                reconstruction, images_dir, args.resolution, 
                model.encoder.data_norm_type, batch_image_ids
            )
            batch_images = batch_images.to(device)
            
            # Run model inference on batch
            with torch.no_grad():
                batch_predictions = run_mapanything_inference(
                    model, batch_images, reconstruction, batch_image_ids_loaded, 
                    args.resolution, dtype, args.memory_efficient_inference, reference_reconstruction,
                    args.verbose, args.output_folder
                )
            
            # Extract point cloud from batch predictions
            batch_points_3d, batch_colors = extract_point_cloud_from_predictions(
                batch_predictions, args.conf_threshold, args.max_points // len(batches)
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
            
            # Clear GPU memory
            del batch_images, batch_predictions, batch_points_3d, batch_colors
            torch.cuda.empty_cache()
        
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
        
    finally:
        print("Processing complete!")


if __name__ == "__main__":
    main()
