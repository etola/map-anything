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
from PIL import Image
from torchvision import transforms as tvf

from mapanything.models import MapAnything
from mapanything.utils.geometry import depthmap_to_world_frame
from mapanything.utils.image import rgb
from mapanything.utils.misc import seed_everything
from colmap_utils import ColmapReconstruction
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


def run_mapanything_inference(model, images, reconstruction, image_ids, target_size, dtype, memory_efficient_inference=False):
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
        
    Returns:
        List of prediction dictionaries
    """
    print("Running MapAnything inference with COLMAP camera parameters...")
    
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
        
        view = {
            "img": images[view_idx][None],  # Add batch dimension
            "data_norm_type": [model.encoder.data_norm_type],
            # Provide COLMAP camera parameters for metric reconstruction
            "intrinsics": torch.tensor(K_scaled, dtype=torch.float32).unsqueeze(0),  # Scaled intrinsics
            "camera_poses": torch.tensor(pose_4x4, dtype=torch.float32).unsqueeze(0),  # Camera-to-world pose (OpenCV convention)
            "is_metric_scale": torch.ones(1, dtype=torch.float32),  # Enable metric scale (COLMAP provides this)
        }
        views.append(view)
    
    # Run inference
    with torch.amp.autocast("cuda", dtype=dtype):
        predictions = model.infer(
            views, memory_efficient_inference=memory_efficient_inference
        )
    
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
                    args.resolution, dtype, args.memory_efficient_inference
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
