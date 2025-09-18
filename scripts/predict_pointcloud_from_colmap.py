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
        "--output_path",
        type=str,
        default=None,
        help="Output path for point cloud file (default: scene_folder/predicted_pointcloud.ply)",
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
    return parser.parse_args()


def load_images_from_colmap(reconstruction, images_dir, target_size=518, data_norm_type=None):
    """
    Load images based on COLMAP reconstruction and preprocess them.
    
    Args:
        reconstruction: ColmapReconstruction object
        images_dir: Directory containing images
        target_size: Target size for image preprocessing
        data_norm_type: Image normalization type
        
    Returns:
        tuple: (images_tensor, image_ids, image_paths)
    """
    print(f"Loading images from COLMAP reconstruction...")
    
    # Get image IDs and paths from COLMAP reconstruction
    image_ids = []
    image_paths = []
    
    for image_id in reconstruction.get_all_image_ids():
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


def run_mapanything_inference(model, images, dtype, memory_efficient_inference=False):
    """
    Run MapAnything model inference on images.
    
    Args:
        model: MapAnything model
        images: Tensor of images [N, 3, H, W]
        dtype: Data type for inference
        memory_efficient_inference: Whether to use memory efficient inference
        
    Returns:
        List of prediction dictionaries
    """
    print("Running MapAnything inference...")
    
    # Prepare views for model
    views = []
    for view_idx in range(images.shape[0]):
        view = {
            "img": images[view_idx][None],  # Add batch dimension
            "data_norm_type": [model.encoder.data_norm_type],
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
    
    # Set output path
    if args.output_path is None:
        args.output_path = os.path.join(args.scene_folder, "predicted_pointcloud.ply")
    
    # Set device and dtype
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
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
    
    # Load images based on COLMAP reconstruction
    images, image_ids, image_paths = load_images_from_colmap(
        reconstruction, images_dir, args.resolution, model.encoder.data_norm_type
    )
    images = images.to(device)
    
    # Run model inference
    with torch.no_grad():
        predictions = run_mapanything_inference(
            model, images, dtype, args.memory_efficient_inference
        )
    
    # Extract point cloud from predictions
    points_3d, colors = extract_point_cloud_from_predictions(
        predictions, args.conf_threshold, args.max_points
    )
    
    if len(points_3d) == 0:
        print("Error: No points generated. Try lowering the confidence threshold.")
        return
    
    # Save point cloud
    print(f"Saving point cloud to {args.output_path}...")
    
    # Convert colors to uint8 for PLY format
    colors_uint8 = (colors * 255).astype(np.uint8)
    
    # Create and save point cloud
    point_cloud = trimesh.PointCloud(vertices=points_3d, colors=colors_uint8)
    point_cloud.export(args.output_path)
    
    print(f"Successfully saved point cloud with {len(points_3d)} points to {args.output_path}")
    
    # Print some statistics
    print("\nPoint cloud statistics:")
    print(f"  Number of points: {len(points_3d)}")
    print(f"  Bounding box min: [{points_3d.min(axis=0)[0]:.3f}, {points_3d.min(axis=0)[1]:.3f}, {points_3d.min(axis=0)[2]:.3f}]")
    print(f"  Bounding box max: [{points_3d.max(axis=0)[0]:.3f}, {points_3d.max(axis=0)[1]:.3f}, {points_3d.max(axis=0)[2]:.3f}]")


if __name__ == "__main__":
    main()
