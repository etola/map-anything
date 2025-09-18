# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
MapAnything Demo: Images with COLMAP Calibration and Visualization

This script extends demo_images_only_inference.py to use calibration data from 
a COLMAP reconstruction for more accurate metric reconstruction.

Usage:
    python demo_images_with_colmap_calibration.py --help
"""

import argparse
import os
import sys

# Add parent directory to path to import mapanything modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import rerun as rr
import torch

from mapanything.models import MapAnything
from mapanything.utils.geometry import depthmap_to_world_frame
from mapanything.utils.viz import (
    predictions_to_glb,
    script_add_rerun_args,
)
from colmap_utils import ColmapReconstruction


def log_data_to_rerun(
    image, depthmap, pose, intrinsics, pts3d, mask, base_name, pts_name, viz_mask=None
):
    """Log visualization data to Rerun"""
    # Log camera info and loaded data
    height, width = image.shape[0], image.shape[1]
    rr.log(
        base_name,
        rr.Transform3D(
            translation=pose[:3, 3],
            mat3x3=pose[:3, :3],
        ),
    )
    rr.log(
        f"{base_name}/pinhole",
        rr.Pinhole(
            image_from_camera=intrinsics,
            height=height,
            width=width,
            camera_xyz=rr.ViewCoordinates.RDF,
            image_plane_distance=1.0,
        ),
    )
    rr.log(
        f"{base_name}/pinhole/rgb",
        rr.Image(image),
    )
    rr.log(
        f"{base_name}/pinhole/depth",
        rr.DepthImage(depthmap),
    )
    if viz_mask is not None:
        rr.log(
            f"{base_name}/pinhole/mask",
            rr.SegmentationImage(viz_mask.astype(int)),
        )

    # Log points in 3D
    filtered_pts = pts3d[mask]
    filtered_pts_col = image[mask]

    rr.log(
        pts_name,
        rr.Points3D(
            positions=filtered_pts.reshape(-1, 3),
            colors=filtered_pts_col.reshape(-1, 3),
        ),
    )


def load_images_from_colmap(reconstruction, images_dir):
    """
    Load images based on COLMAP reconstruction using MapAnything's load_images function.
    
    Args:
        reconstruction: ColmapReconstruction object
        images_dir: Directory containing images
        
    Returns:
        tuple: (views, image_ids, image_paths)
    """
    print(f"Loading images from COLMAP reconstruction...")
    all_image_ids = reconstruction.get_all_image_ids()
    
    # Get image IDs and paths from COLMAP reconstruction in order
    image_paths = []
    valid_image_ids = []
    
    for image_id in all_image_ids:
        image_name = reconstruction.get_image_name(image_id)
        image_path = os.path.join(images_dir, image_name)
        
        if not os.path.exists(image_path):
            print(f"Warning: Image {image_path} not found, skipping...")
            continue
            
        valid_image_ids.append(image_id)
        image_paths.append(image_path)
    
    if len(image_paths) == 0:
        raise ValueError(f"No valid images found in {images_dir}")
    
    print(f"Loading {len(image_paths)} images using MapAnything's load_images...")
    
    # Use MapAnything's load_images function for proper preprocessing
    from mapanything.utils.image import load_images
    views = load_images(image_paths)
    
    return views, valid_image_ids, image_paths


def inject_colmap_calibration_into_views(views, reconstruction, image_ids):
    """
    Inject COLMAP calibration data into preprocessed MapAnything views.
    
    Args:
        views: List of preprocessed view dictionaries from load_images
        reconstruction: ColmapReconstruction object
        image_ids: List of COLMAP image IDs corresponding to the views
        
    Returns:
        Tuple of (enhanced_views, colmap_calibrations) where:
        - enhanced_views: List of views with COLMAP calibration data injected
        - colmap_calibrations: List of dicts with original COLMAP calibration for each view
    """
    print("Injecting COLMAP calibration data into views...")
    
    # Get the target resolution from the first view
    target_size = views[0]["img"].shape[-1]  # Assuming square images
    
    enhanced_views = []
    colmap_calibrations = []
    
    for view_idx, view in enumerate(views):
        image_id = image_ids[view_idx]
        
        # Start with the original view
        enhanced_view = dict(view)
        
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
        
        # Add COLMAP calibration data to the view
        enhanced_view["intrinsics"] = torch.tensor(K_scaled, dtype=torch.float32).unsqueeze(0)
        enhanced_view["camera_poses"] = torch.tensor(pose_4x4, dtype=torch.float32).unsqueeze(0)
        enhanced_view["is_metric_scale"] = torch.ones(1, dtype=torch.bool)  # Use bool for metric scale flag
        
        # Store the original COLMAP calibration for point cloud computation
        colmap_calibration = {
            "intrinsics": torch.tensor(K_scaled, dtype=torch.float32),
            "camera_pose": torch.tensor(pose_4x4, dtype=torch.float32),
            "image_id": image_id,
        }
        
        enhanced_views.append(enhanced_view)
        colmap_calibrations.append(colmap_calibration)
    
    return enhanced_views, colmap_calibrations


def run_mapanything_with_colmap(model, views, reconstruction, image_ids, memory_efficient_inference=False):
    """
    Run MapAnything model inference with COLMAP camera parameters.
    
    Args:
        model: MapAnything model
        views: List of preprocessed view dictionaries from load_images
        reconstruction: ColmapReconstruction object
        image_ids: List of COLMAP image IDs corresponding to the views
        memory_efficient_inference: Whether to use memory efficient inference
        
    Returns:
        Tuple of (predictions, colmap_calibrations) where:
        - predictions: List of prediction dictionaries from the model
        - colmap_calibrations: List of COLMAP calibration data for point cloud computation
    """
    print("Running MapAnything inference with COLMAP camera parameters...")
    
    # Inject COLMAP calibration data into the views and get calibration data
    enhanced_views, colmap_calibrations = inject_colmap_calibration_into_views(views, reconstruction, image_ids)
    
    # Run inference with proper settings (matching working scripts)
    predictions = model.infer(
        enhanced_views, 
        memory_efficient_inference=memory_efficient_inference,
        use_amp=True,  # Use automatic mixed precision like in working scripts
        amp_dtype="bf16",  # Use bfloat16 for best results
        apply_mask=True,  # Apply masking to outputs
        mask_edges=True,  # Remove edge artifacts
    )
    
    return predictions, colmap_calibrations


def get_parser():
    """Create argument parser"""
    parser = argparse.ArgumentParser(
        description="MapAnything Demo: Visualize metric 3D reconstruction from images using COLMAP calibration"
    )
    parser.add_argument(
        "--scene_folder",
        type=str,
        required=True,
        help="Path to scene folder containing 'images' and 'sparse' (COLMAP reconstruction) subdirectories",
    )
    parser.add_argument(
        "--memory_efficient_inference",
        action="store_true",
        default=False,
        help="Use memory efficient inference for reconstruction (trades off speed)",
    )
    parser.add_argument(
        "--apache",
        action="store_true",
        help="Use Apache 2.0 licensed model (facebook/map-anything-apache)",
    )
    parser.add_argument(
        "--viz",
        action="store_true",
        default=False,
        help="Enable visualization with Rerun",
    )
    parser.add_argument(
        "--save_glb",
        action="store_true",
        default=False,
        help="Save reconstruction as GLB file",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="output_with_colmap.glb",
        help="Output path for GLB file (default: output_with_colmap.glb)",
    )

    return parser


def main():
    # Parser for arguments and Rerun
    parser = get_parser()
    script_add_rerun_args(
        parser
    )  # Options: --headless, --connect, --serve, --addr, --save, --stdout
    args = parser.parse_args()

    # Validate scene folder structure
    if not os.path.isdir(args.scene_folder):
        raise ValueError(f"Scene folder {args.scene_folder} does not exist")
    
    images_dir = os.path.join(args.scene_folder, "images")
    sparse_dir = os.path.join(args.scene_folder, "sparse")
    
    if not os.path.isdir(images_dir):
        raise ValueError(f"Images directory {images_dir} does not exist")
    
    if not os.path.isdir(sparse_dir):
        raise ValueError(f"Sparse directory {sparse_dir} does not exist")

    # Get inference device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize model from HuggingFace
    if args.apache:
        model_name = "facebook/map-anything-apache"
        print("Loading Apache 2.0 licensed MapAnything model...")
    else:
        model_name = "facebook/map-anything"
        print("Loading CC-BY-NC 4.0 licensed MapAnything model...")
    model = MapAnything.from_pretrained(model_name).to(device)
    model.eval()  # Set to evaluation mode

    # Load COLMAP reconstruction
    print(f"Loading COLMAP reconstruction from {sparse_dir}...")
    reconstruction = ColmapReconstruction(sparse_dir)
    print(f"Loaded reconstruction with {reconstruction.get_num_images()} images")

    # Load images based on COLMAP reconstruction using MapAnything's load_images
    print(f"Loading images from: {images_dir}")
    views, image_ids, image_paths = load_images_from_colmap(reconstruction, images_dir)
    print(f"Loaded {len(views)} views")

    # Run model inference with COLMAP calibration
    print("Running inference with COLMAP calibration...")
    with torch.no_grad():
        outputs, colmap_calibrations = run_mapanything_with_colmap(
            model, views, reconstruction, image_ids, args.memory_efficient_inference
        )
    print("Inference complete!")

    # Prepare lists for GLB export if needed
    world_points_list = []
    images_list = []
    masks_list = []

    # Initialize Rerun if visualization is enabled
    if args.viz:
        print("Starting visualization...")
        viz_string = "MapAnything_COLMAP_Visualization"
        rr.script_setup(args, viz_string)
        rr.set_time("stable_time", sequence=0)
        rr.log("mapanything", rr.ViewCoordinates.RDF, static=True)

    # Loop through the outputs
    for view_idx, pred in enumerate(outputs):
        image_id = image_ids[view_idx]
        colmap_cal = colmap_calibrations[view_idx]
        
        # Extract predicted depth from model but use COLMAP calibration for point cloud computation
        depthmap_torch = pred["depth_z"][0].squeeze(-1)  # (H, W) - from model prediction
        
        # Use COLMAP calibration data instead of predicted calibration
        # Ensure tensors are on the same device as the depth map
        device = depthmap_torch.device
        intrinsics_torch = colmap_cal["intrinsics"].to(device)  # (3, 3) - from COLMAP, moved to GPU
        camera_pose_torch = colmap_cal["camera_pose"].to(device)  # (4, 4) - from COLMAP, moved to GPU
        
        print(f"View {view_idx}: Using COLMAP calibration for image ID {image_id}")

        # Compute new pts3d using PREDICTED depth with COLMAP intrinsics and camera pose
        pts3d_computed, valid_mask = depthmap_to_world_frame(
            depthmap_torch, intrinsics_torch, camera_pose_torch
        )

        # Convert to numpy arrays
        mask = pred["mask"][0].squeeze(-1).cpu().numpy().astype(bool)
        mask = mask & valid_mask.cpu().numpy()  # Combine with valid depth mask
        pts3d_np = pts3d_computed.cpu().numpy()
        image_np = pred["img_no_norm"][0].cpu().numpy()

        # Store data for GLB export if needed
        if args.save_glb:
            world_points_list.append(pts3d_np)
            images_list.append(image_np)
            masks_list.append(mask)

        # Log to Rerun if visualization is enabled
        if args.viz:
            # Get COLMAP image name for better labeling
            image_name = reconstruction.get_image_name(image_id)
            log_data_to_rerun(
                image=image_np,
                depthmap=depthmap_torch.cpu().numpy(),
                pose=camera_pose_torch.cpu().numpy(),  # Using COLMAP pose for visualization
                intrinsics=intrinsics_torch.cpu().numpy(),  # Using COLMAP intrinsics for visualization
                pts3d=pts3d_np,
                mask=mask,
                base_name=f"mapanything/colmap_view_{image_id}_{image_name}",
                pts_name=f"mapanything/pointcloud_view_{image_id}",
                viz_mask=mask,
            )

    if args.viz:
        print("Visualization complete! Check the Rerun viewer.")

    # Export GLB if requested
    if args.save_glb:
        print(f"Saving GLB file to: {args.output_path}")

        # Stack all views
        world_points = np.stack(world_points_list, axis=0)
        images = np.stack(images_list, axis=0)
        final_masks = np.stack(masks_list, axis=0)

        # Create predictions dict for GLB export
        predictions = {
            "world_points": world_points,
            "images": images,
            "final_masks": final_masks,
        }

        # Convert to GLB scene
        scene_3d = predictions_to_glb(predictions, as_mesh=True)

        # Save GLB file
        scene_3d.export(args.output_path)
        print(f"Successfully saved GLB file: {args.output_path}")
    else:
        print("Skipping GLB export (--save_glb not specified)")

    # Print summary information about the reconstruction
    print("\n=== COLMAP Reconstruction Summary ===")
    summary = reconstruction.get_summary()
    print(f"Number of images: {summary['num_images']}")
    print(f"Number of 3D points: {summary['num_points_3d']}")
    print(f"Number of cameras: {summary['num_cameras']}")
    print(f"Average track length: {summary['avg_track_length']:.2f}")


if __name__ == "__main__":
    main()
