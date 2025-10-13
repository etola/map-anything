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
from typing import List

# Add parent directory to path to import mapanything modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from torchvision import transforms as tvf

from mapanything.models import MapAnything
from mapanything.utils.misc import seed_everything
from uniception.models.encoders.image_normalizations import IMAGE_NORMALIZATION_DICT

# Configure CUDA settings
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

from densification import DensificationProblem, DensificationConfig
from geometric_utility import save_point_cloud

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
        "-o", "--output_folder",
        type=str,
        default="output",
        help="Output folder for results (default: scene_folder/output/)",
    )
    parser.add_argument(
        "-s", "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "-m", "--memory_efficient_inference",
        action="store_true",
        default=False,
        help="Use memory efficient inference for reconstruction (trades off speed)",
    )
    parser.add_argument(
        "-r", "--resolution",
        type=int,
        default=518,
        help="Resolution for MapAnything model inference (default: 518)",
    )
    parser.add_argument(
        "-e", "--export_resolution",
        type=int,
        default=0,
        help="Resolution to export 3dn depthmap format (default: 0 -> disables export)",
    )
    parser.add_argument(
        "-c", "--conf_threshold",
        type=float,
        default=0.0,
        help="Confidence threshold for depth filtering (default: 0.0)",
    )
    parser.add_argument(
        "-p", "--max_points",
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
        "-b", "--batch_size",
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
        "-R", "--reference_reconstruction",
        type=str,
        default=None,
        help="Path to reference lidar fused cloud in COLMAP reconstruction format for prior depth information (default: None)",
    )
    parser.add_argument("--use_sfm_prior", action="store_true", default=False, help="Use SfM depth for prior depth information")
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose output and save colorized prior and predicted depth maps",
    )
    parser.add_argument("--min_track_length", "-t", type=int, default=3, help="Minimum track length for SfM depth to be used for prior depth information")
    return parser.parse_args()


def map_anything_depth_completion(model, depth_problem, config, image_ids) -> None:

    """
    Run MapAnything model inference on images with COLMAP camera parameters.
    """
    views = []

    if model.encoder.data_norm_type in IMAGE_NORMALIZATION_DICT.keys():
        img_norm = IMAGE_NORMALIZATION_DICT[model.encoder.data_norm_type]
        img_transform = tvf.Compose([
            tvf.ToTensor(),
            tvf.Normalize(mean=img_norm.mean, std=img_norm.std)
        ])
    else:
        img_transform = tvf.ToTensor()

    c_image_ids = []

    # setup views for model inference
    print(f"    Setting up views for model inference for {len(image_ids)} images")
    for image_id in image_ids:
        img_depth_data = depth_problem.get_depth_data(image_id)
        c_image_ids.append(image_id)

        # Convert from cam_from_world to cam2world (world_from_cam) for MapAnything
        # MapAnything expects OpenCV cam2world convention: camera coordinates -> world coordinates
        pose_4x4 = np.linalg.inv( img_depth_data['camera_pose'] )

        img = img_transform( img_depth_data['scaled_image'] )
        view = {
            "img": img[None],  # Add batch dimension
            "data_norm_type": [model.encoder.data_norm_type],
            "intrinsics": torch.tensor( img_depth_data['camera_intrinsics'], dtype=torch.float32).unsqueeze(0),  # Scaled intrinsics
            "camera_poses": torch.tensor(pose_4x4, dtype=torch.float32).unsqueeze(0),  # Camera-to-world pose (OpenCV convention)
            "is_metric_scale": torch.ones(1, dtype=torch.bool),  # Enable metric scale (COLMAP provides this) - should be bool
        }

        if img_depth_data['prior_depth_map'] is not None:
            z_depth_tensor = torch.tensor(img_depth_data['prior_depth_map'], dtype=torch.float32).contiguous()
            if torch.isnan(z_depth_tensor).any():
                print(f"    Warning: Found NaN values in prior depth for image {image_id}, setting to 0")
                z_depth_tensor = torch.nan_to_num(z_depth_tensor, nan=0.0)
            if torch.isinf(z_depth_tensor).any():
                print(f"    Warning: Found infinite values in prior depth for image {image_id}, setting to 0")
                z_depth_tensor = torch.inf_to_num(z_depth_tensor, posinf=0.0, neginf=0.0)
            z_depth_tensor = torch.clamp(z_depth_tensor, min=0.0, max=1e6)
            # The model expects z-depth in shape [1, H, W, 1] under 'depth_z' key
            view["depth_z"] = z_depth_tensor.unsqueeze(0).unsqueeze(-1)

        views.append(view)

    print(f"    Running model inference for {len(views)} views")
    with torch.amp.autocast("cuda", dtype=torch.float32):
        predictions = model.infer(
            views, memory_efficient_inference=config.memory_efficient_inference
        )

    print(f"    Updating depth maps for {len(predictions)} predictions")
    # update depth maps for each image
    for i, (prediction, image_id) in enumerate(zip(predictions, c_image_ids)):
        if prediction is None:
            continue
        depth_map = prediction['depth_z'].cpu().numpy()  # (H, W) or (1, H, W, 1)
        confidence_map = prediction['conf'].cpu().numpy()  # (H, W) or (1, H, W, 1)
        depth_problem.update_depth_data(image_id, depth_map, confidence_map)
    print(f"    Updating depth maps for {len(predictions)} predictions - done")

    del predictions

def init_model(config: DensificationConfig):
    # Initialize model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if config.apache:
        model_name = "facebook/map-anything-apache"
        print("Loading Apache 2.0 licensed MapAnything model...")
    else:
        model_name = "facebook/map-anything"
        print("Loading CC-BY-NC 4.0 licensed MapAnything model...")
    model = MapAnything.from_pretrained(model_name).to(device)
    model.eval()
    return model

def run_ma_depth_completion(problem: DensificationProblem, config: DensificationConfig, image_batches: List[List[int]] = None):
    model = init_model(config)
    for batch_idx, batch_image_ids in enumerate(image_batches):
        print(f"Processing batch {batch_idx}/{len(image_batches)} with {len(batch_image_ids)} images")
        print(f"Batch image ids: {batch_image_ids}")
        with torch.no_grad():
            map_anything_depth_completion(model, problem, config, batch_image_ids)
            torch.cuda.empty_cache()

# def run_densification(config: DensificationConfig):

#     assert config.run_depth_completion is not None, "run_depth_completion function is not set"

#     # Set seed for reproducibility
#     seed_everything(config.seed)

#     densification_problem = DensificationProblem(config.scene_folder, config.resolution, config.resolution, config.output_folder)

#     #
#     # fuse lidar dmaps;
#     #
#     if config.fuse_lidar_dmaps:
#         threedn_folder = os.path.join(config.scene_folder, "threedn")
#         if os.path.exists(threedn_folder):
#             print(f"Threedn folder found, initializing from threedn")
#             densification_problem.initialize_from_threedn(prior=False, verbose=config.verbose)
#             densification_problem.apply_fusion()
#             densification_problem.export_fused_point_cloud(file_name="lidar_fused.ply", use_parallel=True)
#         else:
#             raise RuntimeError("Threedn folder not found, cannot fuse lidar dmaps")

#     else:

#         if densification_problem.is_precomputed_depth_data_present():
#             print("Precomputed depth data found, initializing from folder")
#             densification_problem.initialize_from_folder()

#         else:

#             if config.use_sfm_prior:
#                 print("Using SfM depth for prior depth information")
#                 densification_problem.initialize_from_sfm_depth(min_track_length=config.min_track_length, verbose=config.verbose)

#             elif config.init_from_threedn_dmaps:
#                 print(f"Threedn folder found, initializing from threedn")
#                 threedn_folder = os.path.join(config.scene_folder, "threedn")
#                 densification_problem.initialize_from_threedn(prior=False, verbose=config.verbose)
#                 densification_problem.apply_fusion()
#                 densification_problem.export_fused_point_cloud(file_name="lidar_initial_fused.ply", use_parallel=True)
#                 densification_problem.transfer_fused_to_prior()
#                 densification_problem.scale_all_depth_data(config.resolution, is_square=True)
#                 densification_problem.transfer_depthmap_to_prior()

#             else:

#                 print("No threedn folder found, initializing from reference reconstruction")
#                 if config.reference_reconstruction is not None:
#                     densification_problem.initialize_with_reference(config.reference_reconstruction)

#             if config.smart_batching:
#                 print("Using smart batching based on COLMAP reconstruction quality...")
#                 image_batches = densification_problem.get_batches_geometric(config.batch_size)
#                 print(f"Processing {len(image_batches)} smart batches with max batch size {config.batch_size}")
#             else:
#                 print("Using sequential batching...")
#                 image_batches = densification_problem.get_batches_sequential(config.batch_size)
#                 print(f"Processing {len(image_batches)} sequential batches with batch size {config.batch_size}")

#             config.run_depth_completion(densification_problem, config, image_batches)

#             densification_problem.save_current_state()

#         if config.export_resolution > 0:
#             print(f"Exporting 3dn depthmaps at resolution {config.export_resolution}")
#             densification_problem.scale_all_depth_data(config.export_resolution, is_square=False)
#             densification_problem.export_dmaps(max_workers=8, verbose=config.verbose)

#         densification_problem.apply_fusion()
#         densification_problem.export_fused_point_cloud(file_name="fused.ply", use_parallel=True)

#     if config.verbose:
#         densification_problem.save_results()


def main():
    """Main function."""
    args = parse_args()

    # Print configuration
    print("Arguments:", vars(args))

    config = DensificationConfig()
    config.parse(args)

    config.run_depth_completion = run_ma_depth_completion

    problem = DensificationProblem(config)
    problem.run_densification()



if __name__ == "__main__":
    main()
