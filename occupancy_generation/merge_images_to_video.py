#!/usr/bin/env python3
"""
Script to merge PNG images into MP4 videos, separating gt_occ and pred images.
"""

import os
import glob
import re
import cv2
import numpy as np
from pathlib import Path
import argparse


def natural_sort_key(filename):
    """Extract numeric part for natural sorting"""
    match = re.match(r'(\d+)_', filename)
    return int(match.group(1)) if match else 0


def create_video_from_images(image_paths, output_path, fps=10):
    """Create MP4 video from a list of image paths"""
    if not image_paths:
        print(f"No images found for {output_path}")
        return False
    
    # Read first image to get dimensions
    first_img = cv2.imread(image_paths[0])
    if first_img is None:
        print(f"Error reading first image: {image_paths[0]}")
        return False
    
    height, width, channels = first_img.shape
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Creating video: {output_path}")
    print(f"Video dimensions: {width}x{height}")
    print(f"Number of frames: {len(image_paths)}")
    
    for i, img_path in enumerate(image_paths):
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image {img_path}")
            continue
        
        # Resize if necessary (in case images have different sizes)
        if img.shape[:2] != (height, width):
            img = cv2.resize(img, (width, height))
        
        out.write(img)
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(image_paths)} images")
    
    out.release()
    print(f"Video saved: {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description='Merge PNG images into MP4 videos')
    parser.add_argument('--input_dir', type=str, 
                        default='outputs/occ_generation/save_occ_vis',
                        help='Input directory containing PNG images')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for videos (default: same as input_dir)')
    parser.add_argument('--fps', type=int, default=10,
                        help='Frames per second for output videos')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all PNG files
    png_files = list(input_dir.glob("*.png"))
    
    if not png_files:
        print(f"No PNG files found in {input_dir}")
        return
    
    print(f"Found {len(png_files)} PNG files")
    
    # Separate gt_occ and pred files
    gt_occ_files = []
    pred_files = []
    
    for png_file in png_files:
        filename = png_file.name
        if '_gt_occ.npy_' in filename:
            gt_occ_files.append(str(png_file))
        elif '_pred.npy_' in filename:
            pred_files.append(str(png_file))
    
    print(f"Found {len(gt_occ_files)} gt_occ files")
    print(f"Found {len(pred_files)} pred files")
    
    # Sort files by numerical index
    gt_occ_files.sort(key=lambda x: natural_sort_key(os.path.basename(x)))
    pred_files.sort(key=lambda x: natural_sort_key(os.path.basename(x)))
    
    # Create videos
    gt_occ_output = output_dir / "gt_occ_video.mp4"
    pred_output = output_dir / "pred_video.mp4"
    
    print("\n" + "="*50)
    print("Creating GT_OCC video...")
    success_gt = create_video_from_images(gt_occ_files, str(gt_occ_output), args.fps)
    
    print("\n" + "="*50)
    print("Creating PRED video...")
    success_pred = create_video_from_images(pred_files, str(pred_output), args.fps)
    
    print("\n" + "="*50)
    print("Summary:")
    if success_gt:
        print(f"✓ GT_OCC video created: {gt_occ_output}")
    else:
        print("✗ Failed to create GT_OCC video")
    
    if success_pred:
        print(f"✓ PRED video created: {pred_output}")
    else:
        print("✗ Failed to create PRED video")
    
    print(f"\nVideos saved in: {output_dir}")


if __name__ == "__main__":
    main()
