#!/usr/bin/env python3
"""
Inference script for background removal
Use this to remove backgrounds from new images
"""

import argparse
import os
import sys
from src.inference import BackgroundRemover

def main():
    parser = argparse.ArgumentParser(description='Remove background from images')
    parser.add_argument('--input', '-i', required=True, 
                       help='Input image path or folder')
    parser.add_argument('--output', '-o', 
                       help='Output path (optional)')
    parser.add_argument('--model', '-m', default='models/best_model.pth',
                       help='Path to trained model')
    parser.add_argument('--batch', action='store_true',
                       help='Process all images in input folder')
    parser.add_argument('--threshold', '-t', type=float, default=0.5,
                       help='Threshold for binary mask (0.0-1.0)')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        print("Please train the model first by running: python main.py")
        return
    
    # Initialize background remover
    print(f"Loading model from {args.model}...")
    bg_remover = BackgroundRemover(args.model)
    
    try:
        if args.batch:
            # Batch processing
            if not os.path.isdir(args.input):
                print(f"Error: {args.input} is not a directory")
                return
            
            output_dir = args.output or f"{args.input}_no_bg"
            print(f"Processing all images in {args.input}...")
            bg_remover.batch_process(args.input, output_dir)
            
        else:
            # Single image processing
            if not os.path.isfile(args.input):
                print(f"Error: {args.input} is not a file")
                return
            
            print(f"Processing {args.input}...")
            result, mask = bg_remover.remove_background(
                args.input, args.output, threshold=args.threshold
            )
            
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()