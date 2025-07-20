import numpy as np
from image_utils import load_image, save_image_with_alpha, estimate_background_color_range, color_based_segmentation, apply_alpha_mask
from morphology import opening, closing, dilate
from image_utils import load_image, save_image_with_alpha, estimate_background_color_range, color_based_segmentation, apply_alpha_mask, feather_mask


def remove_background(input_path, output_path, border_width=10, tolerance=1.5, kernel_size=5, morph_iterations=2):
    image = load_image(input_path)
    
    # Automatically estimate background color range
    lower_bg, upper_bg = estimate_background_color_range(image, border_width, tolerance)
    print(f"Estimated background color range: Lower={lower_bg}, Upper={upper_bg}")
    
    mask = color_based_segmentation(image, lower_bg, upper_bg)
    
    # Morphological operations to clean the mask
    # Morphological operations to clean the mask
    for _ in range(morph_iterations):
        mask = opening(mask, kernel_size)
        mask = closing(mask, kernel_size)

    # Optional: Dilate slightly to shrink white outline effect
    mask = dilate(mask, kernel_size=3)

    # Smooth edges (feathering)
    mask = feather_mask(mask, sigma=2.0)  # You can tweak sigma


    rgba = apply_alpha_mask(image, mask)
    save_image_with_alpha(rgba, output_path)
    print(f"Background removal complete. Output saved as {output_path}")

if __name__ == "__main__":
    remove_background(
        input_path="test_image1.jpg",      # Put your input image here
        output_path="output.png",    # Output file with transparent bg
        border_width=10,
        tolerance=1.5,
        kernel_size=5,
        morph_iterations=2
    )
