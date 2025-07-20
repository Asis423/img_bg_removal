from PIL import Image
import numpy as np

def load_image(path):
    image = Image.open(path).convert('RGB')
    return np.array(image)

def save_image_with_alpha(rgba_array, output_path):
    img = Image.fromarray(rgba_array, mode='RGBA')
    img.save(output_path)

def estimate_background_color_range(image, border_width=10, tolerance=1.5):
    top = image[0:border_width, :, :]
    bottom = image[-border_width:, :, :]
    left = image[:, 0:border_width, :]
    right = image[:, -border_width:, :]
    
    border_pixels = np.concatenate((
        top.reshape(-1, 3),
        bottom.reshape(-1, 3),
        left.reshape(-1, 3),
        right.reshape(-1, 3)
    ), axis=0)
    
    mean_color = np.mean(border_pixels, axis=0)
    std_color = np.std(border_pixels, axis=0)
    
    lower_bound = np.clip(mean_color - tolerance * std_color, 0, 255).astype(np.uint8)
    upper_bound = np.clip(mean_color + tolerance * std_color, 0, 255).astype(np.uint8)
    
    return lower_bound, upper_bound


def color_based_segmentation(image, lower_rgb, upper_rgb):
    mask_bg = np.all((image >= lower_rgb) & (image <= upper_rgb), axis=2)
    mask_fg = np.logical_not(mask_bg).astype(np.uint8)
    return mask_fg

def apply_alpha_mask(image, alpha_mask):
    h, w = alpha_mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    # Normalize alpha to [0, 1]
    alpha_normalized = alpha_mask.astype(np.float32) / 255.0
    alpha_3c = np.stack([alpha_normalized]*3, axis=-1)

    # Fade RGB based on alpha (to eliminate white edge)
    faded_rgb = image.astype(np.float32) * alpha_3c
    rgba[..., :3] = faded_rgb.astype(np.uint8)
    rgba[..., 3] = alpha_mask

    return rgba


from scipy.ndimage import gaussian_filter

def feather_mask(mask, sigma=2.0):
    # Convert binary mask to float
    mask = mask.astype(np.float32)

    # Smooth the edges using Gaussian blur
    blurred = gaussian_filter(mask, sigma=sigma)

    # Normalize values to range [0, 255]
    blurred = (blurred - blurred.min()) / (blurred.max() - blurred.min() + 1e-8)
    return (blurred * 255).astype(np.uint8)

