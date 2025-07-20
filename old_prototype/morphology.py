import numpy as np

def erode(mask, kernel_size=3):
    pad = kernel_size // 2
    padded = np.pad(mask, pad_width=pad, mode='constant', constant_values=0)
    H, W = mask.shape
    eroded = np.ones_like(mask)
    for i in range(kernel_size):
        for j in range(kernel_size):
            eroded = np.minimum(eroded, padded[i:i+H, j:j+W])
    return eroded

def dilate(mask, kernel_size=3):
    pad = kernel_size // 2
    padded = np.pad(mask, pad_width=pad, mode='constant', constant_values=0)
    H, W = mask.shape
    dilated = np.zeros_like(mask)
    for i in range(kernel_size):
        for j in range(kernel_size):
            dilated = np.maximum(dilated, padded[i:i+H, j:j+W])
    return dilated

def opening(mask, kernel_size=3):
    return dilate(erode(mask, kernel_size), kernel_size)

def closing(mask, kernel_size=3):
    return erode(dilate(mask, kernel_size), kernel_size)
