from PIL import Image
import numpy as np

def load_image(path):
    image = Image.open(path).convert('RGB')
    return np.array(image)

def save_image(np_img, path):
    img = Image.fromarray(np_img)
    img.save(path)

# Example usage
if __name__ == "__main__":
    img_array = load_image("test_image.jpg")
    # process your image here
    save_image(img_array, "output.png")
