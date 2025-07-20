import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2
from model import UNet
from utils import apply_morphological_operations, create_transparent_background, ensure_dir
import os

class BackgroundRemover:
    def __init__(self, model_path, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        self.model = UNet()
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing transform
        self.transform = transforms.Compose([
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Model loaded successfully on {self.device}")
    
    def predict_mask(self, image_path, apply_morphology=True, smooth_edges=True):
        """Predict mask for a single image"""
        # Load and preprocess image
        original_image = Image.open(image_path).convert('RGB')
        original_size = original_image.size
        
        # Transform for model input
        input_tensor = self.transform(original_image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            pred_mask = self.model(input_tensor)
            pred_mask = pred_mask.squeeze().cpu().numpy()
        
        # Resize mask back to original size
        pred_mask_resized = cv2.resize(pred_mask, original_size, interpolation=cv2.INTER_LINEAR)
        
        # Apply morphological operations
        if apply_morphology:
            pred_mask_resized = apply_morphological_operations(pred_mask_resized, kernel_size=3)
        
        # Smooth edges with Gaussian blur
        if smooth_edges:
            pred_mask_resized = cv2.GaussianBlur(pred_mask_resized, (3, 3), 0)
        
        return np.array(original_image), pred_mask_resized
    
    def remove_background(self, image_path, output_path=None, threshold=0.5):
        """Remove background from image and save as PNG with transparency"""
        # Get prediction
        original_image, pred_mask = self.predict_mask(image_path)
        
        # Apply threshold
        binary_mask = (pred_mask > threshold).astype(np.float32)
        
        # Create transparent background image
        result_image = create_transparent_background(
            original_image, binary_mask, blur_edges=True, blur_radius=1
        )
        
        # Save result
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = f"{base_name}_no_bg.png"
        
        result_pil = Image.fromarray(result_image, 'RGBA')
        result_pil.save(output_path, 'PNG')
        
        print(f"Background removed and saved: {output_path}")
        return result_image, binary_mask
    
    def batch_process(self, input_folder, output_folder, file_extensions=('.jpg', '.jpeg', '.png')):
        """Process multiple images in a folder"""
        ensure_dir(output_folder)
        
        processed_count = 0
        for filename in os.listdir(input_folder):
            if filename.lower().endswith(file_extensions):
                input_path = os.path.join(input_folder, filename)
                output_filename = os.path.splitext(filename)[0] + "_no_bg.png"
                output_path = os.path.join(output_folder, output_filename)
                
                try:
                    self.remove_background(input_path, output_path)
                    processed_count += 1
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
        
        print(f"Batch processing completed. Processed {processed_count} images.")

# Test inference
if __name__ == "__main__":
    model_path = "models/best_model.pth"
    
    if os.path.exists(model_path):
        bg_remover = BackgroundRemover(model_path)
        
        # Test with a sample image (you need to provide this)
        test_image = "test_image.jpg"  # Replace with actual test image path
        if os.path.exists(test_image):
            result, mask = bg_remover.remove_background(test_image)
            print("Background removal test completed!")
    else:
        print(f"Model not found at {model_path}. Please train the model first.")