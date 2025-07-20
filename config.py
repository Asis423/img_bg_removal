import torch

class Config:
    # Dataset paths
    TRAIN_IMG_DIR = "data/DUTS-TR/DUTS-TR-Image"
    TRAIN_MASK_DIR = "data/DUTS-TR/DUTS-TR-Mask"
    TEST_IMG_DIR = "data/DUTS-TE/DUTS-TE-Image"
    TEST_MASK_DIR = "data/DUTS-TE/DUTS-TE-Mask"
    
    # Model parameters
    INPUT_SIZE = 320
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 8
    NUM_EPOCHS = 50
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Training parameters
    SAVE_MODEL_PATH = "models/best_model.pth"
    RESULTS_DIR = "results"
    
    # Data augmentation
    USE_AUGMENTATION = True

# Optional: print device when running config directly
if __name__ == "__main__":
    print(f"Using device: {Config.DEVICE}")
