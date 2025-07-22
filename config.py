import torch

class Config:
    # Dataset paths
    TRAIN_IMG_DIR = "data/DUTS-TR/DUTS-TR-Image"
    TRAIN_MASK_DIR = "data/DUTS-TR/DUTS-TR-Mask"
    TEST_IMG_DIR = "data/DUTS-TE/DUTS-TE-Image"
    TEST_MASK_DIR = "data/DUTS-TE/DUTS-TE-Mask"
    
    # Model parameters
    INPUT_SIZE = 224  # 🔽 Reduced from 320
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 8
    NUM_EPOCHS = 15
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Training parameters
    SAVE_MODEL_PATH = "/content/drive/MyDrive/bg_removal_checkpoints/best_model.pth"
    RESULTS_DIR = "results"
    
    # Data augmentation
    USE_AUGMENTATION = True
