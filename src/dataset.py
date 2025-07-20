import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class DUTSDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_transform=None, mask_transform=None, is_train=True):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.is_train = is_train
        
        # Get all image files with corresponding mask
        self.images = []
        for img_file in os.listdir(image_dir):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                mask_name = img_file.rsplit('.', 1)[0] + '.png'
                if os.path.exists(os.path.join(mask_dir, mask_name)):
                    self.images.append(img_file)
        
        print(f"Found {len(self.images)} image-mask pairs")
        
        # Data augmentation for training
        if is_train:
            self.augment_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            ])
        else:
            self.augment_transform = None
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_name = img_name.rsplit('.', 1)[0] + '.png'
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        # Load images
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # grayscale mask
        
        # Apply same augmentation to image and mask if training
        if self.is_train and self.augment_transform:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
            torch.manual_seed(seed)
            image = self.augment_transform(image)
            torch.manual_seed(seed)
            mask = self.augment_transform(mask)
        
        # Apply separate transforms
        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        
        # Binarize mask (ensure values are 0 or 1)
        mask = (mask > 0.5).float()
        
        return image, mask


def get_data_loaders(config):
    """Create train and test data loaders"""
    
    # Image transform (with normalization)
    image_transform = transforms.Compose([
        transforms.Resize((config.INPUT_SIZE, config.INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Mask transform (no normalization)
    mask_transform = transforms.Compose([
        transforms.Resize((config.INPUT_SIZE, config.INPUT_SIZE)),
        transforms.ToTensor(),  # just convert to tensor
    ])
    
    # Create datasets with separate image and mask transforms
    train_dataset = DUTSDataset(
        config.TRAIN_IMG_DIR,
        config.TRAIN_MASK_DIR,
        image_transform=image_transform,
        mask_transform=mask_transform,
        is_train=True
    )
    
    test_dataset = DUTSDataset(
        config.TEST_IMG_DIR,
        config.TEST_MASK_DIR,
        image_transform=image_transform,
        mask_transform=mask_transform,
        is_train=False
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, test_loader


# Optional test code
if __name__ == "__main__":
    from config import Config
    config = Config()
    train_loader, test_loader = get_data_loaders(config)
    
    for batch_idx, (images, masks) in enumerate(train_loader):
        print(f"Batch {batch_idx}: Images {images.shape}, Masks {masks.shape}")
        if batch_idx == 2:
            break
