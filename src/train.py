import torch
import os
from tqdm import tqdm
from src.utils import save_checkpoint  # Make sure you have this function in utils.py or define it here

class Trainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        self.device = self.config.DEVICE
        self.model.to(self.device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.LEARNING_RATE)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        
        self.start_epoch = 0  # Will update if loading checkpoint

    def load_checkpoint(self, filepath):
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.start_epoch = checkpoint['epoch']
            print(f"Checkpoint loaded, resuming from epoch {self.start_epoch}")
        else:
            print("No checkpoint found, starting from scratch.")

    def train(self):
        best_val_loss = float('inf')
        
        # Try to load checkpoint if exists
        self.load_checkpoint(self.config.SAVE_MODEL_PATH)

        train_losses = []
        val_losses = []

        for epoch in range(self.start_epoch, self.config.NUM_EPOCHS):
            self.model.train()
            train_loss = 0.0

            for image, mask in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.NUM_EPOCHS}"):
                image, mask = image.to(self.device), mask.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(image)
                loss = self.criterion(output, mask)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(self.train_loader)
            train_losses.append(avg_train_loss)

            val_loss = self.validate()
            val_losses.append(val_loss)

            print(f"Epoch [{epoch+1}/{self.config.NUM_EPOCHS}], Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                os.makedirs(os.path.dirname(self.config.SAVE_MODEL_PATH), exist_ok=True)
                save_checkpoint(self.model, self.optimizer, epoch + 1, val_loss, self.config.SAVE_MODEL_PATH)
                print("Best model saved!")

            # Optionally save checkpoint every few epochs (uncomment if desired)
            # if (epoch + 1) % 5 == 0:
            #     save_checkpoint(self.model, self.optimizer, epoch + 1, avg_train_loss, f"checkpoints/checkpoint_epoch_{epoch+1}.pth")

        return train_losses, val_losses

    def validate(self):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for image, mask in self.val_loader:
                image, mask = image.to(self.device), mask.to(self.device)
                output = self.model(image)
                loss = self.criterion(output, mask)
                val_loss += loss.item()

        return val_loss / len(self.val_loader)


def save_checkpoint(model, optimizer, epoch, loss, filepath):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved at epoch {epoch} to {filepath}")

def load_checkpoint(self, filepath):
    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
        print(f"Loading checkpoint from {filepath}")
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        print(f"Checkpoint loaded, resuming from epoch {self.start_epoch}")
    else:
        print(f"No valid checkpoint found at {filepath}, starting from scratch.")