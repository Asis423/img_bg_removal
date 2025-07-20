import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from model import UNet
from dataset import get_data_loaders
from utils import calculate_iou, calculate_accuracy, save_prediction_comparison, ensure_dir

class Evaluator:
    def __init__(self, model_path, config):
        self.config = config
        self.device = config.DEVICE
        
        # Load model
        self.model = UNet().to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Model loaded from {model_path}")
    
    def evaluate_dataset(self, data_loader, save_predictions=False, num_save=10):
        """Evaluate model on dataset"""
        total_iou = 0.0
        total_acc = 0.0
        ious = []
        accs = []
        
        predictions_dir = os.path.join(self.config.RESULTS_DIR, "predictions")
        if save_predictions:
            ensure_dir(predictions_dir)
        
        with torch.no_grad():
            progress_bar = tqdm(data_loader, desc='Evaluating')
            
            for batch_idx, (images, masks) in enumerate(progress_bar):
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Calculate metrics for each image in batch
                for i in range(images.size(0)):
                    iou = calculate_iou(outputs[i:i+1], masks[i:i+1])
                    acc = calculate_accuracy(outputs[i:i+1], masks[i:i+1])
                    
                    ious.append(iou.item())
                    accs.append(acc.item())
                    
                    total_iou += iou.item()
                    total_acc += acc.item()
                    
                    # Save some predictions for visualization
                    if save_predictions and len(ious) <= num_save:
                        save_path = os.path.join(predictions_dir, f"prediction_{len(ious):03d}.png")
                        save_prediction_comparison(
                            images[i], masks[i], outputs[i], save_path
                        )
                
                # Update progress bar
                progress_bar.set_postfix({
                    'Avg IoU': f'{total_iou/len(ious):.4f}',
                    'Avg Acc': f'{total_acc/len(accs):.4f}'
                })
        
        avg_iou = total_iou / len(ious)
        avg_acc = total_acc / len(accs)
        
        return {
            'avg_iou': avg_iou,
            'avg_accuracy': avg_acc,
            'all_ious': ious,
            'all_accuracies': accs,
            'std_iou': np.std(ious),
            'std_accuracy': np.std(accs)
        }
    
    def plot_metrics_distribution(self, results, save_path):
        """Plot distribution of metrics"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # IoU distribution
        ax1.hist(results['all_ious'], bins=30, alpha=0.7, color='blue', edgecolor='black')
        ax1.axvline(results['avg_iou'], color='red', linestyle='--', 
                   label=f'Mean: {results["avg_iou"]:.4f}')
        ax1.set_title('IoU Distribution')
        ax1.set_xlabel('IoU Score')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy distribution
        ax2.hist(results['all_accuracies'], bins=30, alpha=0.7, color='green', edgecolor='black')
        ax2.axvline(results['avg_accuracy'], color='red', linestyle='--',
                   label=f'Mean: {results["avg_accuracy"]:.4f}')
        ax2.set_title('Accuracy Distribution')
        ax2.set_xlabel('Accuracy Score')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Metrics distribution saved: {save_path}")
    
    def detailed_evaluation(self, data_loader):
        """Perform detailed evaluation with statistics"""
        print("Starting detailed evaluation...")
        
        results = self.evaluate_dataset(data_loader, save_predictions=True)
        
        # Print detailed results
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(f"Number of samples: {len(results['all_ious'])}")
        print(f"Average IoU: {results['avg_iou']:.4f} ± {results['std_iou']:.4f}")
        print(f"Average Accuracy: {results['avg_accuracy']:.4f} ± {results['std_accuracy']:.4f}")
        print(f"Best IoU: {max(results['all_ious']):.4f}")
        print(f"Worst IoU: {min(results['all_ious']):.4f}")
        print(f"Best Accuracy: {max(results['all_accuracies']):.4f}")
        print(f"Worst Accuracy: {min(results['all_accuracies']):.4f}")
        
        # Calculate percentiles
        iou_percentiles = np.percentile(results['all_ious'], [25, 50, 75, 90, 95])
        acc_percentiles = np.percentile(results['all_accuracies'], [25, 50, 75, 90, 95])
        
        print("\nPercentiles:")
        print(f"IoU - 25th: {iou_percentiles[0]:.4f}, 50th: {iou_percentiles[1]:.4f}, "
              f"75th: {iou_percentiles[2]:.4f}, 90th: {iou_percentiles[3]:.4f}, "
              f"95th: {iou_percentiles[4]:.4f}")
        print(f"Acc - 25th: {acc_percentiles[0]:.4f}, 50th: {acc_percentiles[1]:.4f}, "
              f"75th: {acc_percentiles[2]:.4f}, 90th: {acc_percentiles[3]:.4f}, "
              f"95th: {acc_percentiles[4]:.4f}")
        
        # Save distribution plots
        dist_plot_path = os.path.join(self.config.RESULTS_DIR, "metrics_distribution.png")
        self.plot_metrics_distribution(results, dist_plot_path)
        
        return results

# Test evaluation
if __name__ == "__main__":
    from config import Config
    
    config = Config()
    model_path = config.SAVE_MODEL_PATH
    
    if os.path.exists(model_path):
        # Load test data
        _, test_loader = get_data_loaders(config)
        
        # Initialize evaluator
        evaluator = Evaluator(model_path, config)
        
        # Run evaluation
        results = evaluator.detailed_evaluation(test_loader)
    else:
        print(f"Model not found at {model_path}. Please train the model first.")