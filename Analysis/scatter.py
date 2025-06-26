""" This script computes sequence difficulty based on model loss and plots the distribution.
It evaluates the model's performance on training and validation datasets, computes losses for each sequence,
and generates visualizations to analyze the distribution of losses."""

import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import TensorDataset
from model import MusicTransformer
from hparams import device, hparams
from masking import create_mask
import argparse

# Add TensorDataset to safe globals if using PyTorch 2.6+
torch.serialization.add_safe_globals([TensorDataset])

def loss_fn(prediction, target, criterion=torch.nn.functional.cross_entropy):
    """
    Computes masked loss, ignoring padding values.
    """
    mask = torch.ne(target, torch.zeros_like(target))
    _loss = criterion(prediction, target, reduction='none')
    
    mask = mask.to(_loss.dtype)
    _loss *= mask
    
    return torch.sum(_loss) / torch.sum(mask)

def compute_losses(model, dataset, name="dataset"):
    """
    Compute loss for each sequence in the dataset
    """
    model.eval()
    losses = []
    indices = []
    
    print(f"Computing losses for {name}...")
    print(f"Total number of sequences: {len(dataset)}")
    
    with torch.no_grad():
        for i, (inp, tar) in enumerate(dataset):
            inp, tar = inp.unsqueeze(0).to(device), tar.unsqueeze(0).to(device)
            mask = create_mask(inp, n=inp.dim() + 2)
            predictions = model(inp, mask=mask)
            loss = loss_fn(predictions.transpose(-1, -2), tar)
            losses.append(loss.item())
            indices.append(i)
            
            # Print progress every 100 examples
            if (i + 1) % 100 == 0:
                print(f"Processed {i+1}/{len(dataset)} sequences")
    
    # Save loss values to a text file
    with open(f"{name}_losses.txt", "w") as f:
        f.write("Index\tLoss\n")
        for i, loss in enumerate(losses):
            f.write(f"{i}\t{loss:.6f}\n")
    
    print(f"Loss values saved to {name}_losses.txt")
    return losses, indices

def create_plots(train_losses, train_indices, val_losses, val_indices, model_name, output_prefix):
    """
    Create and save plots for a pair of datasets
    """
    # Set global font size
    plt.rcParams.update({'font.size': 16})
    
    # Create scatter plot
    plt.figure(figsize=(12, 8))
    
    # Sort losses to see the distribution
    sorted_train_losses = sorted(train_losses)
    sorted_val_losses = sorted(val_losses)
    
    # Plot sorted losses
    plt.subplot(2, 2, 1)
    plt.plot(sorted_train_losses, label='Training Data')
    plt.plot(sorted_val_losses, label='Validation Data')
    plt.xlabel('Loss Value', fontsize=16)
    plt.ylabel('Sample Index (Sorted)', fontsize=16)
    plt.title(f'Sorted Loss Distribution - {model_name}', fontsize=16)
    plt.legend(fontsize=16)
    plt.grid(True)
    plt.tick_params(axis='both', which='major', labelsize=16)
    
    # Plot histogram of losses
    plt.subplot(2, 2, 2)
    plt.hist(train_losses, bins=50, alpha=0.5, label='Training Data')
    plt.hist(val_losses, bins=50, alpha=0.5, label='Validation Data')
    plt.xlabel('Loss Value', fontsize=16)
    plt.ylabel('Frequency', fontsize=16)
    plt.title(f'Loss Distribution Histogram - {model_name}', fontsize=16)
    plt.legend(fontsize=16)
    plt.grid(True)
    plt.tick_params(axis='both', which='major', labelsize=16)
    
    # Scatter plot of training losses
    plt.subplot(2, 2, 3)
    plt.scatter(train_losses, train_indices, alpha=0.5, s=5)
    plt.xlabel('Loss Value', fontsize=16)
    plt.ylabel('Sequence Index', fontsize=16)
    plt.title(f'Training Set Loss by Index - {model_name}', fontsize=16)
    plt.grid(True)
    plt.tick_params(axis='both', which='major', labelsize=16)
    
    # Scatter plot of validation losses
    plt.subplot(2, 2, 4)
    plt.scatter(val_losses, val_indices, alpha=0.5, s=5, color='orange')
    plt.xlabel('Loss Value', fontsize=16)
    plt.ylabel('Sequence Index', fontsize=16)
    plt.title(f'Validation Set Loss by Index - {model_name}', fontsize=16)
    plt.grid(True)
    plt.tick_params(axis='both', which='major', labelsize=16)
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_loss_distribution_plots.png', dpi=300)
    print(f"Plots saved to {output_prefix}_loss_distribution_plots.png")
    
    # Create a separate detailed scatter plot comparing train vs val
    plt.figure(figsize=(10, 6))
    
    # Calculate mean and std for each dataset
    train_mean = np.mean(train_losses)
    train_std = np.std(train_losses)
    val_mean = np.mean(val_losses)
    val_std = np.std(val_losses)
    
    # Plot with error bars showing mean and std
    plt.errorbar(0, train_mean, yerr=train_std, fmt='o', capsize=10, 
                 markersize=10, label=f'Train (Mean={train_mean:.4f}, Std={train_std:.4f})')
    plt.errorbar(1, val_mean, yerr=val_std, fmt='o', capsize=10, 
                 markersize=10, label=f'Val (Mean={val_mean:.4f}, Std={val_std:.4f})')
    
    plt.xticks([0, 1], ['Training Data', 'Validation Data'], fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylabel('Loss Value', fontsize=16)
    plt.title(f'Comparison of Loss Statistics - {model_name}', fontsize=16)
    plt.grid(True)
    plt.legend(fontsize=16)
    
    plt.savefig(f'{output_prefix}_loss_comparison.png', dpi=300)
    print(f"Comparison plot saved to {output_prefix}_loss_comparison.png")
    
    # Reset matplotlib settings to default
    plt.rcParams.update(plt.rcParamsDefault)

def main():
    """
    Load models and datasets, compute losses, and plot scatter graphs
    """
    # Define configurations
    configs = [
        {
            "model_path": "checkpoints/cl/cl_experiment15_epoch_182_final.pt",
            "train_dataset_path": "train_ds_15.pt",
            "val_dataset_path": "val_ds_15.pt",
            "model_name": "CL 80%",
            "output_prefix": "CL 80%"
        },

    ]
    
    for config in configs:
        print(f"\n{'='*40}\nProcessing {config['model_name']}\n{'='*40}")
        
        # Load trained model
        print(f"Loading model from {config['model_path']}...")
        try:
            checkpoint = torch.load(config['model_path'], weights_only=False, map_location=device)
            model = MusicTransformer(**hparams).to(device)
            model.load_state_dict(checkpoint["model_state_dict"])
            
            # Load the datasets
            print(f"Loading training dataset from {config['train_dataset_path']}...")
            train_ds = torch.load(config['train_dataset_path'], weights_only=False, map_location=device)
            print(f"Loading validation dataset from {config['val_dataset_path']}...")
            val_ds = torch.load(config['val_dataset_path'], weights_only=False, map_location=device)
            
            # Compute losses for both datasets
            train_losses, train_indices = compute_losses(
                model, train_ds, f"train_{config['output_prefix']}")
            val_losses, val_indices = compute_losses(
                model, val_ds, f"val_{config['output_prefix']}")
            
            # Create and save plots
            create_plots(
                train_losses, train_indices, 
                val_losses, val_indices, 
                config['model_name'], 
                config['output_prefix']
            )
            
        except Exception as e:
            print(f"Error processing {config['model_name']}: {e}")
            continue

if __name__ == "__main__":
    main()