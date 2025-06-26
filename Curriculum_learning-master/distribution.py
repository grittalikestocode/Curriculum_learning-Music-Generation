import torch
import argparse
from torch.utils.data import DataLoader, TensorDataset
from model import MusicTransformer
from hparams import device, hparams
from masking import create_mask
import os
import matplotlib.pyplot as plt
import numpy as np

def loss_fn(prediction, target, criterion=torch.nn.functional.cross_entropy):
    """
    Computes masked loss, ignoring padding values.

    Args:
        prediction: Model output logits.
        target: Ground truth target sequence.
        criterion: Loss function (default: cross-entropy).

    Returns:
        Masked loss value.
    """
    mask = torch.ne(target, torch.zeros_like(target))  # Mask padding positions
    _loss = criterion(prediction, target, reduction='none')  # Compute loss

    mask = mask.to(_loss.dtype)
    _loss *= mask

    return torch.sum(_loss) / torch.sum(mask)


def compute_sequence_difficulty(model, dataloader, source_tags=None, plot_path="sequence_difficulty_plot.png"):
    """
    Computes loss for each sequence and plots the loss distribution.

    Args:
        model: Trained Music Transformer model.
        dataloader: DataLoader containing sequences.
        source_tags: Optional tensor of source tags (0=middle, 1=end).
        plot_path: Path to save the difficulty plot.

    Returns:
        Sorted TensorDataset (easy → hard) and sorted source tags if provided.
    """
    model.eval()
    sequence_difficulties = []
    print("Total number of sequences before:", len(dataloader))
    
    with torch.no_grad():
        for i, (inp, tar) in enumerate(dataloader):
            inp, tar = inp.unsqueeze(0).to(device), tar.unsqueeze(0).to(device)
            mask = create_mask(inp, n=inp.dim() + 2)
            predictions = model(inp, mask=mask)
            loss = loss_fn(predictions.transpose(-1, -2), tar)
            
            # Include source tag if available
            tag = source_tags[i].item() if source_tags is not None else None
            sequence_difficulties.append((inp, tar, loss.item(), tag, i))  # Keep inputs, targets, loss, tag, and original index

    # Sort sequences from easiest (lowest loss) to hardest (highest loss)
    sequence_difficulties.sort(key=lambda x: x[2])

    # Extract loss values and tags
    loss_values = [seq[2] for seq in sequence_difficulties]
    
    # Create the loss distribution plot
    plt.figure(figsize=(12, 8))
    
    # Calculate bins for histogram
    n_bins = min(100, len(loss_values) // 10)  # Adaptive bin size
    
    if source_tags is not None:
        # Separate losses by source tag
        sorted_tags = [seq[3] for seq in sequence_difficulties]
        middle_indices = [i for i, tag in enumerate(sorted_tags) if tag == 0]
        end_indices = [i for i, tag in enumerate(sorted_tags) if tag == 1]
        
        middle_losses = [loss_values[i] for i in middle_indices]
        end_losses = [loss_values[i] for i in end_indices]
        
        # Plot separate histograms for middle and end sections
        plt.hist(middle_losses, bins=n_bins, alpha=0.6, color='blue', label='Middle sections')
        plt.hist(end_losses, bins=n_bins, alpha=0.6, color='red', label='End sections')
        
        # Add source-specific statistics
        middle_mean = np.mean(middle_losses) if middle_losses else 0
        end_mean = np.mean(end_losses) if end_losses else 0
        
        plt.axvline(x=middle_mean, color='blue', linestyle='--', 
                   label=f'Middle Mean: {middle_mean:.4f}')
        plt.axvline(x=end_mean, color='red', linestyle='--', 
                   label=f'End Mean: {end_mean:.4f}')
        
        # Add distribution statistics text
        middle_percent = len(middle_indices) / len(loss_values) * 100
        end_percent = len(end_indices) / len(loss_values) * 100
        
        stats_text = f"Dataset composition:\n"
        stats_text += f"Middle sections: {len(middle_indices)} ({middle_percent:.1f}%)\n"
        stats_text += f"End sections: {len(end_indices)} ({end_percent:.1f}%)\n\n"
        
        # Calculate how many middle/end sections are in each difficulty quartile
        quartiles = [0.25, 0.5, 0.75, 1.0]
        quartile_stats = []
        
        for i, q in enumerate(quartiles):
            start_idx = 0 if i == 0 else int(len(loss_values) * quartiles[i-1])
            end_idx = int(len(loss_values) * q)
            
            q_tags = sorted_tags[start_idx:end_idx]
            q_middle = sum(1 for t in q_tags if t == 0)
            q_end = sum(1 for t in q_tags if t == 1)
            q_total = end_idx - start_idx
            
            quartile_name = f"{'0' if i == 0 else f'{int(quartiles[i-1]*100)}'}-{int(q*100)}%"
            quartile_stats.append((quartile_name, q_middle, q_end, q_total))
        
        stats_text += "Source distribution by difficulty quartile:\n"
        for name, middle, end, total in quartile_stats:
            stats_text += f"{name}: {middle/total*100:.1f}% middle, {end/total*100:.1f}% end\n"
        
        plt.figtext(0.15, 0.02, stats_text, fontsize=10, 
                   bbox=dict(facecolor='white', alpha=0.8))
    else:
        # Single histogram for all sequences
        plt.hist(loss_values, bins=n_bins, alpha=0.7, color='skyblue', label='All sequences')
    
    # Add overall statistics
    mean_loss = np.mean(loss_values)
    median_loss = np.median(loss_values)
    
    plt.axvline(x=mean_loss, color='black', linestyle='--', 
               label=f'Overall Mean: {mean_loss:.4f}')
    plt.axvline(x=median_loss, color='green', linestyle='--', 
               label=f'Overall Median: {median_loss:.4f}')
    
    # Add percentile markers
    percentiles = [25, 50, 75]
    percentile_values = [np.percentile(loss_values, p) for p in percentiles]
    
    for i, (p, val) in enumerate(zip(percentiles, percentile_values)):
        plt.axvline(x=val, color=['purple', 'orange', 'red'][i], linestyle=':', 
                   label=f'{p}th percentile: {val:.4f}')
    
    # Add title and labels
    plt.xlabel('Loss Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of Sequence Difficulty (Loss)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save the distribution plot
    plt.tight_layout(rect=[0, 0.08, 1, 1])  # Make room for the text at the bottom
    plt.savefig(plot_path)
    print(f"Loss distribution plot saved to {plot_path}")
    
    # Extract sorted inputs, targets, and tags
    sorted_inputs = torch.stack([seq[0].squeeze(0) for seq in sequence_difficulties])  
    sorted_targets = torch.stack([seq[1].squeeze(0) for seq in sequence_difficulties])
    
    if source_tags is not None:
        sorted_tags = torch.tensor([seq[3] for seq in sequence_difficulties])
        return TensorDataset(sorted_inputs, sorted_targets), sorted_tags
    else:
        return TensorDataset(sorted_inputs, sorted_targets)


def main(model_path, output_path, batch_size=16):
    """
    Loads model & dataset, computes difficulty scores, and plots loss distribution.

    Args:
        model_path: Path to trained model checkpoint (.pt file).
        output_path: Path to save sorted sequences (.pt file).
        batch_size: Batch size for evaluation.
    """
    
    # Load the TensorDataset
    print("Loading training dataset...")
    # train_ds = torch.load("train_ds.pt", weights_only=False) 
    train_ds = torch.load(sortedsequence/sorted1cl1.pt, weights_only=False) 
    
    # Load TensorDataset
    print("Training dataset loaded.")
    
    # Check if we have source tags available
    try:
        source_tags = torch.load("train_tags.pt", weights_only=False)
        print(f"Source tags loaded: {len(source_tags)} tags")
        has_tags = True
    except:
        print("No separate source tags file found, checking if tags are in the dataset...")
        # Check if the dataset has more than 2 tensors (might include tags)
        if len(train_ds.tensors) > 2:
            source_tags = train_ds.tensors[2]
            print(f"Source tags found in dataset: {len(source_tags)} tags")
            has_tags = True
        else:
            print("No source tags found. Proceeding without source information.")
            source_tags = None
            has_tags = False
    
    # Access the underlying tensors and move them to the device
    train_inp = train_ds.tensors[0].long().to(device)  # Input sequences
    train_tar = train_ds.tensors[1].long().to(device)  # Target sequences
    print("Training input shape:", train_inp.shape)
    print("Training target shape:", train_tar.shape)

    train_dl = TensorDataset(train_inp, train_tar)
    
    # Load trained model
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, weights_only=True, map_location=device)
    model = MusicTransformer(**hparams).to(device)
    
    # Handle different checkpoint formats
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        raise ValueError("Could not find model state dict in checkpoint")

    # Set plot path
    plot_path = os.path.splitext(output_path)[0] + "_distribution.png"
    
    # Compute sequence difficulty and generate plot
    print("Computing sequence difficulty and generating distribution plot...")
    if has_tags:
        sorted_ds, sorted_tags = compute_sequence_difficulty(model, train_dl, source_tags, plot_path)
        
        # Save sorted sequences and tags
        print(f"Saving sorted sequences to {output_path}...")
        # Two options: save as separate files or as one combined dataset
        
        # Option 1: Save as separate files
        torch.save(sorted_ds, output_path)
        torch.save(sorted_tags, os.path.splitext(output_path)[0] + "_tags.pt")
        
        # Option 2: Create a new TensorDataset with tags included
        sorted_inputs, sorted_targets = sorted_ds.tensors
        combined_ds = TensorDataset(sorted_inputs, sorted_targets, sorted_tags)
        torch.save(combined_ds, os.path.splitext(output_path)[0] + "_with_tags.pt")
        
        print(f"Saved sorted sequences with tags.")
    else:
        sorted_ds = compute_sequence_difficulty(model, train_dl, None, plot_path)
        
        # Save sorted sequences
        print(f"Saving sorted sequences to {output_path}...")
        torch.save(sorted_ds, output_path)
    
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute sequence difficulty and plot loss distribution")
    parser.add_argument("model_path", help="Path to trained model checkpoint (.pt file)")
    parser.add_argument("output_path", help="Path to save sorted sequences (.pt file)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for difficulty computation")
    
    args = parser.parse_args()
    main(args.model_path, args.output_path, args.batch_size)