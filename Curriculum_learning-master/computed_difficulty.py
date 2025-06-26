import torch
import argparse
from torch.utils.data import DataLoader, TensorDataset
from model import MusicTransformer
from hparams import device, hparams
from masking import create_mask
import os


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


def compute_sequence_difficulty(model, dataloader):
    """
    Computes loss for each sequence and sorts them by difficulty.

    Args:
        model: Trained Music Transformer model.
        dataloader: DataLoader containing sequences.

    Returns:
        Sorted TensorDataset (easy → hard).
    """
    model.eval()
    sequence_difficulties = []
    original_indices = []  # Keep track of original indices
    print("Total number of sequences before:", len(dataloader))
    
    with torch.no_grad():
        for i, (inp, tar) in enumerate(dataloader):
            inp, tar = inp.unsqueeze(0).to(device), tar.unsqueeze(0).to(device)
            mask = create_mask(inp, n=inp.dim() + 2)
            predictions = model(inp, mask=mask)
            loss = loss_fn(predictions.transpose(-1, -2), tar)
            sequence_difficulties.append((inp, tar, loss.item(), i))  # Add original index
            original_indices.append(i)

    # Sort sequences from easiest (lowest loss) to hardest (highest loss)
    sequence_difficulties.sort(key=lambda x: x[2])

    # Print top 5 most difficult sequences
    for i in range(-1, -6, -1):
        seq_inp, seq_tar, loss_value, orig_idx = sequence_difficulties[i]
        print(f"Sequence {len(sequence_difficulties) + i + 1}: Loss = {loss_value}")
        print(f"Original index: {orig_idx}")
        print("Input Sequence:", seq_inp)
        print("Target Sequence:", seq_tar)

    # Extract sorted inputs and targets
    sorted_inputs = torch.stack([seq[0].squeeze(0) for seq in sequence_difficulties])  
    sorted_targets = torch.stack([seq[1].squeeze(0) for seq in sequence_difficulties])  

    # Save loss values to a text file in the exact format requested
    with open("Loss_experiment2_1.txt", "w") as f:
        f.write("Index\tOriginal_Index\tLoss\t\n")
        for i, (_, _, loss, orig_idx) in enumerate(sequence_difficulties):
            f.write(f"{i}\t{orig_idx}\t{loss:.6f}\t\n")
    
    print("Loss values saved to sequence_loss_values.txt in the requested format")
    print("Shape of sorted inputs:", sorted_inputs.shape)
    print("Shape of sorted targets:", sorted_targets.shape)

    # Return sorted dataset in the same format as train_ds
    return TensorDataset(sorted_inputs, sorted_targets)


def main(model_path, dataset_path, output_path, batch_size=32):
    """
    Loads model & dataset, computes difficulty scores, sorts sequences, and saves results.

    Args:
        model_path: Path to trained model checkpoint (.pt file).
        dataset_path: Path to preprocessed MIDI dataset (.pt file).
        output_path: Path to save sorted sequences (.pt file).
        batch_size: Batch size for evaluation.
    """
    
    # Load the TensorDataset
    print("Loading training dataset...")
    train_ds = torch.load("train_ds_15.pt", weights_only=False)  # Load TensorDataset
    # train_inp, train_tar = torch.load("train_ds.pt")
    print("Training dataset loaded.")
    # Access the underlying tensors and move them to the device
    train_inp = train_ds.tensors[0].long().to(device)  # Input sequences
    train_tar = train_ds.tensors[1].long().to(device)  # Target sequences

    # train_inp = train_inp.long().to(device)
    # train_tar = train_tar.long().to(device)
    print("Training input shape:", train_inp.shape)
    print("Training target shape:", train_tar.shape)

    # Create a DataLoader for batch processing
    # train_dl = DataLoader(TensorDataset(train_inp, train_tar), batch_size=batch_size, shuffle=False)
    train_dl = TensorDataset(train_inp, train_tar)
    
    # Load trained model
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, weights_only=True, map_location=device)
    model = MusicTransformer(**hparams).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Compute sequence difficulty
    print("Computing sequence difficulty...")
    sorted_ds = compute_sequence_difficulty(model, train_dl)

    # Save sorted sequences in the same format as train_ds
    print(f"Saving sorted sequences to {output_path}...")
    torch.save(sorted_ds, output_path)
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute sequence difficulty and save sorted dataset")
    parser.add_argument("model_path", help="Path to trained model checkpoint (.pt file)")
    parser.add_argument("dataset_path", help="Path to preprocessed dataset (.pt file)")
    parser.add_argument("output_path", help="Path to save sorted sequences (.pt file)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for difficulty computation")
    
    args = parser.parse_args()
    main(args.model_path, args.dataset_path, args.output_path, args.batch_size)