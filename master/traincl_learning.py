"""
Copyright 2021 Aditya Gomatam.

This file is part of music-transformer (https://github.com/spectraldoy/music-transformer), my project to build and
train a Music Transformer. music-transformer is open-source software licensed under the terms of the GNU General
Public License v3.0. music-transformer is free software: you can redistribute it and/or modify it under the terms of
the GNU General Public License as published by the Free Software Foundation, either version 3 of the License,
or (at your option) any later version. music-transformer is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU General Public License for more details. A copy of this license can be found within the GitHub repository
for music-transformer, or at https://www.gnu.org/licenses/gpl-3.0.html.

Modifications made by Gritta Joshy and Qi Chen as part of Master's thesis project:
"Investigation of Curriculum Learning in Deep Generative Modelling Using Western Classical Music"
June 2025
"""

import argparse
import time
import os
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from hparams import device
from masking import create_mask
from model import MusicTransformer
from tensorboardX import SummaryWriter
from generate import generate

"""
Functionality to train a Music Transformer on a single CPU or single GPU

The transformer is an autoregressive model, which means that at the inference stage, it will make next predictions 
based on its previous outputs. However, while training, we can use teacher forcing - feeding the target into the 
model as previous output regardless of the true output of the model. This significantly cuts down on the compute 
required, while usually reducing loss (at the expense of generalizability of the model). Since we are training a 
generative model, the targets are simply the inputs shifted right by 1 position.
"""


# def transformer_lr_schedule(d_model, step_num, warmup_steps=113580):
#     """
#     As per Vaswani et. al, 2017, the post-LayerNorm transformer performs vastly better a custom learning rate
#     schedule. Though the PyTorch implementation of the Music-Transformer uses pre-LayerNorm, which has been observed
#     not to require a custom schedule, this function is here for utility.

#     Args:
#         d_model: embedding / hidden dimenision of the transformer
#         step_num: current training step
#         warmup_steps: number of transformer schedule warmup steps. Set to 0 for a continuously decaying learning rate

#     Returns:
#         learning rate at current step_num
#     """
#     if warmup_steps <= 0:
#         step_num += 113580
#         warmup_steps = 113580
#     step_num = step_num + 1e-6  # avoid division by 0

#     if type(step_num) == torch.Tensor:
#         arg = torch.min(step_num ** -0.5, step_num * (warmup_steps ** -1.5))
#     else:
#         arg = min(step_num ** -0.5, step_num * (warmup_steps ** -1.5))

#     return (d_model ** -0.5) * arg



def transformer_lr_schedule(d_model, step_num, warmup_steps=2000, stable_until_step=113580):
    """
    Three-phase learning rate schedule:
    1. Original transformer schedule until warmup_steps (2000)
    2. Stay constant until stable_until_step (113580)
    3. Decay after stable_until_step
    """
    step_num = step_num + 1e-6  # avoid division by 0
    base_lr_factor = (d_model ** -0.5)
    
    if step_num <= warmup_steps:
        # Phase 1: Original transformer schedule (same as before)
        if type(step_num) == torch.Tensor:
            arg = torch.min(step_num ** -0.5, step_num * (warmup_steps ** -1.5))
        else:
            arg = min(step_num ** -0.5, step_num * (warmup_steps ** -1.5))
        return base_lr_factor * arg
        
    elif step_num <= stable_until_step:
        # Phase 2: Stable (maintain LR from step warmup_steps)
        # Calculate what the LR was at warmup_steps using original formula
        stable_step = warmup_steps + 1e-6
        if type(stable_step) == torch.Tensor:
            stable_arg = torch.min(stable_step ** -0.5, stable_step * (warmup_steps ** -1.5))
        else:
            stable_arg = min(stable_step ** -0.5, stable_step * (warmup_steps ** -1.5))
        return base_lr_factor * stable_arg
        
    else:
        # Phase 3: Exponential decay from the stable value
        # Start decay from the LR value at warmup_steps
        stable_step = warmup_steps + 1e-6
        if type(stable_step) == torch.Tensor:
            stable_arg = torch.min(stable_step ** -0.5, stable_step * (warmup_steps ** -1.5))
        else:
            stable_arg = min(stable_step ** -0.5, stable_step * (warmup_steps ** -1.5))
        
        stable_lr = base_lr_factor * stable_arg
        decay_steps = step_num - stable_until_step
        decay_rate = 0.9999  # Adjust this for faster/slower decay
        return stable_lr * (decay_rate ** decay_steps)


def loss_fn(prediction, target, criterion=F.cross_entropy):
    """
    Since some positions of the input sequences are padded, we must calculate the loss by appropriately masking
    padding values

    Args:
        prediction: output of the model for some input
        target: true value the model was supposed to predict
        criterion: vanilla loss criterion

    Returns:
        masked loss between prediction and target
    """
    mask = torch.ne(target, torch.zeros_like(target))           # ones where target is 0
    _loss = criterion(prediction, target, reduction='none')     # loss before masking

    # multiply mask to loss elementwise to zero out pad positions
    mask = mask.to(_loss.dtype)
    _loss *= mask

    # output is average over the number of values that were not masked
    return torch.sum(_loss) / torch.sum(mask)

def train_step(model: MusicTransformer, opt, sched, inp, tar):
    """
    Computes loss and backward pass for a single training step of the model

    Args:
        model: MusicTransformer model to train
        opt: optimizer initialized with model's parameters
        sched: scheduler properly initialized with opt
        inp: input batch
        tar: input batch shifted right by 1 position; MusicTransformer is a generative model

    Returns:
        loss before current backward pass
    """
    # forward pass
    predictions = model(inp, mask=create_mask(inp, n=inp.dim() + 2))

    # backward pass
    opt.zero_grad()
    loss = loss_fn(predictions.transpose(-1, -2), tar)
    loss.backward()
    opt.step()
    sched.step()

    return float(loss)


def val_step(model: MusicTransformer, inp, tar):
    """
    Computes loss for a single evaluation / validation step of the model

    Args:
        model: MusicTransformer model to evaluate
        inp: input batch
        tar: input batch shifted right by 1 position

    Returns:
        loss of model on input batch
    """
    # # Ensure proper shapes
    # if len(inp.shape) == 1:
    #     inp = inp.unsqueeze(0)
    #     tar = tar.unsqueeze(0)
    predictions = model(inp, mask=create_mask(inp, n=max(inp.dim() + 2, 2)))
    loss = loss_fn(predictions.transpose(-1, -2), tar)
    return float(loss)



class MusicTransformerTrainer:
    """
    Trainer class for the MusicTransformer with curriculum learning capabilities.
    Handles loading/saving checkpoints and maintaining training progress.
    """
    
    def __init__(self, hparams_, batch_size, total_epochs, warmup_steps=2000, stable_until_step=113580, ckpt_path="sortedsequence/cl/experiment15.pt", 
                 load_from_checkpoint=False, curriculum=True):
        """
        Initialize with proper log continuation when loading from checkpoint
        """
        # Constants for curriculum learning
        self.TARGET_BATCHES = 189300  # Target batch count for full curriculum - DO NOT CHANGE
        
        # Initialize basic parameters
        self.batch_size = batch_size
        self.curriculum = curriculum
        self.total_epochs = total_epochs
        self.current_percent = 0.2  # Start with 20% easiest sequences
        self.global_step = 0  # Initialize global step counter
        self.val_step_counter = 0
        
        # Create checkpoint directory if needed
        checkpoint_dir = os.path.dirname(ckpt_path)
        if checkpoint_dir and not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Initialize TensorBoard logging
        log_dir = "logs/cl_experiment_large_15_60_lr"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            
        # Create the TensorBoard writer
        if load_from_checkpoint and os.path.exists(log_dir):
            try:
                self.writer = SummaryWriter(log_dir=log_dir)
            except Exception as e:
                print(f"Error creating TensorBoard writer: {e}")
                self.writer = SummaryWriter(log_dir=log_dir)
        else:
            self.writer = SummaryWriter(log_dir=log_dir)
            
        map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Load datasets
        try:
            self.train_data_len = len(torch.load("sortedsequence/cl/experiment15.pt", weights_only=False,map_location=map_location))
            self.full_train_dataset = torch.load("sortedsequence/cl/experiment15.pt", weights_only=False,map_location=map_location)
            self.val_dataset = torch.load("val_ds_15.pt", weights_only=False, map_location=map_location)
            
            self.val_dl = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)
            print(f"Loaded datasets with {self.train_data_len} training sequences")
        except Exception as e:
            print(f"Error loading dataset files: {e}")
            raise
        
        # Create model
        self.model = MusicTransformer(**hparams_).to(device) 
        self.hparams = hparams_
        
        # Setup training
        self.warmup_steps = warmup_steps
        self.stable_until_step= stable_until_step
        self.optimizer = optim.Adam(self.model.parameters(), lr=1.0, betas=(0.9, 0.98))
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda x: transformer_lr_schedule(self.hparams['d_model'], x, self.warmup_steps,self.stable_until_step)
        )
        
        # Setup checkpointing
        self.ckpt_path = ckpt_path
        self.train_losses = []
        self.val_losses = []
        
        # Load checkpoint if necessary
        if load_from_checkpoint and os.path.isfile(self.ckpt_path):
            self.load()
            # Calculate current curriculum percentage based on global step
            self.current_percent = self.calculate_curriculum_percent(self.global_step)
            # self.val_step_counter = 0
            print(f"Training will continue from global step {self.global_step}")
            print(f"Current curriculum percentage: {self.current_percent*100:.1f}%")
            print(f"Current learning rate: {self.optimizer.param_groups[0]['lr']:.6e}")

    def calculate_curriculum_percent(self, total_batches):
        """
        Calculate curriculum percentage with three phases:
        1. Start with 20% of data
        2. Linearly increase to 100% until 80% of target batches
        3. Train with 100% data for the final 20% of batches
        """
        # Phase transition points
        phase1_end = 0  # Start immediately with phase 2
        phase2_end = int(0.6 * self.TARGET_BATCHES)  # End linear increase at 60% of target
        
        if total_batches <= phase1_end:
            # Phase 1: Fixed at 20%
            return 0.2
        elif total_batches <= phase2_end:
            # Phase 2: Linear increase from 20% to 100%
            progress = (total_batches - phase1_end) / (phase2_end - phase1_end)
            return 0.2 + 0.8 * progress
        else:
            # Phase 3: Fixed at 100%
            return 1.0

    def save(self, ckpt_path=None):
        """
        Saves a checkpoint with global step information
        """
        if ckpt_path is not None:
            self.ckpt_path = ckpt_path
    
        # Create checkpoint directory if it doesn't exist
        checkpoint_dir = os.path.dirname(self.ckpt_path)
        if checkpoint_dir and not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir, exist_ok=True)
    
        # Prepare checkpoint data
        ckpt = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "train_losses": self.train_losses,
            "validation_losses": self.val_losses,
            "warmup_steps": self.warmup_steps,
            "stable_until_step":self.stable_until_step,
            "hparams": self.hparams,
            "global_step": self.global_step,
            "val_step_counter": self.val_step_counter, 
            "current_percent": self.current_percent
        }
    
        # Save with atomic write pattern
        temp_path = f"{self.ckpt_path}.tmp"
        try:
            torch.save(ckpt, temp_path)
            if os.path.exists(temp_path):
                if os.path.exists(self.ckpt_path):
                    os.remove(self.ckpt_path)
                os.rename(temp_path, self.ckpt_path)
                print(f"Checkpoint saved successfully to {self.ckpt_path}")
        except Exception as e:
            print(f"Error saving checkpoint: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        return

    def load(self, ckpt_path=None):
        """
        Loads a checkpoint and restores model state
        """
        if ckpt_path is not None:
            self.ckpt_path = ckpt_path

        print(f"Loading checkpoint from {self.ckpt_path}...")
        ckpt = torch.load(self.ckpt_path, weights_only=False)

        # Clean up existing model components
        del self.model, self.optimizer, self.scheduler

        # Restore model
        self.model = MusicTransformer(**ckpt["hparams"]).to(device)
        self.hparams = ckpt["hparams"]
        print("Loading model state...", end="")
        print(self.model.load_state_dict(ckpt["model_state_dict"]))

        # Restore optimizer and scheduler
        self.warmup_steps = ckpt["warmup_steps"]
        self.stable_until_step = ckpt["stable_until_step"]
        self.optimizer = optim.Adam(self.model.parameters(), lr=1.0, betas=(0.9, 0.98))
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        
        # IMPORTANT: Load the scheduler state before updating its lambda function
        # This ensures the learning rate continues from where it left off
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda x: transformer_lr_schedule(self.hparams['d_model'], x, warmup_steps=2000,stable_until_step=113580)
        )
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        
        # Verify the learning rate is properly restored
        current_lr = self.optimizer.param_groups[0]['lr']
        print(f"Restored learning rate: {current_lr:.6e}")

        # Restore training history
        self.train_losses = ckpt["train_losses"]
        self.val_losses = ckpt["validation_losses"]
        
        # Restore training progress
        if "global_step" in ckpt:
            self.global_step = ckpt["global_step"]
            print(f"Restored global step: {self.global_step}")
        else:
            # Calculate based on epochs if not in checkpoint
            completed_epochs = len(self.train_losses)
            est_batches_per_epoch = self.train_data_len // self.batch_size
            self.global_step = completed_epochs * est_batches_per_epoch
            print(f"Global step not found, estimated at: {self.global_step}")
            
        if "val_step_counter" in ckpt:
            self.val_step_counter = ckpt["val_step_counter"]
            print(f"Restored validation step counter: {self.val_step_counter}")
        else:
            # Estimate based on epochs if missing
            self.val_step_counter = len(self.val_losses) * len(self.val_dl)
            print(f"Validation step counter not found, estimated at: {self.val_step_counter}")
            
            
        # Note: We don't restore current_percent directly - we recalculate it
        if "current_percent" in ckpt:
            stored_percent = ckpt["current_percent"]
            print(f"Checkpoint stored curriculum percentage: {stored_percent*100:.1f}%")
            
        # Recalculate using consistent formula
        recalculated_percent = self.calculate_curriculum_percent(self.global_step)
        print(f"Using curriculum percentage: {recalculated_percent*100:.1f}%")
        
        return

    def fit(self, epochs):
        """
        Training loop with multi-phase curriculum learning
        """
        train_losses = []
        val_losses = []
        start = time.time()
        
        # Use the existing global_step counter
        total_batches_processed = self.global_step
        val_step_counter=self.val_step_counter
        
        # Create checkpoint directory
        checkpoint_dir = os.path.dirname(self.ckpt_path)
        if checkpoint_dir and not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir, exist_ok=True)
            
        # Show expected curriculum progression
        print("\nCurriculum learning schedule:")
        print("- Starting with: 20% of data")
        print("- Linear growth phase: 0% to 60% of target batches")
        print("- Full dataset phase: 60% to 100% of target batches\n")
        
        print("Expected progression milestones:")
        for percentage in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            step = percentage * self.TARGET_BATCHES // 100
            expected_percent = self.calculate_curriculum_percent(step)
            print(f"  - At {step} batches ({percentage}%): {expected_percent*100:.1f}% of data")
        
        print(f"\nCurrent position: {total_batches_processed} batches " +
              f"({total_batches_processed/self.TARGET_BATCHES*100:.1f}% complete)")
        current_percent = self.calculate_curriculum_percent(total_batches_processed)
        print(f"Current curriculum percentage: {current_percent*100:.1f}% of data")
        print(f"Remaining batches: {self.TARGET_BATCHES - total_batches_processed}")
        
        # Starting epoch number
        starting_epoch = len(self.train_losses)
        
        print("\n=== Starting Curriculum Learning Training ===")
        print(f"Resuming from epoch {starting_epoch}" if starting_epoch > 0 else "Starting new training")
        print(f"Target batches: {self.TARGET_BATCHES}")
        print(f"Total epochs planned: {epochs}")
        print(f"Batch size: {self.batch_size}")
        print(f"Using {'GPU' if torch.cuda.is_available() else 'CPU'} for training\n")
        
        try:
            early_stop = False
            for epoch in range(epochs):
                if early_stop:
                    break 
                epoch_start_time = time.time()
                train_epoch_losses = []
    
                # Calculate current curriculum percentage with consistent formula
                current_percent = self.calculate_curriculum_percent(total_batches_processed)
                self.current_percent = current_percent  # Store for checkpointing
                subset_size = int(self.train_data_len * current_percent)
                
                # Create subset dataset (always using the same range)
                subset_indices = range(subset_size)  # Always take easiest first (sorted)
                subset_dataset = Subset(self.full_train_dataset, subset_indices)
                train_loader = DataLoader(subset_dataset, batch_size=self.batch_size, shuffle=True)
                self.train_dl = train_loader
    
                # Get current epoch number
                current_epoch = len(self.train_losses)
                
                print(f"\n=== Epoch {current_epoch + 1}/{epochs + starting_epoch} ===")
                print(f"Global step: {total_batches_processed}/{self.TARGET_BATCHES} " +
                      f"({total_batches_processed/self.TARGET_BATCHES*100:.1f}%)")
                print(f"Using {current_percent*100:.1f}% of training data ({subset_size}/{self.train_data_len} sequences)")
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Current learning rate: {current_lr:.6e}")
    
                # Training loop
                self.model.train()
    
                for batch in train_loader:
                    train_inp, train_tar = batch
                    
                    # Ensure proper shapes
                    if len(train_inp.shape) == 1:
                        train_inp = train_inp.unsqueeze(0)
                        train_tar = train_tar.unsqueeze(0)
                    
                    # Forward and backward pass
                    loss = train_step(self.model, self.optimizer, self.scheduler, train_inp, train_tar)
                    
                    # Track and log metrics
                    train_epoch_losses.append(loss)
                    self.writer.add_scalar("Loss/Train_per_batch", loss, total_batches_processed)
                    self.writer.add_scalar("learning_rate/by_batch", current_lr, total_batches_processed)
                    
                    # Update counters
                    total_batches_processed += 1
                    self.global_step = total_batches_processed
                    
                    # Stop if target reached
                    # if total_batches_processed >= self.TARGET_BATCHES:
                    if total_batches_processed >= 189300 and current_percent >= 1.0:
                        print(f"Reached target batch count of {self.TARGET_BATCHES}. Stopping.")
                        early_stop = True
                        break
    
                # Validation
                self.model.eval()
                val_epoch_losses = []
                # with torch.no_grad():
                #     for val_inp, val_tar in self.val_dl:
                #         loss = val_step(self.model, val_inp, val_tar)
                #         val_epoch_losses.append(loss)
                #         self.writer.add_scalar("Loss/Validation_per_batch", loss, total_batches_processed)


                with torch.no_grad():
                    for val_inp, val_tar in self.val_dl:
                        loss = val_step(self.model, val_inp, val_tar)
                        val_epoch_losses.append(loss)
                        self.writer.add_scalar("Loss/Validation_per_batch", loss, val_step_counter)
                        val_step_counter += 1
                    self.val_step_counter = val_step_counter

    
                # Calculate and log metrics
                train_mean = sum(train_epoch_losses) / len(train_epoch_losses)
                val_mean = sum(val_epoch_losses) / len(val_epoch_losses)
                
                self.train_losses.append(train_mean)
                train_losses.append(train_mean)
                self.val_losses.append(val_mean)
                val_losses.append(val_mean)
                
                # Print epoch summary
                epoch_time = time.time() - epoch_start_time
                print(f"\nEpoch {current_epoch + 1} Summary:")
                print(f"Training Loss: {train_mean:.4f}")
                print(f"Validation Loss: {val_mean:.4f}")
                print(f"Time taken: {epoch_time:.2f} seconds")
                print(f"Batches processed: {total_batches_processed}/{self.TARGET_BATCHES} " +
                      f"({total_batches_processed/self.TARGET_BATCHES*100:.1f}%)")
                print(f"Curriculum percentage: {current_percent*100:.1f}%")
                print("-" * 50)
                
                # TensorBoard logging
                self.writer.add_scalar("Loss/Train", train_mean, current_epoch + 1)
                self.writer.add_scalar("Loss/Validation", val_mean, current_epoch + 1)
                self.writer.add_scalar("Data_Percentage", current_percent, current_epoch + 1)
                self.writer.add_scalar("learning_rate/by_epoch", current_lr, current_epoch + 1)
                
                # Save checkpoint periodically
                if (current_epoch + 1) % 10 == 0 or (current_epoch + 1) in [1, 5]:
                    clean_checkpoint_path = f"{checkpoint_dir}/cl_experiment15_epoch_{current_epoch + 1}.pt"
                    print(f"Saving intermediate checkpoint at epoch {current_epoch + 1}...")
                    self.save(clean_checkpoint_path)
    
        except KeyboardInterrupt:
            print("\nTraining interrupted! Saving checkpoint...")
            interrupted_epoch = len(self.train_losses)
            interrupted_checkpoint_path = f"{checkpoint_dir}/cl_experiment15_epoch_{interrupted_epoch}_interrupted.pt"
            self.save(interrupted_checkpoint_path)
        
        print("\n=== Curriculum Training Complete ===")
        print(f"Total batches processed: {total_batches_processed}/{self.TARGET_BATCHES}")
        if total_batches_processed >= self.TARGET_BATCHES:
            print("✓ Target batch count reached successfully!")
        else:
            print(f"✗ Target batch count not reached. {self.TARGET_BATCHES - total_batches_processed} batches remaining.")
        
        print(f"Final curriculum percentage: {self.current_percent*100:.1f}%")
        print(f"Final Training Loss: {self.train_losses[-1]:.4f}")
        print(f"Final Validation Loss: {self.val_losses[-1]:.4f}")
        print(f"Total training time: {(time.time() - start)/60:.2f} minutes")
        
        # Save final checkpoint
        final_epoch = len(self.train_losses)
        final_checkpoint_path = f"{checkpoint_dir}/cl_experiment15_epoch_{final_epoch}_final.pt"
        self.save(final_checkpoint_path)
        
        self.writer.close()
        return train_losses, val_losses


if __name__ == "__main__":
    from hparams import hparams

    def check_positive(x):
        if x is None:
            return x
        x = int(x)
        if x <= 0:
            raise argparse.ArgumentTypeError(f"{x} is not a positive integer")
        return x

    parser = argparse.ArgumentParser(
        prog="train.py",
        description="Train a Music Transformer on single tensor dataset of preprocessed MIDI files"
    )

    # trainer arguments
    # parser.add_argument("datapath", help="path at which preprocessed MIDI files are stored as a single tensor after "
    #                                      "being translated into an event vocabulary")
    parser.add_argument("ckpt_path", help="path at which to load / store checkpoints while training; "
                                          "KeyboardInterrupt while training to checkpoint the model; MUST end in .pt "
                                          "or .pth", type=str)
    parser.add_argument("save_path", help="path at which to save the model's state dict and hyperparameters after "
                                          "training; model will only be saved if the training loop finishes before a "
                                          "KeyboardInterrupt; MUST end in .pt or .pth", type=str)
    parser.add_argument("epochs", help="number of epochs to train for", type=check_positive)
    parser.add_argument("-bs", "--batch-size", help="number of sequences to batch together to compute a single "
                                                    "training step while training; default: 32", type=check_positive)
    parser.add_argument("-l", "--load-checkpoint", help="flag to load a previously saved checkpoint from which to "
                                                        "resume training; default: False", action="store_true")
    parser.add_argument("-w", "--warmup-steps", help="number of warmup steps for transformer learning rate scheduler; "
                                                     "if loading from checkpoint, this will be overwritten by saved "
                                                     "value; default: 4000", type=int)
    parser.add_argument("--curriculum", action="store_true", help="Enable linear curriculum learning")
    # parser.add_argument("--total-epochs", type=int, default=100, help="Total epochs for curriculum scheduling")
    parser.add_argument("--stable-until-step", help="step until which learning rate remains stable; "
                                                "default: 113580", type=int)

    parser.add_argument("--start-percent", type=float, default=None, 
                        help="Starting curriculum percentage (0.0-1.0) when resuming training")

    # hyperparameters
    parser.add_argument("-d", "--d-model",
                        help="music transformer hidden dimension size; if loading from checkpoint "
                             "this will be overwritten by saved hparams; default: 128", type=check_positive)
    parser.add_argument("-nl", "--num-layers",
                        help="number of transformer decoder layers in the music transformer; if loading from "
                             "checkpoint, this will be overwritten by saved hparams; default: 3", type=check_positive)
    parser.add_argument("-nh", "--num-heads",
                        help="number of attention heads over which to compute multi-head relative attention in the "
                             "music transformer; if loading from checkpoint, this will be overwritten by saved "
                             "hparams; default: 8", type=check_positive)
    parser.add_argument("-dff", "--d-feedforward",
                        help="hidden dimension size of pointwise FFN layers in the music transformer; if loading from "
                             "checkpoint, this will be overwritten by saved hparams; default: 512", type=check_positive)
    parser.add_argument("-mrd", "--max-rel-dist",
                        help="maximum relative distance between tokens to consider in relative attention calculation "
                             "in the music transformer; if loading from checkpoint, this will be overwritten by saved "
                             "hparams; default: 1024", type=check_positive)
    parser.add_argument("-map", "--max-abs-position",
                        help="maximum absolute length of an input sequence; set this to a very large value to be able "
                             "to generalize to longer sequences than in the dataset; if a sequence longer than the "
                             "passed in value is passed into the dataset, max_abs_position is set to that value not "
                             "the passed in; if loading from checkpoint, this will be overwritten by saved hparams; "
                             "default: 20000", type=int)
    parser.add_argument("-vs", "--vocab-size",
                        help="length of the vocabulary in which the input training data has been tokenized. if "
                             "loading from checkpoint, this will be overwritten by saved hparams; default: 416 (size "
                             "of Oore et. al MIDI vocabulary)", type=check_positive)
    parser.add_argument("-nb", "--no-bias",
                        help="flag to not use a bias in the linear layers of the music transformer; if loading from "
                             "checkpoint, this will be overwritten by saved hparams; default: False",
                        action="store_false")
    parser.add_argument("-dr", "--dropout",
                        help="dropout rate for training the model; if loading from checkpoint, this will be "
                             "overwritten by saved hparams; default: 0.1")
    parser.add_argument("-le", "--layernorm-eps",
                        help="epsilon in layernorm layers to avoid zero division; if loading from checkpoint, "
                             "this will be overwritten by saved hparams; default: 1e-6")

    args = parser.parse_args()

    # fix optional parameters
    batch_size_ = 16 if args.batch_size is None else args.batch_size #changed from 32 to 16
    warmup_steps_ = 2000 if args.warmup_steps is None else args.warmup_steps
    stable_until_step = 113580 if args.stable_until_step is None else args.stable_until_step




    # fix hyperparameters
    hparams["d_model"] = args.d_model if args.d_model else hparams["d_model"]
    hparams["num_layers"] = args.num_layers if args.num_layers else hparams["num_layers"]
    hparams["num_heads"] = args.num_heads if args.num_heads else hparams["num_heads"]
    hparams["d_ff"] = args.d_feedforward if args.d_feedforward else hparams["d_ff"]
    hparams["max_rel_dist"] = args.max_rel_dist if args.max_rel_dist else hparams["max_rel_dist"]
    hparams["max_abs_position"] = args.max_abs_position if args.max_abs_position else hparams["max_abs_position"]
    hparams["vocab_size"] = args.vocab_size if args.vocab_size else hparams["vocab_size"]
    hparams["bias"] = args.no_bias
    hparams["dropout"] = args.dropout if args.dropout else hparams["dropout"]
    hparams["layernorm_eps"] = args.layernorm_eps if args.layernorm_eps else hparams["layernorm_eps"]


    # set up the trainer
    print("Setting up the curriculum learning trainer...")
    # trainer = MusicTransformerTrainer(hparams, batch_size_, args.epochs, warmup_steps_, args.ckpt_path, args.load_checkpoint, curriculum=args.curriculum)
    trainer = MusicTransformerTrainer(
        hparams, batch_size_, args.epochs, warmup_steps_, stable_until_step,
        args.ckpt_path, args.load_checkpoint, 
        curriculum=args.curriculum,

    )    

    print()

    # train the model
    trainer.fit(args.epochs)
    # trainer.fit(args.total_epochs)

    # done training, save the model
    print("Saving...")
    save_file = {
        "state_dict": trainer.model.state_dict(),
        "hparams": trainer.hparams
    }
    torch.save(save_file, args.save_path)
    print("Done with curriculum training training!")
