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
from torch.utils.data import DataLoader, TensorDataset
from hparams import device
from masking import create_mask
from model import MusicTransformer
from tensorboardX import SummaryWriter

"""
Functionality to train a Music Transformer on a single CPU or single GPU

The transformer is an autoregressive model, which means that at the inference stage, it will make next predictions 
based on its previous outputs. However, while training, we can use teacher forcing - feeding the target into the 
model as previous output regardless of the true output of the model. This significantly cuts down on the compute 
required, while usually reducing loss (at the expense of generalizability of the model). Since we are training a 
generative model, the targets are simply the inputs shifted right by 1 position.
"""


def transformer_lr_schedule(d_model, step_num, warmup_steps=4000):
    """
    As per Vaswani et. al, 2017, the post-LayerNorm transformer performs vastly better a custom learning rate
    schedule. Though the PyTorch implementation of the Music-Transformer uses pre-LayerNorm, which has been observed
    not to require a custom schedule, this function is here for utility.

    Args:
        d_model: embedding / hidden dimenision of the transformer
        step_num: current training step
        warmup_steps: number of transformer schedule warmup steps. Set to 0 for a continuously decaying learning rate

    Returns:
        learning rate at current step_num
    """
    if warmup_steps <= 0:
        step_num += 4000
        warmup_steps = 4000
    step_num = step_num + 1e-6  # avoid division by 0

    if type(step_num) == torch.Tensor:
        arg = torch.min(step_num ** -0.5, step_num * (warmup_steps ** -1.5))
    else:
        arg = min(step_num ** -0.5, step_num * (warmup_steps ** -1.5))

    return (d_model ** -0.5) * arg


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
    # Ensure proper shapes
    if len(inp.shape) == 1:
        inp = inp.unsqueeze(0)
        tar = tar.unsqueeze(0)
    predictions = model(inp, mask=create_mask(inp, n=max(inp.dim() + 2, 2)))
    loss = loss_fn(predictions.transpose(-1, -2), tar)
    return float(loss)


class MusicTransformerTrainer:
    """
    As the transformer is a large model and takes a while to train on a GPU, or even a TPU, I wrote this Trainer
    class to make it easier to load and save checkpoints with the model. The way I've designed it instantiates the
    model, optimizer, and scheduler within the class itself, as there are some problems with passing them in. But,
    to get these objects back just call:
        trainer.model
        trainer.optimizer
        trainer.scheduler

    This class also tracks the cumulative losses while training, which you can get back with:
        trainer.train_losses
        trainer.val_losses
    as lists of floats

    To save a checkpoint, call trainer.save()
    To load a checkpoint, call trainer.load( (optional) ckpt_path)
    """

    def __init__(self, hparams_, datapath, batch_size, warmup_steps=4000,
                 ckpt_path="music_transformer_ckpt.pt", load_from_checkpoint=False):
        """
        Args:
            hparams_: hyperparameters of the model
            datapath: path to the data to train on
            batch_size: batch size to batch the data
            warmup_steps: number of warmup steps for transformer learning rate schedule
            ckpt_path: path at which to save checkpoints while training; MUST end in .pt or .pth
            load_from_checkpoint (bool, optional): if true, on instantiating the trainer, this will load a previously
                                                   saved checkpoint at ckpt_path
        """
        # Initialize global step counter for continuous logging
        self.global_step = 0
        
        # Set up TensorBoard logging
        log_dir = "logs/epoch100_large_aligned"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)

        # get the data
        self.datapath = datapath
        self.batch_size = batch_size
        data = torch.load(datapath, weights_only=False).long().to(device)

        # max absolute position must be able to account for the largest sequence in the data
        if hparams_["max_abs_position"] > 0:
            hparams_["max_abs_position"] = max(hparams_["max_abs_position"], data.shape[-1])

        indices = list(range(data.shape[0]))

        val_indices = indices[::5]
        train_indices = [i for i in indices if i not in val_indices]
        
        train_data = data[train_indices]
        val_data = data[val_indices]
        
        print(f"There are {data.shape[0]} samples in the data, {len(train_data)} training samples and {len(val_data)} "
              f"validation samples (stratified sampling)")

        # datasets and dataloaders: split data into first (n-1) and last (n-1) tokens
        train_data_input = train_data[:, :-1]
        train_data_target = train_data[:, 1:]
        self.train_ds = TensorDataset(train_data_input, train_data_target)
        self.train_dl = DataLoader(dataset=self.train_ds, batch_size=batch_size, shuffle=True)

        # Load or create validation dataset
        try:
            print("Trying to load validation dataset from val_ds_15.pt...")
            self.val_ds = torch.load("val_ds_15.pt", weights_only=False)
            print("Validation dataset loaded successfully!")
        except:
            print("Creating validation dataset...")
            val_data_input = val_data[:, :-1]
            val_data_target = val_data[:, 1:]
            self.val_ds = TensorDataset(val_data_input, val_data_target)
            torch.save(self.val_ds, "val_ds_15.pt")
            print("Validation dataset created and saved!")
        
        # IMPORTANT: Create validation DataLoader with shuffle=False for consistency
        self.val_dl = DataLoader(dataset=self.val_ds, batch_size=batch_size, shuffle=False)

        # create model
        self.model = MusicTransformer(**hparams_).to(device) 
        self.hparams = hparams_

        # setup training
        self.warmup_steps = warmup_steps
        self.optimizer = optim.Adam(self.model.parameters(), lr=1.0, betas=(0.9, 0.98))
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda x: transformer_lr_schedule(self.hparams['d_model'], x, self.warmup_steps)
        )

        # setup checkpointing / saving
        self.ckpt_path = ckpt_path
        self.train_losses = []
        self.val_losses = []

        # Load checkpoint if necessary - AFTER initializing all required properties
        if load_from_checkpoint and os.path.isfile(self.ckpt_path):
            print(f"Loading checkpoint from {self.ckpt_path}...")
            self.load()
            print(f"Training will continue from global step {self.global_step}")

    def save(self, ckpt_path=None):
        """
        Saves a checkpoint at ckpt_path

        Args:
            ckpt_path (str, optional): if None, saves the checkpoint at the previously stored self.ckpt_path
                                       else saves the checkpoints at the new passed-in path, and stores this new path at
                                       the member variable self.ckpt_path
        """
        if ckpt_path is not None:
            self.ckpt_path = ckpt_path

        # Ensure checkpoint directory exists
        checkpoint_dir = os.path.dirname(self.ckpt_path)
        if checkpoint_dir and not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir, exist_ok=True)

        ckpt = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "train_losses": self.train_losses,
            "validation_losses": self.val_losses,
            "warmup_steps": self.warmup_steps,
            "hparams": self.hparams,
            "global_step": self.global_step  # Save global step to resume consistent logging
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
        Loads a checkpoint from ckpt_path
        NOTE: OVERWRITES THE MODEL STATE DICT, OPTIMIZER STATE DICT, SCHEDULER STATE DICT, AND HISTORY OF LOSSES

        Args:
            ckpt_path (str, optional): if None, loads the checkpoint at the previously stored self.ckpt_path
                                       else loads the checkpoints from the new passed-in path, and stores this new path
                                       at the member variable self.ckpt_path
        """
        if ckpt_path is not None:
            self.ckpt_path = ckpt_path

        ckpt = torch.load(self.ckpt_path)

        del self.model, self.optimizer, self.scheduler

        # create and load model
        self.model = MusicTransformer(**ckpt["hparams"]).to(device)
        self.hparams = ckpt["hparams"]
        print("Loading model state...", end="")
        print(self.model.load_state_dict(ckpt["model_state_dict"]))

        # create and load load optimizer and scheduler
        self.warmup_steps = ckpt["warmup_steps"]
        self.optimizer = optim.Adam(self.model.parameters(), lr=1.0, betas=(0.9, 0.98))
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda x: transformer_lr_schedule(self.hparams['d_model'], x, self.warmup_steps)
        )
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        # load loss histories
        self.train_losses = ckpt["train_losses"]
        self.val_losses = ckpt["validation_losses"]
        
        # Load global step for consistent logging
        if "global_step" in ckpt:
            self.global_step = ckpt["global_step"]
            print(f"Restored global step: {self.global_step}")
        else:
            # Estimate based on epochs completed
            batches_per_epoch = len(self.train_ds) // self.batch_size
            self.global_step = len(self.train_losses) * batches_per_epoch
            print(f"Estimated global step: {self.global_step} based on {len(self.train_losses)} epochs")

        return

    def fit(self, epochs):
        """
        Training loop to fit the model to the data stored at the passed in datapath. 
        Modified to align validation behavior with curriculum learning approach.
    
        Args:
            epochs: number of epochs to train for.
    
        Returns:
            history of training and validation losses for this training session
        """
        train_losses = []
        val_losses = []
        start = time.time()
        
        # Track total batches processed - start from current global step
        total_batches = self.global_step
        
        # Target validation frequency to match curriculum learning
        validation_frequency = 200  # Adjust this to match curriculum learning approach
        
        # Checkpoint saving frequency (in epochs)
        checkpoint_frequency = 20
        
        print("Beginning training with aligned validation...")
        print(f"Starting from global step: {total_batches}")
        print(time.strftime("%Y-%m-%d %H:%M"))
        model = torch.compile(self.model)
        torch.set_float32_matmul_precision("high") # this speeds up training
    
        try:
            for epoch in range(epochs):
                train_epoch_losses = []
                val_epoch_losses = []
                
                # Get current learning rate
                current_lr = self.optimizer.param_groups[0]["lr"]
                print(f"Epoch {epoch+1}: learning rate = {current_lr}")
                self.writer.add_scalar("learning_rate", current_lr, total_batches)
    
                model.train()
                for batch_idx, (train_inp, train_tar) in enumerate(self.train_dl):
                    # Train step
                    loss = train_step(model, self.optimizer, self.scheduler, train_inp, train_tar)
                    train_epoch_losses.append(loss)
                    
                    # Log training loss every batch
                    total_batches += 1
                    self.global_step = total_batches  # Update global step for checkpoint saving
                    self.writer.add_scalar("Loss/Train_per_batch", loss, total_batches)
                    
                    # Perform validation at regular intervals to match curriculum learning
                    if total_batches % validation_frequency == 0:
                        model.eval()
                        with torch.no_grad():
                            interim_val_losses = []
                            for val_inp, val_tar in self.val_dl:
                                val_loss = val_step(model, val_inp, val_tar)
                                interim_val_losses.append(val_loss)
                                # Log the validation loss at current training step
                                # This matches curriculum learning's logging approach
                                self.writer.add_scalar("Loss/Validation_per_batch", val_loss, total_batches)
                            
                            # Also log average validation loss if desired
                            if interim_val_losses:
                                avg_val_loss = sum(interim_val_losses) / len(interim_val_losses)
                                self.writer.add_scalar("Loss/Validation_avg", avg_val_loss, total_batches)
                        
                        # Resume training
                        model.train()
                    
                    # Log model parameters less frequently to reduce overhead
                    if total_batches % 500 == 0:
                        for name, param in model.named_parameters():
                            self.writer.add_histogram(f"params/{name}", param, total_batches)
    
                # Complete validation at end of epoch
                model.eval()
                with torch.no_grad():
                    for val_inp, val_tar in self.val_dl:
                        loss = val_step(model, val_inp, val_tar)
                        val_epoch_losses.append(loss)
    
                # Calculate mean losses for the epoch
                train_mean = sum(train_epoch_losses) / len(train_epoch_losses)
                val_mean = sum(val_epoch_losses) / len(val_epoch_losses)
    
                # Store loss history
                self.train_losses.append(train_mean)
                train_losses.append(train_mean)
                self.val_losses.append(val_mean)
                val_losses.append(val_mean)
                
                # Log epoch-level metrics
                self.writer.add_scalar("Loss/Train", train_mean, epoch + len(self.train_losses) - epochs)
                self.writer.add_scalar("Loss/Validation", val_mean, epoch + len(self.train_losses) - epochs)
                
                current_epoch = len(self.train_losses)
                print(f"Epoch {epoch+1}/{epochs} (Total epochs: {current_epoch}, Batches: {total_batches}) "
                      f"Time taken {round(time.time() - start, 2)} seconds "
                      f"Train Loss {train_losses[-1]:.4f} Val Loss {val_losses[-1]:.4f}")
                
                # Save checkpoint every checkpoint_frequency epochs
                if (epoch + 1) % checkpoint_frequency == 0 or (epoch + 1) == epochs:
                    checkpoint_path = self.ckpt_path.replace('.pt', f'_epoch_{current_epoch}_batch_{total_batches}.pt')
                    print(f"Saving intermediate checkpoint at epoch {current_epoch}, batch {total_batches}...")
                    self.save(checkpoint_path)
    
                start = time.time()
    
        except KeyboardInterrupt:
            print("\nTraining interrupted! Saving checkpoint...")
            interrupted_checkpoint_path = self.ckpt_path.replace('.pt', f'_interrupted_batch_{total_batches}.pt')
            self.save(interrupted_checkpoint_path)
    
        print("Checkpointing final model...")
        self.save()
        self.writer.close()
        print("Done")
        print(time.strftime("%Y-%m-%d %H:%M"))
    
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
        prog="train_batch_aligned.py",
        description="Train a Music Transformer on single tensor dataset of preprocessed MIDI files"
    )

    # trainer arguments
    parser.add_argument("datapath", help="path at which preprocessed MIDI files are stored as a single tensor")
    parser.add_argument("ckpt_path", help="path at which to load/store checkpoints while training; MUST end in .pt/.pth", type=str)
    parser.add_argument("save_path", help="path at which to save the model's state dict and hyperparameters", type=str)
    parser.add_argument("epochs", help="number of epochs to train for", type=check_positive)
    parser.add_argument("-bs", "--batch-size", help="number of sequences per batch; default: 16", type=check_positive)
    parser.add_argument("-l", "--load-checkpoint", help="flag to load checkpoint; default: False", action="store_true")
    parser.add_argument("-w", "--warmup-steps", help="warmup steps for transformer lr scheduler; default: 2000", type=int)
    parser.add_argument("-vf", "--validation-frequency", help="validation frequency in batches; default: 200", type=int, default=200)
    parser.add_argument("-cf", "--checkpoint-frequency", help="save checkpoint every N epochs; default: 20", type=int, default=20)

    # hyperparameters
    parser.add_argument("-d", "--d-model", help="transformer hidden dim; default: 128", type=check_positive)
    parser.add_argument("-nl", "--num-layers", help="transformer decoder layers; default: 3", type=check_positive)
    parser.add_argument("-nh", "--num-heads", help="attention heads; default: 8", type=check_positive)
    parser.add_argument("-dff", "--d-feedforward", help="FFN hidden dim; default: 512", type=check_positive)
    parser.add_argument("-mrd", "--max-rel-dist", help="max relative distance; default: 1024", type=check_positive)
    parser.add_argument("-map", "--max-abs-position", help="max seq length; default: 20000", type=int)
    parser.add_argument("-vs", "--vocab-size", help="vocabulary size; default: 416", type=check_positive)
    parser.add_argument("-nb", "--no-bias", help="no bias in linear layers; default: False", action="store_false")
    parser.add_argument("-dr", "--dropout", help="dropout rate; default: 0.1")
    parser.add_argument("-le", "--layernorm-eps", help="layernorm epsilon; default: 1e-6")

    args = parser.parse_args()

    # fix optional parameters
    batch_size_ = 16 if args.batch_size is None else args.batch_size
    warmup_steps_ = 2000 if args.warmup_steps is None else args.warmup_steps

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
    print("Setting up the baseline trainer...")
    trainer = MusicTransformerTrainer(hparams, args.datapath, batch_size_, warmup_steps_,
                                     args.ckpt_path, args.load_checkpoint)
    print()

    # train the model
    trainer.fit(args.epochs)

    # done training, save the model
    print("Saving final model...")
    save_file = {
        "state_dict": trainer.model.state_dict(),
        "hparams": trainer.hparams
    }
    torch.save(save_file, args.save_path)
