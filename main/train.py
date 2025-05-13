# train.py
# Trains the AI image classifier using a CVT backbone and custom classifier

import argparse
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
import glob
import os

from model import get_model
from custom_dataset import get_dataset

def load_checkpoint_if_exists(model, optimizer, checkpoint_path, learning_rate):
    """
    Loads the latest checkpoint from the given path, if it exists.

    Returns:
        - starting_epoch (int)
        - train_losses (list)
    """
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return 0, []

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    train_losses = checkpoint.get('train_losses', [])
    starting_epoch = checkpoint.get('epoch', -1) + 1

    # Optionally override LR
    if learning_rate is not None:
        for g in optimizer.param_groups:
            g['lr'] = learning_rate

    print(f"✅ Loaded checkpoint from {checkpoint_path} (starting at epoch {starting_epoch})")
    return starting_epoch, train_losses

def train_model(model, train_data_dir, device, optimizer, criterion, total_epochs=50, save_path="./models", checkpoint_path=None, learning_rate=None):
    """
    Trains the model on the dataset from the specified directory.
    """
    train_data = get_dataset(train_data_dir)
    train_loader = DataLoader(train_data, batch_size=128, num_workers=6, pin_memory=True)
    scaler = GradScaler()

    # Try to resume from checkpoint
    starting_epoch, train_losses = load_checkpoint_if_exists(model, optimizer, checkpoint_path, learning_rate)

    for epoch in range(starting_epoch, total_epochs):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\n📦 Epoch {epoch+1}/{total_epochs} — Learning Rate: {current_lr:.6f}")

        model.train()
        total_loss = 0.0

        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}", unit="batch") as progress_bar:
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs.logits, labels)

                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()
                progress_bar.set_postfix({"Loss": loss.item()})
                progress_bar.update()

                # Cleanup
                del inputs, labels, outputs

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"✅ Epoch {epoch+1} completed — Avg Loss: {avg_train_loss:.4f}")

        # Save checkpoint
        os.makedirs(save_path, exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'avg_loss': avg_train_loss,
            'scaler_state_dict': scaler.state_dict(),
            'train_losses': train_losses,
        }, os.path.join(save_path, f"model_epoch_{epoch}.pth"))

        torch.cuda.empty_cache()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the image classifier model.')
    parser.add_argument('train_data_dir', type=str, help='Path to training data folder.')
    parser.add_argument('--total_epochs', type=int, default=50, help='Total number of training epochs.')
    parser.add_argument('--save_path', type=str, default="./models", help='Where to save model checkpoints.')
    parser.add_argument('--learning_rate', type=float, default=None, help='Optional: override default learning rate.')
    parser.add_argument('--model_weight_path', type=str, default=None, help='Path to resume training from a checkpoint.')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(device)

    # Default LR if not provided
    lr = args.learning_rate if args.learning_rate is not None else 1e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-5, eps=1e-8)
    criterion = torch.nn.CrossEntropyLoss()

    train_model(
        model=model,
        train_data_dir=args.train_data_dir,
        device=device,
        optimizer=optimizer,
        criterion=criterion,
        total_epochs=args.total_epochs,
        save_path=args.save_path,
        checkpoint_path=args.model_weight_path,
        learning_rate=args.learning_rate
    )
