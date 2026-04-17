import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import Talonet
from config import Config
import os

class Trainer:
    def __init__(self, model, config, device, checkpoint_path=None, best_checkpoint_path=None):
        self.model = model
        self.config = config
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.best_checkpoint_path = best_checkpoint_path

        self.criterion = nn.BCEWithLogitsLoss()
        self.scaler = GradScaler()
        
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=config.lr, 
            weight_decay=config.weight_decay
        )

        self.scheduler = None
        self.start_epoch = 0

        self.best_val_loss = float('inf')

        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            self.load_checkpoint()

    def save_checkpoint(self, epoch, record=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict(),
            'config': self.config,
            'best_val_loss': self.best_val_loss
        }
        torch.save(checkpoint, self.checkpoint_path)
        print(f"Saved checkpoint from epoch {epoch} to {self.checkpoint_path}")

        if record:
            torch.save(checkpoint, self.best_checkpoint_path)
            print(f"New record! Saved to {self.best_checkpoint_path}")

    def load_checkpoint(self):
        print(f"Loading checkpoint: {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        
        if 'best_val_loss' in checkpoint:
            self.best_val_loss = checkpoint['best_val_loss']
            
        self.loaded_scheduler_state = checkpoint.get('scheduler_state_dict')

    def train(self, dataset, num_epochs):
        if self.scheduler is None:
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.config.lr,
                steps_per_epoch=len(dataset),
                epochs=num_epochs,
                pct_start=0.1,
                anneal_strategy='cos'
            )

            if hasattr(self, 'loaded_scheduler_state') and self.loaded_scheduler_state:
                self.scheduler.load_state_dict(self.loaded_scheduler_state)
                self.loaded_scheduler_state = None

        for epoch in range(self.start_epoch, num_epochs):
            print(f"\nStarting epoch {epoch}.")

            train_loader, val_loader = dataset.generate_data() #Todo
            avg_train_loss = self.train_epoch(train_loader)
            avg_val_loss, val_acc = self.validate(val_loader)

            print(f"Epoch done! \n Training Loss: {avg_train_loss} \n Validation Loss: {avg_val_loss} \n Validation Accuracy: {val_acc}")

            new_record = avg_val_loss < self.best_val_loss
            if new_record: self.best_val_loss = avg_val_loss
            self.save_checkpoint(epoch, record=new_record)

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        
        loop = tqdm(loader, desc=f"Training", leave=False)

        for waveforms, labels in loop:
            waveforms = waveforms.to(self.device)
            labels = labels.to(self.device)

            with torch.amp.autocast(device_type=self.device.type if 'cuda' in self.device.type else 'cpu'):
                output = self.model(waveforms)
                loss = self.criterion(output, labels)

            self.optimizer.zero_grad()

            self.scaler.scale(loss).backward()

            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            if self.scheduler:
                self.scheduler.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        return total_loss / len(loader)
    
    @torch.no_grad()
    def validate(self, loader):
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_samples = 0
        
        loop = tqdm(loader, desc="Validating", leave=False, colour="blue")

        for waveforms, labels in loop:
            waveforms = waveforms.to(self.device)
            labels = labels.to(self.device)

            with torch.amp.autocast(device_type=self.device.type if 'cuda' in self.device.type else 'cpu'):
                output = self.model(waveforms)
                loss = self.criterion(output, labels)

            total_loss += loss.item()

            _, predicted = torch.max(output, 1)
            _, target = torch.max(labels, 1)
            
            correct_predictions += (predicted == target).sum().item()
            total_samples += labels.size(0)

            loop.set_postfix(val_loss=loss.item())

        avg_loss = total_loss / len(loader)
        accuracy = correct_predictions / total_samples
        
        return avg_loss, accuracy