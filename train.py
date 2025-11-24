import os
import torch
from torch.amp import GradScaler
from torch.amp import autocast
from detection.detection_loss import set_optimizer_lr
from tqdm import tqdm

class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, epochs, device,
                 ema, lr_scheduler, fp16=True):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs = epochs
        self.device = device
        self.ema = ema
        self.lr_scheduler = lr_scheduler
        self.fp16 = fp16
        self.scaler = GradScaler(enabled=self.fp16) # for mixed precision
        self.epoch = 0
        self.steps_per_epoch = len(train_loader)

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0

        # Progress bar for training epoch
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch+1}/{self.epochs} [Train]", leave=True)

        for i, batch in enumerate(pbar):
            # set new lr for current step
            current_step = self.epoch * self.steps_per_epoch + i
            set_optimizer_lr(self.optimizer, self.lr_scheduler, current_step)

            radars, targets = batch['radar'].to(self.device), batch['label'].to(self.device)

            self.optimizer.zero_grad()

            # Mixed precision training
            with autocast(enabled=self.fp16, device_type=self.device.type):
                outputs = self.model(radars)
                loss = self.criterion(outputs, targets)

            # Mixed precision backward
            self.scaler.scale(loss).backward()

            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Update EMA model
            if self.ema is not None:
                self.ema.update(self.model)

            running_loss += loss.item()
        epoch_loss = running_loss / len(self.train_loader)
        return epoch_loss

    def validate_epoch(self):
        # Use EMA model for validation if available
        model_to_evaluate = self.ema.ema.eval() if self.ema is not None else self.model.eval()
        
        running_loss = 0.0
        with torch.no_grad():
            for batch in self.val_loader:
                radars, targets = batch['radar'].to(self.device), batch['label'].to(self.device)

                # Mixed precision inference
                with autocast(enabled=self.fp16, device_type=self.device.type):
                    outputs = model_to_evaluate(radars)
                    loss = self.criterion(outputs, targets)

                running_loss += loss.item()
        epoch_loss = running_loss / len(self.val_loader)
        return epoch_loss

    def train(self, final_model_path):
        for epoch in range(self.epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate_epoch()
            print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            self.epoch += 1
        
        # Save the final model
        if os.path.dirname(final_model_path) != '':
            os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
        torch.save(self.model.state_dict(), final_model_path)

