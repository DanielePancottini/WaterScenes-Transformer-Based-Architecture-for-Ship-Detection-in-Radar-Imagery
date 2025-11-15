import torch

class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, epochs, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs = epochs
        self.device = device

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        for batch in self.train_loader:
            radars, targets = batch['radar'].to(self.device), batch['labels'].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(radars)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
        epoch_loss = running_loss / len(self.train_loader)
        return epoch_loss

    def validate_epoch(self):
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for batch in self.val_loader:
                radars, targets = batch['radar'].to(self.device), batch['labels'].to(self.device)

                outputs = self.model(radars)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item()
        epoch_loss = running_loss / len(self.val_loader)
        return epoch_loss

    def train(self):
        for epoch in range(self.epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate_epoch()
            print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

