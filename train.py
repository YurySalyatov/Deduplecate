import os.path

import torch
import torch.nn.functional as F


class Trainer:
    def __init__(self, model, optimizer, device):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device

    def train_epoch(self, data, mask_train):
        self.model.train()
        self.optimizer.zero_grad()

        # Forward pass
        out = self.model(data.x_dict, data.edge_index_dict,
                         data['author', 'duplicate', 'author'].edge_index)[mask_train]

        # Compute loss
        target = data['author', 'duplicate', 'author'].edge_label[mask_train]
        loss = F.binary_cross_entropy(out, target)

        # Backward pass
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def evaluate(self, data, mask):
        self.model.eval()
        with torch.no_grad():
            out = self.model(data.x_dict, data.edge_index_dict,
                             data['author', 'duplicate', 'author'].edge_index)
            target = data['author', 'duplicate', 'author'].edge_label
            out = out[mask]
            target = target[mask]

            loss = F.binary_cross_entropy(out, target)
            predictions = (out > 0.5).float()
            accuracy = (predictions == target).float().mean()

        return loss.item(), accuracy.item()


def train_model(model, data, epochs, learning_rate, weight_decay, device, save_root_path):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=weight_decay)
    model.to(device)
    trainer = Trainer(model, optimizer, device)
    save_path = os.path.join(save_root_path, "best_model.pt")

    best_val_loss = float('inf')
    for epoch in range(epochs):
        train_loss = trainer.train_epoch(data, data.train_mask)
        val_loss, val_acc = trainer.evaluate(data, data.val_mask)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc
            }, save_path)

        if epoch % 10 == 0:
            print(f'Epoch {epoch:03d}, Train Loss: {train_loss:.4f}, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'\nLoaded best model from epoch {checkpoint["epoch"]} with val_loss = {checkpoint["val_loss"]:.4f}')

    # Тестирование на test_mask
    test_loss, test_acc = trainer.evaluate(data, data.test_mask)
    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')

    return model, test_loss, test_acc

