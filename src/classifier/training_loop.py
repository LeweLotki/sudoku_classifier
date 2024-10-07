import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.notebook import tqdm

def train(model, criterion, optimizer, data_loader, epochs):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(total=len(data_loader), desc=f'Epoch {epoch + 1}/{epochs}', unit='batch', leave=True)

        for features, target in data_loader:
            optimizer.zero_grad()
            output = model(features)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

            progress_bar.update(1)  
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

def evaluate(model, criterion, data_loader):
    model.eval() 
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad(): 
        for features, target in data_loader:
            output = model(features)
            loss = criterion(output, target)
            total_loss += loss.item()

            _, predicted = torch.max(output, 1)
            _, target_labels = torch.max(target, 1)
            correct += (predicted == target_labels).sum().item()
            total += target.size(0)

    average_loss = total_loss / len(data_loader)
    accuracy = 100 * correct / total

    print(f'Average Loss: {average_loss:.4f}, Accuracy: {accuracy:.2f}%')
    return average_loss, accuracy
