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
    y_pred = []
    y_prob = []
    y_true = []

    with torch.no_grad(): 
        for features, target in data_loader:
            output = model(features)
            loss = criterion(output, target)
            total_loss += loss.item()

            _, preds = torch.max(output, 1)

            probs = torch.softmax(output, dim=1)[:, 1]  

            if target.ndim > 1:  
                true_labels = torch.argmax(target, dim=1)
            else:  
                true_labels = target

            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())
            y_true.extend(true_labels.cpu().numpy())

            correct += (preds == true_labels).sum().item()
            total += true_labels.size(0)

    average_loss = total_loss / len(data_loader)
    accuracy = 100 * correct / total

    print(f'Average Loss: {average_loss:.4f}, Accuracy: {accuracy:.2f}%')
    return average_loss, accuracy, y_pred, y_prob, y_true

