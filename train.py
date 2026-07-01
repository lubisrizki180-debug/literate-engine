import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from tqdm import tqdm

def create_data_loaders(batch_size):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    trainset = datasets.CIFAR10(root='./data', train=True, 
                               download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, 
                                            batch_size=batch_size, 
                                            shuffle=True)
    
    testset = datasets.CIFAR10(root='./data', train=False,
                              download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset,
                                           batch_size=batch_size,
                                           shuffle=False)
    
    return trainloader, testloader

def evaluate_model(model, testloader, device):
    model.eval()
    correct = 0
    total = 0
    test_loss = 0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    avg_test_loss = test_loss / len(testloader)
    return accuracy, avg_test_loss

def main():
    # Get hyperparameters from environment
    learning_rate = float(os.environ.get("LEARNING_RATE", 0.001))
    batch_size = int(os.environ.get("BATCH_SIZE", 32))
    epochs = 1  # For demonstration; increase for real training
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create data loaders
    trainloader, testloader = create_data_loaders(batch_size)
    
    # Initialize model
    model = models.resnet50(weights=None, num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    model.train()
    train_losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(enumerate(trainloader), 
                          total=len(trainloader),
                          desc=f"Epoch {epoch+1}/{epochs}",
                          unit="batch")
        
        for i, (inputs, labels) in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
        
        avg_loss = epoch_loss / len(trainloader)
        train_losses.append(avg_loss)
        tqdm.write(f"Epoch {epoch+1} completed, average loss: {avg_loss:.4f}")
    
    # Final evaluation
    accuracy, test_loss = evaluate_model(model, testloader, device)
    
    # Save metrics
    metrics = {
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "final_train_loss": train_losses[-1],
        "test_loss": test_loss,
        "accuracy": accuracy,
        "epochs": epochs
    }
    
    metrics_file = f"metrics_{learning_rate}_{batch_size}.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)
    
    print(f"\nTraining completed!")
    print(f"Final test accuracy: {accuracy:.4f}")
    print(f"Final test loss: {test_loss:.4f}")

if __name__ == "__main__":
    main()