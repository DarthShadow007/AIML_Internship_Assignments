import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

# 1. Dataset Pipeline for Wildlife Face Features (5 species/individual classes)
class WildlifeFaceDataset(Dataset):
    def __init__(self, num_samples=300, num_classes=5, transform=None):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.transform = transform
        # 64x64 RGB images simulating wildlife facial features
        self.images = [np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(num_samples)]
        self.labels = [np.random.randint(0, num_classes) for _ in range(num_samples)]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img = Image.fromarray(self.images[idx], mode='RGB')
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

# 2. Convolutional Neural Network Architecture
class WildlifeFaceCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(WildlifeFaceCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 16 * 16)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def main():
    print("Initializing Wildlife Face Recognition Pipeline...")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    print("Generating Wildlife Face Dataset (5 Animal Classes)...")
    trainset = WildlifeFaceDataset(num_samples=800, num_classes=5, transform=transform)
    testset = WildlifeFaceDataset(num_samples=200, num_classes=5, transform=transform)

    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
    testloader = DataLoader(testset, batch_size=32, shuffle=False)

    device = torch.device("cpu")
    model = WildlifeFaceCNN(num_classes=5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Training CNN Model for Wildlife Face Recognition...")
    epochs = 3
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(trainloader):.4f}")

    print("\nEvaluating Model on Test Wildlife Scans...")
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print("\n" + "="*50)
    print("     WILDLIFE FACE RECOGNITION RESULTS")
    print("="*50)
    print(f"Test Accuracy: {accuracy:.2f}%")
    print("Status: Model successfully compiled and evaluated.")
    print("="*50)

if __name__ == "__main__":
    main()