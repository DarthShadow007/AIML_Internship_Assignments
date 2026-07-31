import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

# 1. Dataset Handler (Simulating MRI data so it runs instantly without huge downloads)
class SyntheticMRIDataset(Dataset):
    def __init__(self, num_samples=200, transform=None):
        self.num_samples = num_samples
        self.transform = transform
        # Simulating 64x64 grayscale MRI scans
        self.images = [np.random.randint(0, 255, (64, 64), dtype=np.uint8) for _ in range(num_samples)]
        # Binary classification: 0 (Benign/No Cancer), 1 (Malignant/Cancer)
        self.labels = [np.random.randint(0, 2) for _ in range(num_samples)]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img = Image.fromarray(self.images[idx], mode='L')
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

# 2. CNN Architecture for Image Classification
class CancerDetectionCNN(nn.Module):
    def __init__(self):
        super(CancerDetectionCNN, self).__init__()
        # 1 input channel (grayscale MRI), 16 output channels, 3x3 kernel
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 16 * 16) # Flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def main():
    print("Initializing MRI Cancer Detection Pipeline...")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    print("Generating Synthetic MRI Dataset...")
    # Using 800 training and 200 testing samples
    trainset = SyntheticMRIDataset(num_samples=800, transform=transform)
    testset = SyntheticMRIDataset(num_samples=200, transform=transform)
    
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
    testloader = DataLoader(testset, batch_size=32, shuffle=False)

    device = torch.device("cpu") # Defaulting to CPU for stability
    model = CancerDetectionCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Training CNN Model...")
    epochs = 3
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(trainloader):.4f}")

    print("\nEvaluating Model on Test MRI Scans...")
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print("\n" + "="*50)
    print("      MRI CANCER DETECTION RESULTS")
    print("="*50)
    print(f"Test Accuracy: {accuracy:.2f}%")
    print("Status: Model successfully compiled and evaluated.")
    print("="*50)

if __name__ == "__main__":
    main()