import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
from serial_cnn import SimpleCNN

def train_parallel():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(SimpleCNN()).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    losses = []
    start_time = time.time()
    for epoch in range(2):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f"[Parallel] Epoch {epoch+1}, Loss: {loss.item():.4f}")
    end_time = time.time()
    training_time = end_time - start_time

    # Accuracy evaluation
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"[Parallel] Test Accuracy: {accuracy:.2f}%")

    # Save loss curve
    plt.plot(losses, label="Parallel Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Parallel Training Loss Curve")
    plt.legend()
    plt.savefig("results/loss_curves_parallel.png")
    plt.close()

    # Save training time and accuracy
    with open("results/training_times.txt", "a") as f:
        f.write(f"Parallel Training Time: {training_time:.2f} seconds | Accuracy: {accuracy:.2f}%\n")

if __name__ == "__main__":
    train_parallel()
