import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.fc1 = nn.Linear(26*26*32, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = x.view(-1, 26*26*32)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

def train_serial(epochs=5):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    losses = []
    start_time = time.time()
    for epoch in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f"[Serial] Epoch {epoch+1}, Loss: {loss.item():.4f}")
    end_time = time.time()
    training_time = end_time - start_time

    # Accuracy evaluation
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"[Serial] Test Accuracy: {accuracy:.2f}%")

    # Save loss curve
    plt.plot(losses, label="Serial Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Serial Training Loss Curve")
    plt.legend()
    plt.savefig("results/loss_curves_serial.png")
    plt.close()

    # Save training time and accuracy
    with open("results/training_times.txt", "a") as f:
        f.write(f"Serial Training Time: {training_time:.2f} seconds | Accuracy: {accuracy:.2f}%\n")

if __name__ == "__main__":
    train_serial(epochs=5)
