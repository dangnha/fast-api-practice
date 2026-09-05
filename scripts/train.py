"""Train the tiny MNIST CNN and save weights to ``models/mnist_cnn.pt``.

Run once to produce the artifact the API loads at startup:
    python -m scripts.train
"""
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from app.ml import MnistCNN


def main() -> None:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_data = datasets.MNIST(
        root="data", train=True, download=True, transform=transform
    )
    test_data = datasets.MNIST(
        root="data", train=False, download=True, transform=transform
    )
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1000)

    model = MnistCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 3
    model.train()
    for epoch in range(epochs):
        running = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running += loss.item() * images.size(0)
        print(f"epoch {epoch + 1}/{epochs} loss={running / len(train_data):.4f}")

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            logits = model(images)
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    print(f"accuracy={correct / total:.4f}")

    out = Path("models")
    out.mkdir(exist_ok=True)
    torch.save(model.state_dict(), out / "mnist_cnn.pt")
    print(f"saved {out / 'mnist_cnn.pt'}")


if __name__ == "__main__":
    main()
