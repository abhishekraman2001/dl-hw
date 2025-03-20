import torch
import torch.nn as nn
import torch.optim as optim
from homework.models import Classifier
from homework.datasets.classification_datasets import load_data
from homework.metrics import accuracy
from homework.utils import save_model


def train_classification_model():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    model = Classifier().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    train_loader, val_loader = load_data(batch_size=64, transform_pipeline='default')

    num_epochs = 15
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            images, labels = batch["image"].to(device), batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_acc = 0.0
            total = 0
            for batch in val_loader:
                images, labels = batch["image"].to(device), batch["label"].to(device)
                outputs = model(images)
                val_acc += accuracy(outputs, labels) * images.size(0)
                total += images.size(0)

            val_acc /= total

        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {running_loss/len(train_loader):.4f} Val Acc: {val_acc:.4f}")

    save_model(model, 'classifier.pt')


if __name__ == "__main__":
    train_classification_model()