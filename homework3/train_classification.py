import os
import torch
import torch.nn as nn
import torch.optim as optim

# 1) Our Classifier model
from homework.models import Classifier
# 2) Our classification dataset/dataloader
from homework.datasets.classification_dataset import load_data
# 3) Provided utility for saving the model (assuming it's in models.py)
from homework.models import save_model


def train_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    for images, labels in dataloader:


        images, labels = images.to(device), labels.to(device)
        outputs = model(images)


        loss = loss_fn(outputs, labels)


        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        running_loss += loss.item() * images.size(0)
        running_correct += (outputs.argmax(dim=1) == labels).sum().item()
        total_samples += images.size(0)

    avg_loss = running_loss / total_samples
    avg_acc = running_correct / total_samples
    return avg_loss, avg_acc

@torch.inference_mode()
def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        running_loss += loss.item() * images.size(0)
        running_correct += (outputs.argmax(dim=1) == labels).sum().item()
        total_samples += images.size(0)

    avg_loss = running_loss / total_samples
    avg_acc = running_correct / total_samples
    return avg_loss, avg_acc


def main():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using device: CUDA")
    else:
        device = torch.device("cpu")
        print("Using device: CPU")

    dataset_path = "classification_data"
    transform_pipeline = "aug"

    # Lower batch size in case of shape or memory issues
    batch_size = 64

    train_data = load_data(
        os.path.join(dataset_path, "train"),
        transform_pipeline=transform_pipeline,
        batch_size=batch_size,
        shuffle=True
    )
    val_data = load_data(
        os.path.join(dataset_path, "val"),
        transform_pipeline="default",
        batch_size=batch_size,
        shuffle=False
    )

    model = Classifier().to(device)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_val_acc = 0.0
    num_epochs = 20
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_data, optimizer, loss_fn, device)
        val_loss, val_acc = evaluate(model, val_data, loss_fn, device)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f},   Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print("Saving best classifier model as classifier.th")
            save_model(model)

if __name__ == "__main__":
    main()
