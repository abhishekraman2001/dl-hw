import torch
import torch.nn as nn
import torch.optim as optim

# 1) The 'Classifier' model in models.py
from models import Classifier
# 2) Our classification dataset/dataloader
from datasets.classification_dataset import load_data
# 3) Provided utility for saving the model
from models import save_model

def train_epoch(model, dataloader, optimizer, loss_fn, device):
    """
    One training epoch for classification.
    """
    model.train()
    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        running_correct += (outputs.argmax(dim=1) == labels).sum().item()
        total_samples += images.size(0)

    avg_loss = running_loss / total_samples
    avg_acc = running_correct / total_samples
    return avg_loss, avg_acc

@torch.inference_mode()
def evaluate(model, dataloader, loss_fn, device):
    """
    Single evaluation epoch for classification.
    """
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
    # 1) Choose device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2) Load train and val data
    #    Adjust paths if needed, e.g. "classification_data/train"
    #    For data augmentation, set transform_pipeline="aug" for training
    train_loader = load_data(
        dataset_path="classification_data/train",
        transform_pipeline="aug",
        batch_size=64,
        shuffle=True
    )
    val_loader = load_data(
        dataset_path="classification_data/val",
        transform_pipeline="default",
        batch_size=64,
        shuffle=False
    )

    # 3) Create model, loss, optimizer
    model = Classifier().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    # 4) Train & Evaluate
    best_val_acc = 0.0
    num_epochs = 20
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f},   Val Acc: {val_acc:.4f}")

        # 5) Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print("Saving best classifier model as classifier.th")
            save_model(model)

if __name__ == "__main__":
    main()
