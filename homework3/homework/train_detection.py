# train_detection.py

import torch
import torch.nn as nn
import torch.optim as optim
from homework.models import Detector
from homework.datasets.drive_dataset import load_data
from homework.metrics import mean_iou, depth_mae
from homework.utils import save_model


def train_detection_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Detector().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    seg_loss_fn = nn.CrossEntropyLoss()
    depth_loss_fn = nn.L1Loss()

    train_loader, val_loader = load_data(batch_size=16)

    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            images = batch["image"].to(device)
            seg_labels = batch["track"].to(device)
            depth_labels = batch["depth"].to(device)

            optimizer.zero_grad()
            seg_logits, depth_pred = model(images)

            loss_seg = seg_loss_fn(seg_logits, seg_labels)
            loss_depth = depth_loss_fn(depth_pred, depth_labels)

            loss = loss_seg + loss_depth
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            iou_total = 0.0
            depth_error_total = 0.0
            count = 0
            for batch in val_loader:
                images = batch["image"].to(device)
                seg_labels = batch["track"].to(device)
                depth_labels = batch["depth"].to(device)

                seg_logits, depth_pred = model(images)

                iou_total += mean_iou(seg_logits, seg_labels).item() * images.size(0)
                depth_error_total += depth_mae(depth_pred, depth_labels).item() * images.size(0)
                count += images.size(0)

            avg_iou = iou_total / count
            avg_depth_error = depth_error_total / count

        print(f"Epoch [{epoch+1}/{num_epochs}] mIoU: {avg_iou:.4f}, Depth MAE: {avg_depth_error:.4f}")

    save_model(model, 'detector.pt')


if __name__ == "__main__":
    train_detection_model()