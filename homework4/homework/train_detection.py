import torch
import torch.nn as nn
import torch.optim as optim
from models import Detector
from datasets.road_dataset import load_data
from metrics import ConfusionMatrix


def train_epoch(model, dataloader, optimizer, seg_loss_fn, depth_loss_fn, device):
    model.train()
    running_seg_loss = 0.0
    running_depth_loss = 0.0
    cm = ConfusionMatrix(num_classes=3)

    for batch in dataloader:
        images = batch['image'].to(device)
        depths = batch['depth'].to(device)
        tracks = batch['track'].to(device)

        optimizer.zero_grad()
        logits, pred_depth = model(images)

        seg_loss = seg_loss_fn(logits, tracks)
        depth_loss = depth_loss_fn(pred_depth, depths)
        loss = seg_loss + depth_loss

        loss.backward()
        optimizer.step()

        running_seg_loss += seg_loss.item() * images.size(0)
        running_depth_loss += depth_loss.item() * images.size(0)

        cm.update(logits.argmax(dim=1).cpu(), tracks.cpu())

    avg_seg_loss = running_seg_loss / len(dataloader.dataset)
    avg_depth_loss = running_depth_loss / len(dataloader.dataset)
    iou = cm.compute()['iou']

    return avg_seg_loss, avg_depth_loss, iou

def evaluate(model, dataloader, seg_loss_fn, depth_loss_fn, device):
    model.eval()
    running_seg_loss = 0.0
    running_depth_loss = 0.0
    cm = ConfusionMatrix(num_classes=3)
    depth_errors_on_lane = []

    with torch.inference_mode():
        for batch in dataloader:
            images = batch['image'].to(device)
            depths = batch['depth'].to(device)
            tracks = batch['track'].to(device)

            logits, pred_depth = model(images)

            seg_loss = seg_loss_fn(logits, tracks)
            depth_loss = depth_loss_fn(pred_depth, depths)

            running_seg_loss += seg_loss.item() * images.size(0)
            running_depth_loss += depth_loss.item() * images.size(0)

            cm.update(logits.argmax(dim=1).cpu(), tracks.cpu())

            boundary_mask = (tracks.cpu() > 0)
            if boundary_mask.sum() > 0:
                depth_errors_on_lane.append(torch.abs(pred_depth.cpu() - depths.cpu())[boundary_mask])

    avg_seg_loss = running_seg_loss / len(dataloader.dataset)
    avg_depth_loss = running_depth_loss / len(dataloader.dataset)
    iou = cm.compute()['iou']
    lane_mae = torch.cat(depth_errors_on_lane).mean().item() if depth_errors_on_lane else 0.0

    return avg_seg_loss, avg_depth_loss, iou, lane_mae

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_loader, val_loader, _ = load_data(transform_pipeline="default")

    # Create model
    model = Detector().to(device)

    # Optimizer and losses
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    seg_loss_fn = nn.CrossEntropyLoss()
    depth_loss_fn = nn.L1Loss()

    best_iou = 0.0
    num_epochs = 25

    for epoch in range(num_epochs):
        train_seg_loss, train_depth_loss, train_iou = train_epoch(model, train_loader, optimizer, seg_loss_fn, depth_loss_fn, device)
        val_seg_loss, val_depth_loss, val_iou, lane_mae = evaluate(model, val_loader, seg_loss_fn, depth_loss_fn, device)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Seg Loss: {train_seg_loss:.4f}, Train Depth Loss: {train_depth_loss:.4f}, Train IOU: {train_iou:.4f}")
        print(f"Val Seg Loss: {val_seg_loss:.4f}, Val Depth Loss: {val_depth_loss:.4f}, Val IOU: {val_iou:.4f}, Lane MAE: {lane_mae:.4f}")

        if val_iou > best_iou:
            best_iou = val_iou
            print("Saving best detector model as detector.th")
            save_model(model)

if __name__ == "__main__":
    main()
