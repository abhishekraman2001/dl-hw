# train_detection.py
import torch
import torch.nn as nn
import torch.optim as optim

# 1) Our Detector model
from homework.models import Detector, save_model
# 2) Our drive dataset/dataloader
from homework.datasets.road_dataset import load_data
# 3) Provided metrics for segmentation and depth
from homework.metrics import ConfusionMatrix

##################################
# IoU loss function
##################################
import torch.nn.functional as F

def iou_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-7):
    """
    logits: shape (B, num_classes, H, W)
    targets: shape (B, H, W) in {0..num_classes-1}
    """
    probs = F.softmax(logits, dim=1)
    num_classes = logits.shape[1]

    # One-hot encode targets: (B,H,W) -> (B,H,W,C) -> (B,C,H,W)
    targets_oh = F.one_hot(targets, num_classes=num_classes).permute(0,3,1,2).float()

    intersection = (probs * targets_oh).sum(dim=(2,3))
    union        = (probs + targets_oh).sum(dim=(2,3)) - intersection
    iou_per_class = (intersection + eps) / (union + eps)
    # average across classes and batch
    iou_val = iou_per_class.mean()
    return 1.0 - iou_val


##################################
# One epoch of training
##################################
def train_one_epoch(model, dataloader, optimizer,
                    ce_loss_fn, depth_loss_fn,
                    device, alpha=1.0, beta=1.0, gamma=0.5):
    """
    alpha = CrossEntropy seg loss weight
    beta  = Depth loss weight
    gamma = IoU loss weight
    """
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch in dataloader:
        images  = batch["image"].to(device)
        seg_t   = batch["track"].to(device)
        depth_t = batch["depth"].to(device)

        optimizer.zero_grad()
        logits, depth_pred = model(images)

        # Weighted cross-entropy
        seg_ce  = ce_loss_fn(logits, seg_t)
        # IoU loss
        seg_iou = iou_loss(logits, seg_t)
        # Depth
        depth_l = depth_loss_fn(depth_pred, depth_t)

        # combine them
        loss = alpha * seg_ce + beta * depth_l + gamma * seg_iou
        loss.backward()
        optimizer.step()

        bsize = images.size(0)
        total_loss += loss.item() * bsize
        total_samples += bsize

    return total_loss / total_samples


##################################
# Evaluation loop
##################################
@torch.inference_mode()
def evaluate(model, dataloader, ce_loss_fn, depth_loss_fn, device):
    model.eval()
    total_seg_loss = 0.0
    total_depth_loss = 0.0
    total_samples = 0

    cm = ConfusionMatrix(num_classes=3)

    for batch in dataloader:
        images  = batch["image"].to(device)
        seg_t   = batch["track"].to(device)
        depth_t = batch["depth"].to(device)

        logits, depth_pred = model(images)

        seg_ce  = ce_loss_fn(logits, seg_t)
        depth_l = depth_loss_fn(depth_pred, depth_t)

        bsize = images.size(0)
        total_seg_loss += seg_ce.item() * bsize
        total_depth_loss += depth_l.item() * bsize
        total_samples += bsize

        preds = logits.argmax(dim=1).cpu()
        cm.add(preds, seg_t.cpu())

    avg_seg_loss   = total_seg_loss / total_samples
    avg_depth_loss = total_depth_loss / total_samples
    metrics        = cm.compute()  # includes 'iou', 'accuracy'
    iou_val        = metrics["iou"]

    return avg_seg_loss, avg_depth_loss, iou_val


##################################
# Main training routine
##################################
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Data
    train_dl = load_data("drive_data/train", transform_pipeline="default",
                         batch_size=8, shuffle=True)
    val_dl   = load_data("drive_data/val",   transform_pipeline="default",
                         batch_size=8, shuffle=False)

    # Model
    model = Detector().to(device)

    # Weighted cross-entropy (if needed)
    # Example: background=0.5, road=1.0, boundary=2.0
    class_weights = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float).to(device)
    ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    # Depth loss
    depth_loss_fn = nn.L1Loss()

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

    best_iou = 0.0
    patience = 5
    no_improve = 0
    epochs = 30

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model, train_dl, optimizer,
            ce_loss_fn, depth_loss_fn,
            device,
            alpha=1.0,     # seg ce
            beta=0.5,      # depth
            gamma=0.5,     # iou
        )

        seg_val_loss, depth_val_loss, iou_val = evaluate(
            model, val_dl, ce_loss_fn, depth_loss_fn, device
        )

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val => seg: {seg_val_loss:.4f}, depth: {depth_val_loss:.4f}, iou: {iou_val:.4f}")

        # track best IoU
        if iou_val > best_iou:
            best_iou = iou_val
            no_improve = 0
            print(f"=> Best model updated (IoU={best_iou:.4f})")
            # Save
            save_model(model)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"=> Early stopping at epoch {epoch+1}")
                break

    print(f"Done! Best IoU = {best_iou:.4f}")


if __name__ == "__main__":
    main()
