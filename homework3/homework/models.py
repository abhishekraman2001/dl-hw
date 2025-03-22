from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class Classifier(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 6,
    ):
        """
        A convolutional network for image classification.

        Args:
            in_channels: int, number of input channels
            num_classes: int
        """
        super().__init__()

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),


            nn.Dropout(p=0.5),
        )

        self.fc = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass for classification.
        Returns (b, num_classes) logits.
        """
        # optional: normalizes input
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        # pass through conv layers
        z = self.model(z)       # e.g. shape (b, 128, 8, 8)
        z = z.view(z.size(0), -1)  # flatten to (b, 128*8*8)
        logits = self.fc(z)        # (b, 6)

        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Used for inference; returns class labels.
        This is what the AccuracyMetric uses as input (grader will use).
        You should not have to modify this function.
        """
        return self(x).argmax(dim=1)


class Detector(nn.Module):
    def __init__(self, in_channels=3, num_classes=3):
        super().__init__()

        # For optional normalization if needed
        self.register_buffer("input_mean", torch.tensor([0.2788, 0.2657, 0.2629]))
        self.register_buffer("input_std", torch.tensor([0.2064, 0.1944, 0.2252]))

        # ---------- ENCODER ----------
        self.down1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1)
        self.bn1   = nn.BatchNorm2d(32)

        self.down2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2   = nn.BatchNorm2d(64)

        self.down3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3   = nn.BatchNorm2d(128)

        # bottleneck (no further down-sampling)
        self.bottleneck = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn_bottleneck = nn.BatchNorm2d(256)

        # ---------- DECODER ----------
        # up3: from (B,256,h/8,w/8) -> (B,128,h/4,w/4) then cat with d2
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up3_conv = nn.Conv2d(128 + 64, 128, kernel_size=3, padding=1)
        self.up3_bn = nn.BatchNorm2d(128)

        # up2: from (B,128,h/4,w/4) -> (B,64,h/2,w/2) then cat with d1
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up2_conv = nn.Conv2d(64 + 32, 64, kernel_size=3, padding=1)
        self.up2_bn = nn.BatchNorm2d(64)

        # up1: from (B,64,h/2,w/2) -> (B,32,h,w) -> cat with nothing?
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.up1_conv = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.up1_bn = nn.BatchNorm2d(32)

        # ---------- HEADS ----------
        self.seg_head   = nn.Conv2d(32, num_classes, kernel_size=1)
        self.depth_head = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # optional: input normalization
        # comment out if you already do transforms.Normalize in your dataset
        x = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        # -------- Encoder --------
        d1 = F.relu(self.bn1(self.down1(x)))   # (B,32,h/2,w/2)
        d2 = F.relu(self.bn2(self.down2(d1)))  # (B,64,h/4,w/4)
        d3 = F.relu(self.bn3(self.down3(d2)))  # (B,128,h/8,w/8)

        bn = F.relu(self.bn_bottleneck(self.bottleneck(d3)))  # (B,256,h/8,w/8)

        # -------- Decoder --------
        # up3 -> cat with d2
        u3  = F.relu(self.up3(bn))                   # (B,128,h/4,w/4)
        cat3 = torch.cat([u3, d2], dim=1)            # (B,128+64=192,h/4,w/4)
        dec3 = F.relu(self.up3_bn(self.up3_conv(cat3)))  # (B,128,h/4,w/4)

        # up2 -> cat with d1
        u2  = F.relu(self.up2(dec3))                 # (B,64,h/2,w/2)
        cat2 = torch.cat([u2, d1], dim=1)            # (B,64+32=96,h/2,w/2)
        dec2 = F.relu(self.up2_bn(self.up2_conv(cat2)))  # (B,64,h/2,w/2)

        # up1 -> no skip (or if you had a d0 above, you'd cat it here)
        u1  = F.relu(self.up1(dec2))                 # (B,32,h,w)
        dec1 = F.relu(self.up1_bn(self.up1_conv(u1)))# (B,32,h,w)

        # -------- Heads --------
        logits = self.seg_head(dec1)                 # (B,num_classes,h,w)
        raw_depth = self.depth_head(dec1).squeeze(1) # (B,h,w)

        return logits, raw_depth

    @torch.no_grad()
    def predict(self, x):
        """
        Inference function: returns seg classes + 0..1 depth
        """
        logits, raw_depth = self(x)
        seg_pred = logits.argmax(dim=1)
        return seg_pred, raw_depth


# For grading + saving/loading:
MODEL_FACTORY = {
    "classifier": Classifier,
    "detector": Detector,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    model_size_mb = calculate_model_size_mb(m)
    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None
    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n
    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: nn.Module) -> float:
    """
    Returns size in MB
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024


def debug_model(batch_size: int = 1):
    """
    Quick test
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_batch = torch.rand(batch_size, 3, 64, 64).to(device)


    model = load_model("classifier", in_channels=3, num_classes=6).to(device)
    output = model(sample_batch)


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
if __name__ == "__main__":
    debug_model()
