import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import VOCDetection
from torch.utils.data import DataLoader, random_split, Subset
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from collections import defaultdict
import time
import os

# =============================================================
# 1. KONFIGURASI GLOBAL
# =============================================================
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES  = 20
BATCH_SIZE   = 16
EPOCHS       = 10
IMG_SIZE     = 224
SUBSET_SIZE  = 1000  # Gunakan subset agar cepat untuk analisis
LR_PROBE     = 1e-3
LR_FINETUNE  = 1e-4

VOC_CLASSES = [
    "aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair","cow",
    "diningtable","dog","horse","motorbike","person","pottedplant","sheep","sofa","train","tvmonitor"
]
CLASS2IDX = {c: i for i, c in enumerate(VOC_CLASSES)}

# =============================================================
# 2. DATASET & TRANSFORMS
# =============================================================
class VOCMultiLabel(torch.utils.data.Dataset):
    def __init__(self, root, year="2012", image_set="train", download=True, transform=None):
        self.base      = VOCDetection(root=root, year=year, image_set=image_set, download=download)
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, target = self.base[idx]
        if self.transform:
            img = self.transform(img)
        label = torch.zeros(NUM_CLASSES)
        objects = target["annotation"].get("object", [])
        if isinstance(objects, dict): objects = [objects]
        for obj in objects:
            name = obj["name"]
            if name in CLASS2IDX:
                label[CLASS2IDX[name]] = 1.0
        return img, label

# =============================================================
# 3. METRICS & HELPERS
# =============================================================
def compute_accuracy(logits, labels, threshold=0.5):
    preds = (torch.sigmoid(logits) >= threshold).float()
    correct = (preds == labels).all(dim=1).sum().item()
    return correct / labels.size(0)

def compute_ap(scores, labels):
    """Average Precision untuk satu kelas tunggal."""
    sorted_idx    = np.argsort(-scores)
    labels_sorted = labels[sorted_idx]
    tp_cumsum     = np.cumsum(labels_sorted)
    precision     = tp_cumsum / (np.arange(len(labels_sorted)) + 1)
    recall        = tp_cumsum / (labels_sorted.sum() + 1e-8)
    try:
        return np.trapezoid(precision, recall)
    except AttributeError:
        return np.trapz(precision, recall)

def compute_map(all_logits, all_labels):
    scores = 1 / (1 + np.exp(-all_logits))
    aps = []
    for c in range(NUM_CLASSES):
        if all_labels[:, c].sum() == 0: continue
        aps.append(compute_ap(scores[:, c], all_labels[:, c]))
    return np.mean(aps) if aps else 0.0

def build_model(strategy: str):
    model = models.resnet18(weights="DEFAULT")
    if strategy == "linear_probing":
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
        optimizer = torch.optim.Adam(model.fc.parameters(), lr=LR_PROBE)
    elif strategy == "full_finetuning":
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR_FINETUNE)
    
    model = model.to(DEVICE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    return model, optimizer, scheduler

# =============================================================
# 4. TRAINING LOOP
# =============================================================
def train_one_strategy(strategy, train_loader, val_loader, train_size, val_size):
    print(f"\n{'='*60}\nMelatih strategi: {strategy.upper()}\n{'='*60}")
    model, optimizer, scheduler = build_model(strategy)
    criterion = nn.BCEWithLogitsLoss()
    history   = defaultdict(list)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Param Dilatih: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)\n")

    for epoch in range(1, EPOCHS + 1):
        t_start = time.time()
        # --- Train ---
        model.train()
        t_loss, t_acc = 0.0, 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            t_loss += loss.item() * imgs.size(0)
            t_acc  += compute_accuracy(logits, labels) * imgs.size(0)

        # --- Val ---
        model.eval()
        v_loss, v_acc = 0.0, 0.0
        all_logits, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                logits = model(imgs)
                v_loss += criterion(logits, labels).item() * imgs.size(0)
                v_acc  += compute_accuracy(logits, labels) * imgs.size(0)
                all_logits.append(logits.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        scheduler.step()
        elapsed = time.time() - t_start
        v_map   = compute_map(np.concatenate(all_logits), np.concatenate(all_labels))

        # Record History
        history["train_loss"].append(t_loss/train_size)
        history["val_loss"].append(v_loss/val_size)
        history["train_acc"].append(t_acc/train_size)
        history["val_acc"].append(v_acc/val_size)
        history["val_map"].append(v_map)
        history["epoch_time"].append(elapsed)

        print(f"Epoch [{epoch:02d}] Val Acc: {v_acc/val_size:.4f} | mAP: {v_map:.4f} | {elapsed:.1f}s")

    return history, model

# =============================================================
# 5. MAIN EXECUTION
# =============================================================
if __name__ == '__main__':
    # --- Data Prep ---
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_ds = VOCMultiLabel(root="./data", image_set="train", download=True, transform=transform)
    if SUBSET_SIZE:
        indices = torch.randperm(len(full_ds))[:SUBSET_SIZE].tolist()
        full_ds = Subset(full_ds, indices)

    v_size = int(0.2 * len(full_ds))
    t_size = len(full_ds) - v_size
    train_ds, val_ds = random_split(full_ds, [t_size, v_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # --- Run Experiments ---
    h_probe, m_probe   = train_one_strategy("linear_probing", train_loader, val_loader, t_size, v_size)
    h_fine, m_fine     = train_one_strategy("full_finetuning", train_loader, val_loader, t_size, v_size)

    # --- Analysis & Visual ---
    threshold_80 = min(max(h_probe["val_acc"]), max(h_fine["val_acc"])) * 0.8
    
    # Plotting
    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor("#0d1117")
    gs = GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)
    
    colors = {"probe": "#58a6ff", "fine": "#3fb950", "text": "#c9d1d9", "grid": "#30363d"}
    
    def style_ax(ax, title, ylabel):
        ax.set_facecolor("#161b22")
        ax.set_title(title, color=colors["text"], fontweight="bold")
        ax.set_ylabel(ylabel, color=colors["text"])
        ax.tick_params(colors=colors["text"])
        ax.grid(True, color=colors["grid"], linestyle="--")

    # Panel 1: Loss
    ax1 = fig.add_subplot(gs[0, 0]); style_ax(ax1, "Loss History", "BCE Loss")
    ax1.plot(h_probe["val_loss"], color=colors["probe"], label="Probe Val")
    ax1.plot(h_fine["val_loss"], color=colors["fine"], label="Fine Val")
    ax1.legend()

    # Panel 2: Accuracy
    ax2 = fig.add_subplot(gs[0, 1]); style_ax(ax2, "Validation Accuracy", "Accuracy")
    ax2.plot(h_probe["val_acc"], color=colors["probe"], marker='o', label="Probe")
    ax2.plot(h_fine["val_acc"], color=colors["fine"], marker='o', label="Fine-tune")
    ax2.axhline(y=threshold_80, color="orange", linestyle=":", label="80% Threshold")
    ax2.legend()