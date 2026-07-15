import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import VOCDetection
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# =============================================================
# 1. KONFIGURASI
# =============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 20  # Pascal VOC memiliki 20 kelas objek [cite: 397-398]
BATCH_SIZE = 4   # Gunakan 4 jika RAM laptop terbatas
EPOCHS = 10
LR = 1e-3
IMG_SIZE = 126

VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]
CLASS2IDX = {c: i for i, c in enumerate(VOC_CLASSES)}

# =============================================================
# 2. TRANSFORMASI
# =============================================================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], # ImageNet mean [cite: 441-443]
        std=[0.229, 0.224, 0.225]   # ImageNet std [cite: 445-447]
    )
])

# =============================================================
# 3. DATASET
# =============================================================
class VOCMultiLabel(torch.utils.data.Dataset):
    """Membungkus VOCDetection untuk vektor multi-label binary."""
    def __init__(self, root, year="2012", image_set="train", download=True, transform=None):
        self.base = VOCDetection(
            root=root, year=year, image_set=image_set, download=download
        )
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, target = self.base[idx]
        if self.transform:
            img = self.transform(img)
            
        # Membangun vektor multi-label (shape: 20,) [cite: 460, 498]
        label = torch.zeros(NUM_CLASSES)
        objects = target["annotation"].get("object", [])
        
        if isinstance(objects, dict):  # Kasus jika hanya ada 1 objek
            objects = [objects]
            
        for obj in objects:
            name = obj["name"]
            if name in CLASS2IDX:
                label[CLASS2IDX[name]] = 1.0
        return img, label

# Load dan bagi dataset (80% Train, 20% Val)
full_dataset = VOCMultiLabel(
    root="./data", year="2012", image_set="train", download=True, transform=transform
)
val_size = int(0.2 * len(full_dataset))
train_size = len(full_dataset) - val_size
train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# =============================================================
# 4. MODEL - Linear Probing dengan ResNet-18
# =============================================================
model = models.resnet18(weights="DEFAULT")

# Freeze semua parameter backbone [cite: 556-560]
for param in model.parameters():
    param.requires_grad = False

# Ganti FC head terakhir (otomatis requires_grad=True) [cite: 563-565]
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(DEVICE)

# =============================================================
# 5. LOSS DAN OPTIMIZER
# =============================================================
criterion = nn.BCEWithLogitsLoss() # Tepat untuk multi-label [cite: 578-579]
optimizer = torch.optim.Adam(model.fc.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

# =============================================================
# 6. FUNGSI PEMBANTU (METRICS)
# =============================================================
def compute_accuracy(logits, labels, threshold=0.5):
    """Exact-match accuracy: semua label dalam satu gambar harus benar."""
    preds = (torch.sigmoid(logits) >= threshold).float()
    correct = (preds == labels).all(dim=1).sum().item()
    return correct / labels.size(0)

def compute_ap(scores, labels):
    """Average Precision untuk satu kelas tunggal."""
    sorted_idx = np.argsort(-scores)
    labels_sorted = labels[sorted_idx]
    tp_cumsum = np.cumsum(labels_sorted)
    precision = tp_cumsum / (np.arange(len(labels_sorted)) + 1)
    recall = tp_cumsum / (labels_sorted.sum() + 1e-8)
    return np.trapezoid(precision, recall)

def compute_map(all_logits, all_labels):
    """Mean Average Precision untuk 20 kelas Pascal VOC."""
    scores = 1 / (1 + np.exp(-all_logits))  # Sigmoid [cite: 638]
    aps = []
    for c in range(NUM_CLASSES):
        if all_labels[:, c].sum() == 0:  # Lewati jika tidak ada ground truth
            continue
        aps.append(compute_ap(scores[:, c], all_labels[:, c]))
    return np.mean(aps) if aps else 0.0

# =============================================================
# 7. LOOP TRAINING DAN VALIDASI
# =============================================================
history = defaultdict(list)

for epoch in range(1, EPOCHS + 1):
    # -- Fase Training --
    model.train()
    train_loss, train_acc = 0.0, 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * imgs.size(0)
        train_acc += compute_accuracy(logits, labels) * imgs.size(0)

    train_loss /= train_size
    train_acc /= train_size

    # -- Fase Validasi --
    model.eval()
    val_loss, val_acc = 0.0, 0.0
    all_logits, all_labels_np = [], []
    
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            logits = model(imgs)
            loss = criterion(logits, labels)
            
            val_loss += loss.item() * imgs.size(0)
            val_acc += compute_accuracy(logits, labels) * imgs.size(0)
            all_logits.append(logits.cpu().numpy())
            all_labels_np.append(labels.cpu().numpy())

    val_loss /= val_size
    val_acc /= val_size
    val_map = compute_map(
        np.concatenate(all_logits, axis=0),
        np.concatenate(all_labels_np, axis=0)
    )
    
    scheduler.step()

    # Simpan history
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)
    history["val_map"].append(val_map)

    print(f"Epoch [{epoch:02d}/{EPOCHS}] | "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} mAP: {val_map:.4f}")

# =============================================================
# 8. VISUALISASI
# =============================================================
epochs_range = range(1, EPOCHS + 1)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot Loss
axes[0].plot(epochs_range, history["train_loss"], label="Train", marker="o")
axes[0].plot(epochs_range, history["val_loss"], label="Val", marker="s")
axes[0].set_title("Loss History")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("BCE Loss")
axes[0].legend(); axes[0].grid(True)

# Plot Accuracy
axes[1].plot(epochs_range, history["train_acc"], label="Train", marker="o")
axes[1].plot(epochs_range, history["val_acc"], label="Val", marker="s")
axes[1].set_title("Accuracy History (Exact Match)")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].legend(); axes[1].grid(True)

# Plot mAP
axes[2].plot(epochs_range, history["val_map"], label="Val mAP", marker="^", color="green")
axes[2].set_title("mAP History")
axes[2].set_xlabel("Epoch")
axes[2].set_ylabel("mAP")
axes[2].legend(); axes[2].grid(True)

plt.tight_layout()
plt.savefig("training_history.png", dpi=150)
plt.show()