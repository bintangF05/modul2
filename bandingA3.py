import torch
import torch.nn as nn
import math
import numpy as np

# =============================================================
# 1. KODE ASLI LISTING 1 — SimpleBackbone
# Tidak ada perubahan sama sekali
# =============================================================
class SimpleBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7)) # Fixed feature size
        )
    def forward(self, x): return self.conv(x)

# =============================================================
# 2. MODIFIKASI A3 — DetectionHead dengan prediksi orientasi
# =============================================================
class DetectionHead(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256), nn.ReLU(),
            # MODIFIKASI: Menambahkan +2 pada output layer untuk (sin theta, cos theta)
            nn.Linear(256, 4 + num_classes + 2) 
        )

    def forward(self, x):
        out = self.fc(x)

        # Pemisahan Output:
        # 0-3  : BBox (x, y, w, h)
        # 4-23 : Class Logits (20 kelas)
        # 24-25: Orientation (sin, cos)
        bbox   = out[:, :4]
        cls    = out[:, 4 : 4 + self.num_classes]
        orient = out[:, 4 + self.num_classes :]

        return bbox, cls, orient

# =============================================================
# 3. ANALISIS TEKNIS & UJI FORWARD PASS
# =============================================================
def run_analysis_a3():
    print("=" * 65)
    print("ANALISIS A3: MODIFIKASI DETECTION HEAD UNTUK ORIENTASI")
    print("=" * 65)

    backbone = SimpleBackbone()
    head     = DetectionHead(num_classes=20)
    
    # Dummy input: 4 gambar RGB 224x224
    dummy_input = torch.randn(4, 3, 224, 224)
    
    # Forward Pass
    features = backbone(dummy_input)
    bbox, cls, orient = head(features)

    print(f"1. Input Shape        : {dummy_input.shape}")
    print(f"2. Backbone Output    : {features.shape}")
    print(f"3. Head Output (BBox) : {bbox.shape} -> [x, y, w, h]")
    print(f"4. Head Output (Class): {cls.shape} -> 20 Classes")
    print(f"5. Head Output (Orient): {orient.shape} -> [sin θ, cos θ]")
    print(f"Total Output Nodes    : {4 + 20 + 2} (Bertambah 2 dari original)")

    # ---------------------------------------------------------
    # Simulasi Pemulihan Sudut (Inference)
    # ---------------------------------------------------------
    print("\n" + "-" * 65)
    print("RESTRUKTURISASI SUDUT (INFERENCE LOGIC)")
    print("-" * 65)
    
    # Menggunakan atan2 untuk mendapatkan kembali sudut dalam radian
    # atan2(y, x) -> atan2(sin, cos)
    sudut_radian = torch.atan2(orient[:, 0], orient[:, 1])
    sudut_derajat = (sudut_radian * 180.0 / math.pi) % 360

    for i in range(len(sudut_derajat)):
        s_val = orient[i, 0].item()
        c_val = orient[i, 1].item()
        print(f"Objek {i+1}: Output Raw [sin={s_val:.2f}, cos={c_val:.2f}] -> Sudut: {sudut_derajat[i]:.1f}°")

    # ---------------------------------------------------------
    # Estimasi Sudut dari BBox (Data Engineering)
    # ---------------------------------------------------------
    print("\n" + "-" * 65)
    print("ESTIMASI ORIENTASI DARI ASPECT RATIO (DATASET VOC)")
    print("-" * 65)

    def estimasi_sudut(xmin, ymin, xmax, ymax):
        w = xmax - xmin
        h = ymax - ymin
        ratio = w / (h + 1e-6)
        # Jika sangat lebar, kemungkinan besar menghadap samping (90/270 derajat)
        # Jika mendekati kotak, kemungkinan menghadap depan/belakang
        return 90.0 if ratio > 1.3 else 0.0

    test_bboxes = [
        (50, 100, 250, 150),  # Lebar (Samping)
        (100, 100, 180, 250), # Tinggi/Kotak (Depan)
        (20, 50, 400, 150)    # Sangat Lebar (Samping)
    ]

    print(f"{'W':>5} | {'H':>5} | {'Ratio':>7} | {'Estimasi Sudut'}")
    for b in test_bboxes:
        w, h = b[2]-b[0], b[3]-b[1]
        sudut = estimasi_sudut(*b)
        print(f"{w:>5} | {h:>5} | {w/h:>7.2f} | {sudut:>10.1f}°")

# =============================================================
# 4. RINGKASAN JAWABAN UNTUK LAPORAN
# =============================================================
def print_summary_a3():
    summary = f"""
{'=' * 65}
RINGKASAN JAWABAN TUGAS A3
{'=' * 65}

[1] MODIFIKASI DETECTION HEAD
    Perubahan dilakukan pada output layer (nn.Linear terakhir):
    - Original: nn.Linear(256, 4 + num_classes)
    - Modified: nn.Linear(256, 4 + num_classes + 2)

    PENJELASAN '+2':
    Kita menggunakan representasi vektor [sin θ, cos θ] alih-alih satu nilai 
    derajat langsung. Hal ini krusial karena sudut memiliki sifat sirkular. 
    Contoh: Sudut 359° dan 1° secara visual sangat dekat, namun secara numerik 
    sangat jauh. Dengan (sin, cos), nilai 359° (~ -0.01, 0.99) dan 1° 
    (~ 0.01, 0.99) memiliki jarak Euclidean yang sangat kecil, memudahkan 
    model untuk konvergen.

[2] DATA YANG DIBUTUHKAN
    A. Data Tersedia (VOC XML):
       - Bounding Box (xmin, ymin, xmax, ymax)
       - Class Label (mobil, bus, dll)
    
    B. Data Tambahan yang Diperlukan:
       - Ground Truth Sudut (θ): Orientasi rotasi objek terhadap kamera.
    
    C. Strategi Akuisisi Data:
       - Dataset KITTI: Sudah menyediakan field 'rotation_y' secara native.
       - Pseudo-labeling VOC: Menghitung rasio lebar/tinggi bbox (seperti simulasi 
         di atas) untuk mendapatkan estimasi orientasi kasar (0° atau 90°).
"""
    print(summary)

if __name__ == "__main__":
    run_analysis_a3()
    print_summary_a3()