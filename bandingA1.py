import torch
import torch.nn as nn
import torchvision.models as models
import timm
import time
import numpy as np
import matplotlib
# Gunakan 'Agg' agar stabil saat simpan gambar di berbagai OS
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# =============================================================
# KONFIGURASI
# =============================================
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 20
IMG_SIZE    = 224
N_WARMUP    = 10
N_RUNS      = 100
BATCH_SIZE  = 1

# =============================================================
# FUNGSI PEMBANTU (HELPER)
# =============================================================
def count_parameters(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def measure_latency(model, input_tensor, n_warmup=N_WARMUP, n_runs=N_RUNS):
    model.eval()
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(input_tensor)

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            _ = model(input_tensor)
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            end   = time.perf_counter()
            times.append((end - start) * 1000)  # ms
    return times

# =============================================================
# EKSEKUSI UTAMA
# =============================================================
if __name__ == '__main__':
    print(f"Device yang digunakan : {DEVICE}")
    print(f"PyTorch version       : {torch.__version__}")
    print("=" * 60)

    # 1. MEMUAT BACKBONE
    print("Memuat backbone...")
    # ResNet-18
    resnet18_full   = models.resnet18(weights="DEFAULT")
    backbone_resnet = nn.Sequential(*list(resnet18_full.children())[:-2]).to(DEVICE).eval()
    
    # EfficientNet-B0
    backbone_effnet = timm.create_model("efficientnet_b0", pretrained=True, features_only=True).to(DEVICE).eval()

    # 2. HITUNG PARAMETER
    total_res, train_res = count_parameters(backbone_resnet)
    total_eff, train_eff = count_parameters(backbone_effnet)
    
    resnet18_full_params, _ = count_parameters(resnet18_full)
    effnet_full = timm.create_model("efficientnet_b0", pretrained=True, num_classes=NUM_CLASSES)
    effnet_full_params, _   = count_parameters(effnet_full)

    # 3. UKUR LATENCY
    dummy_input = torch.randn(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
    print(f"Mengukur latency (batch_size={BATCH_SIZE})...")
    
    times_res = measure_latency(backbone_resnet, dummy_input)
    times_eff = measure_latency(backbone_effnet, dummy_input)
    
    lat_res_mean, lat_res_std = np.mean(times_res), np.std(times_res)
    lat_eff_mean, lat_eff_std = np.mean(times_eff), np.std(times_eff)

    # 4. TAMPILKAN OUTPUT TEKS (ANALISIS)
    print("\n" + "=" * 60)
    print(f"{'METRIK':<25} {'RESNET-18':<15} {'EFFNET-B0':<15}")
    print("-" * 60)
    print(f"{'Param Backbone (M)':<25} {total_res/1e6:<15.2f} {total_eff/1e6:<15.2f}")
    print(f"{'Param Full Model (M)':<25} {resnet18_full_params/1e6:<15.2f} {effnet_full_params/1e6:<15.2f}")
    print(f"{'Latency Mean (ms)':<25} {lat_res_mean:<15.2f} {lat_eff_mean:<15.2f}")
    print(f"{'Speedup':<25} {'1.00x':<15} {lat_res_mean/lat_eff_mean:<15.2f}x")
    print("=" * 60)

    # 5. VISUALISASI
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor("#0f1117")
    gs  = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.3)
    
    COLOR_RES, COLOR_EFF = "#4FC3F7", "#A5D6A7"
    TEXT_COL, BG_AX = "#E0E0E0", "#1a1d27"

    def style_plot(ax, title, ylabel):
        ax.set_facecolor(BG_AX)
        ax.set_title(title, color=TEXT_COL, fontsize=11, fontweight="bold")
        ax.set_ylabel(ylabel, color=TEXT_COL)
        ax.tick_params(colors=TEXT_COL)
        for spine in ax.spines.values(): spine.set_color("#444")

    # Plot 1: Param Backbone
    ax1 = fig.add_subplot(gs[0, 0])
    style_plot(ax1, "Param Backbone (Juta)", "Mio Params")
    ax1.bar(["ResNet-18", "EffNet-B0"], [total_res/1e6, total_eff/1e6], color=[COLOR_RES, COLOR_EFF])

    # Plot 2: Latency
    ax2 = fig.add_subplot(gs[0, 1])
    style_plot(ax2, "Latency Mean (ms)", "Milidetik")
    ax2.bar(["ResNet-18", "EffNet-B0"], [lat_res_mean, lat_eff_mean], yerr=[lat_res_std, lat_eff_std], color=[COLOR_RES, COLOR_EFF], capsize=10)

    # Plot 3: Distribusi Latency
    ax3 = fig.add_subplot(gs[1, :2])
    style_plot(ax3, "Distribusi Latency", "Frekuensi")
    ax3.hist(times_res, bins=30, alpha=0.6, color=COLOR_RES, label="ResNet-18")
    ax3.hist(times_eff, bins=30, alpha=0.6, color=COLOR_EFF, label="EffNet-B0")
    ax3.legend()

    # Plot 4: Scorecard (Tabel)
    ax4 = fig.add_subplot(gs[:, 2])
    ax4.axis('off')
    table_data = [
        ["Fitur", "ResNet-18", "EffNet-B0"],
        ["Params (M)", f"{total_res/1e6:.2f}", f"{total_eff/1e6:.2f}"],
        ["Latency (ms)", f"{lat_res_mean:.2f}", f"{lat_eff_mean:.2f}"],
        ["Efisiensi", "Standard", "High"]
    ]
    tbl = ax4.table(cellText=table_data, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 2)

    plt.suptitle("Analisis A1: ResNet-18 vs EfficientNet-B0", color="white", fontsize=16, fontweight="bold")
    plt.savefig("A1_backbone_comparison.png", dpi=150, facecolor=fig.get_facecolor())
    print("\n[INFO] Plot analisis berhasil disimpan ke: A1_backbone_comparison.png")
    
    # 6. KESIMPULAN OTOMATIS
    print("\nKESIMPULAN ANALISIS:")
    if total_eff < total_res:
        print(f"1. EfficientNet-B0 lebih ringan ({total_eff/1e6:.2f}M params) dibanding ResNet-18 ({total_res/1e6:.2f}M params).")
    
    if lat_eff_mean < lat_res_mean:
        print(f"2. EfficientNet-B0 lebih cepat {lat_res_mean/lat_eff_mean:.2f}x pada device {DEVICE}.")
    else:
        print(f"2. ResNet-18 lebih cepat pada CPU x86 (Optimasi konvolusi standar).")
    
    print("3. Untuk Mobile/Embedded, EfficientNet-B0 adalah pilihan unggul karena efisiensi parameter.")