import os
import yaml
from ultralytics import YOLO
from multiprocessing import freeze_support

# ==============================
# PATH SETTING (SUDAH DIPERBAIKI KE FOLDER BARU)
# ==============================
# Arahkan ke folder 80:10:10 yang baru kita buat tadi
dataset_path = r"C:\Users\Ivan\OneDrive\Documents\Skripsi_Malaria\dataset_final_malaria"
project_save_path = r"D:\Skripsi_Malaria_Output"
yaml_path = os.path.join(dataset_path, "malaria_final.yaml")

# ==============================
# MAIN TRAINING FUNCTION
# ==============================
def run_training():

    print("🚀 MENJALANKAN FINAL TRAINING 80:10:10 (RESEP TRIAL 35)...")

    # Buat YAML dataset secara otomatis (Termasuk folder 'test')
    yaml_content = {
        'path': dataset_path,
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',  # Tambahan krusial untuk evaluasi akhir!
        'names': {0: 'P. falciparum', 1: 'P. vivax'}
    }

    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
        print(f"✅ File konfigurasi dataset berhasil dibuat di: {yaml_path}")

    # Load model pretrained (Model Dasar COCO, bukan model Optuna)
    model = YOLO("yolov8m.pt")

    # Training dengan Hyperparameter Trial 35
    model.train(
        data=yaml_path,
        epochs=100,            # Sesuai Trial 35
        imgsz=640,
        batch=8,
        device=0,             # Menggunakan GPU RTX 5060
        workers=4,
        optimizer="AdamW",
        
        # --- RESEP SAKTI TRIAL 35 ---
        lr0=0.000619,
        lrf=0.02869,
        momentum=0.9312,
        weight_decay=1.7275e-06,
        warmup_epochs=7.5,
        degrees=80.0,
        translate=0.0847,
        scale=0.4046,
        hsv_s=0.5778,
        hsv_v=0.5506,
        mosaic=0.4956,
        mixup=0.2869,
        
        # --- SETTINGAN WAJIB MEDIS & STABILITAS ---
        cos_lr=True,          
        fliplr=0.5,           # Wajib untuk sel darah
        flipud=0.5,           # Wajib untuk sel darah
        
        # --- OUTPUT & SYSTEM ---
        project=project_save_path,
        name="Training_Final_801010_Resep Trial BEST",
        save=True,
        plots=True,
        patience=30,          
        cache=False           # Menghindari I/O Bottleneck / RAM Penuh
    )

    print("\n🎉 FINAL TRAINING SELESAI!")
    print(f"📁 Output dan grafik performa (mAP) tersimpan di: {project_save_path}")

# ==============================
# ENTRY POINT (WAJIB UNTUK WINDOWS)
# ==============================
if __name__ == "__main__":
    freeze_support()
    run_training()