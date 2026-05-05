import os
import yaml
from ultralytics import YOLO
from multiprocessing import freeze_support

# ==============================
# PATH SETTING
# ==============================
dataset_path = r"C:\Users\Ivan\OneDrive\Documents\Skripsi_Malaria\datasets\malaria"
project_save_path = r"D:\Skripsi_Malaria_Output"
yaml_path = os.path.join(dataset_path, "malaria.yaml")

# ==============================
# MAIN TRAINING FUNCTION
# ==============================
def run_training():

    print("🚀 MENJALANKAN TRAINING ULANG (RESEP TRIAL 20 - mAP 88.95%)...")

    # Buat YAML dataset
    yaml_content = {
        'path': dataset_path,
        'train': 'images/train',
        'val': 'images/val',
        'names': {0: 'P.falciparum', 1: 'P.vivax'}
    }

    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)

    # Load model pretrained
    model = YOLO("yolov8m.pt")

    # Training dengan Hyperparameter Trial 20
    model.train(
        data=yaml_path,
        epochs=100,           # Dikembalikan ke 100 sesuai ARGS Trial 20
        imgsz=640,
        batch=16,
        device=0,
        workers=6,
        optimizer="AdamW",
        
        # --- RESEP TRIAL 20 ---
        lr0=0.0021464035477145866,
        lrf=0.0042683683244226795,
        momentum=0.937,
        weight_decay=3.5299987701933912e-06,
        warmup_epochs=6.6902538291572045,
        warmup_momentum=0.94220796847618,
        degrees=114.0,
        translate=0.1,
        scale=0.4694983607092159,
        hsv_s=0.7,
        hsv_v=0.4,
        mosaic=1.0,
        erasing=0.36910053000778975,
        
        # --- SETTINGAN WAJIB MEDIS & STABILITAS ---
        cos_lr=True,          # Sesuai konfigurasi Trial 20
        fliplr=0.5,           # Wajib untuk sel darah
        flipud=0.5,           # Wajib untuk sel darah
        
        # --- OUTPUT & SYSTEM ---
        project=project_save_path,
        name="Training_Final_Trial20_Recipe",
        save=True,
        plots=True,
        patience=100,         # Sesuai konfigurasi Trial 20
        cache=False           # Menghindari I/O Bottleneck / RAM Penuh
    )

    print("\n🎉 TRAINING SELESAI!")
    print(f"📁 Output tersimpan di: {project_save_path}")

# ==============================
# ENTRY POINT
# ==============================
if __name__ == "__main__":
    freeze_support()
    run_training()