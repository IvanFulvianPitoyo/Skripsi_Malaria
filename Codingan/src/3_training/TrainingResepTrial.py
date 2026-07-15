import os
import yaml
from ultralytics import YOLO
from multiprocessing import freeze_support

# ==============================
# PATH SETTING
# ==============================
dataset_path = r"C:\Users\Ivan\OneDrive\Documents\Skripsi_Malaria\dataset_final_malaria"
project_save_path = r"D:\Skripsi_Malaria_Output"
yaml_path = os.path.join(dataset_path, "malaria_final.yaml")

def run_training():
    print("🚀 MENJALANKAN FINAL TRAINING (RESEP TERBAIK OPTUNA TRIAL 14 + DATA AUGMENTASI)...")

    # Buat YAML dataset
    yaml_content = {
        'path': dataset_path,
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test', 
        'names': {0: 'P. falciparum', 1: 'P. vivax'}
    }

    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)

    model = YOLO("yolov8m.pt")

    # Training dengan Hyperparameter Optuna Trial 14
    model.train(
        data=yaml_path,
        epochs=50,           
        imgsz=640,
        batch=8,              # Sesuai argumen Optuna & aman untuk memori
        device=0,             
        workers=4,            # WAJIB 2 agar RAM fisik aman
        optimizer="AdamW",
        
        # --- 🏆 RESEP PEMENANG OPTUNA (TRIAL 14) ---
        lr0=0.0003,
        lrf=0.00426 ,
        momentum=0.8652,
        weight_decay=8.35e-05,
        warmup_epochs=6.6902538291572045,
        warmup_momentum=0.94220796847618,
        warmup_bias_lr=0.1,
        
        # --- SPASIAL & AUGMENTASI (DARI OPTUNA) ---
        degrees=114.0,
        translate=0.1,
        scale=0.4694983607092159,
        hsv_s=0.7,
        hsv_v=0.4,
        mosaic=1.0,
        mixup=0.0,                  # Optuna menonaktifkan mixup
        erasing=0.36910053000778975, # 
        
        # --- SETTINGAN WAJIB MEDIS ---
        cos_lr=True,          
        fliplr=0.5,           
        flipud=0.5,           
        
        # --- OUTPUT & SYSTEM ---
        project=project_save_path,
        name="Training_Coba-Coba_Hyperparameter_lr0ish",
        save=True,
        plots=True,
        cache=False           
    )

    print("\n🎉 FINAL TRAINING SELESAI!")

if __name__ == "__main__":
    freeze_support()
    run_training()