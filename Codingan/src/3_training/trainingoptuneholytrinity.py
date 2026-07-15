import os
import yaml
import optuna
from ultralytics import YOLO
from multiprocessing import freeze_support

# ==============================
# PATH SETTING (ABSOLUT PADA SKRIPSI_MALARIA)
# ==============================
# Path langsung diarahkan ke folder Skripsi_Malaria Anda
project_root_dir = r"C:\Users\Ivan\OneDrive\Documents\Skripsi_Malaria"
dataset_path = os.path.join(project_root_dir, "datasets", "malaria")
project_save_path = r"D:\Skripsi_Malaria_Output"

# File YAML akan otomatis dibuat di dalam folder dataset
yaml_path = os.path.join(dataset_path, "malaria.yaml")

# Path menuju model mentah yolov8m.pt di dalam folder 'models'
model_path = os.path.join(project_root_dir, "models", "yolov8m.pt")

# ==============================
# TARGET SETTING
# ==============================
TARGET_MAP50 = 0.90
MAX_TOTAL_TRIALS = 30  # Pencarian tidak butuh banyak trial karena hanya 3 parameter

# ==============================
# OBJECTIVE FUNCTION (TUNING HOLY TRINITY)
# ==============================
def objective(trial):
    print(f"\n🔎 MEMULAI TRIAL {trial.number} (Fokus 3 Parameter Utama)...")

    # The Holy Trinity Hyperparameters
    lr0 = trial.suggest_float("lr0", 1e-4, 1e-2, log=True)
    momentum = trial.suggest_float("momentum", 0.85, 0.98)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)

    # Menggunakan model mentah YOLOv8m yang sudah diarahkan path-nya
    model = YOLO(model_path)

    results = model.train(
        data=yaml_path,
        epochs=50,          
        imgsz=640,
        batch=8,            
        device=0,
        workers=2,          
        optimizer="AdamW",
        
        # --- PARAMETER YANG DI-TUNING OPTUNA ---
        lr0=lr0,
        momentum=momentum,
        weight_decay=weight_decay,
        
        # --- PARAMETER YANG DIKUNCI (DARI REKAM JEJAK TERBAIK) ---
        lrf=0.0042683683244226795,
        warmup_epochs=6.6902538291572045,
        warmup_momentum=0.94220796847618,
        degrees=114.0,
        translate=0.1,
        scale=0.4694983607092159,
        erasing=0.36910053000778975,
        mosaic=1.0,
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,
        flipud=0.5,
        cos_lr=True,
        
        # --- OUTPUT SYSTEM ---
        project=project_save_path,
        name=f"optuna_holy3_trial_{trial.number}",
        save=False,         
        plots=False,
        verbose=False
    )
    
    return results.results_dict["metrics/mAP50(B)"]

# ==============================
# RUN OPTUNA STUDY
# ==============================
def run_optuna_training():
    
    # 1. Buat/Update file YAML
    yaml_content = {
        'path': dataset_path,
        'train': 'images/train',
        'val': 'images/val',
        'names': {0: 'P.falciparum', 1: 'P.vivax'}
    }
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)

    print("🚀 MEMULAI PENCARIAN TAHAP KEDUA (FINE-TUNING HOLY TRINITY)...")
    print(f"📂 Lokasi Dataset : {dataset_path}")
    print(f"📂 Lokasi Model   : {model_path}")
    
    # 2. Mulai Optuna
    study = optuna.create_study(direction="maximize", study_name="Malaria_YOLO_Holy3")
    
    while True:
        study.optimize(objective, n_trials=1)
        best_val = study.best_value
        print(f"\n🏆 Best mAP50 Saat Ini: {best_val:.4f} (Target: {TARGET_MAP50})")

        if best_val >= TARGET_MAP50:
            print("🎯 Target 90% Tercapai! Menghentikan pencarian parameter...")
            break

        if len(study.trials) >= MAX_TOTAL_TRIALS:
            print("⚠️ Batas Trial tercapai. Lanjut ke Final Training...")
            break

    # ==============================
    # TRAINING FINAL (MODEL TERBAIK)
    # ==============================
    print("\n🔥 TRAINING FINAL DENGAN PARAMETER TERBAIK (100 EPOCH)...")

    best_params = study.best_params
    print(f"Resep Pemenang Optuna: {best_params}")

    model = YOLO(model_path)

    model.train(
        data=yaml_path,
        epochs=100,         
        imgsz=640,
        batch=8,
        device=0,
        workers=2,
        optimizer="AdamW",
        
        # --- HASIL TUNING OPTUNA ---
        lr0=best_params["lr0"],
        momentum=best_params["momentum"],
        weight_decay=best_params["weight_decay"],
        
        # --- PARAMETER YANG DIKUNCI ---
        lrf=0.0042683683244226795,
        warmup_epochs=6.6902538291572045,
        warmup_momentum=0.94220796847618,
        degrees=114.0,
        translate=0.1,
        scale=0.4694983607092159,
        erasing=0.36910053000778975,
        mosaic=1.0,
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,
        flipud=0.5,
        cos_lr=True,
        
        # --- OUTPUT SYSTEM ---
        project=project_save_path,
        name="Training_Holy3",
        save=True,          
        plots=True          
    )

if __name__ == "__main__":
    freeze_support()
    run_optuna_training()