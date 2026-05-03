import os
import yaml
import optuna
from ultralytics import YOLO
from multiprocessing import freeze_support

# ==============================
# PATH SETTING
# ==============================
dataset_path = r"C:\Users\Ivan\OneDrive\Documents\Skripsi_Malaria\datasets\malaria"
project_save_path = r"D:\Skripsi_Malaria_Output"
yaml_path = os.path.join(dataset_path, "malaria.yaml")

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

    # ==================================================
    # THE HOLY TRINITY HYPERPARAMETERS (Yang Diacak)
    # ==================================================
    # Optuna akan mencari di sekitar rekor Anda sebelumnya
    lr0 = trial.suggest_float("lr0", 1e-4, 1e-2, log=True)
    momentum = trial.suggest_float("momentum", 0.85, 0.98)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)

    model = YOLO("yolov8m.pt")

    results = model.train(
        data=yaml_path,
        epochs=50,          # 50 Epoch per trial cukup untuk menilai potensi model
        imgsz=640,
        batch=8,            # Pengaman VRAM/RAM (Mencegah error PID)
        device=0,
        workers=2,          # Pengaman Dataloader (Mencegah error multiprocessing)
        optimizer="AdamW",
        
        # --- PARAMETER YANG DI-TUNING OPTUNA (HOLY TRINITY) ---
        lr0=lr0,
        momentum=momentum,
        weight_decay=weight_decay,
        
        # --- PARAMETER YANG DIKUNCI (DARI ARGS.YAML 88.95%) ---
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
        save=False,         # Dimatikan agar penyimpanan D: tidak penuh selama tuning
        plots=False,
        verbose=False
    )
    
    # Optuna akan mengevaluasi berdasarkan nilai mAP50
    return results.results_dict["metrics/mAP50(B)"]

# ==============================
# RUN OPTUNA STUDY
# ==============================
def run_optuna_training():
    
    # Buat YAML dataset
    yaml_content = {
        'path': dataset_path,
        'train': 'images/train',
        'val': 'images/val',
        'names': {0: 'P.falciparum', 1: 'P.vivax'}
    }
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)

    print("🚀 MEMULAI PENCARIAN TAHAP KEDUA (FINE-TUNING HOLY TRINITY)...")
    
    study = optuna.create_study(direction="maximize", study_name="Malaria_YOLO_Holy3")
    
    # Looping Trial Manual agar bisa berhenti jika target 90% tercapai
    while True:
        study.optimize(objective, n_trials=1)
        best_val = study.best_value
        print(f"\n🏆 Best mAP50 Saat Ini: {best_val:.4f} (Target: {TARGET_MAP50})")

        if best_val >= TARGET_MAP50:
            print("🎯 Target 90% Tercapai! Menghentikan pencarian...")
            break

        if len(study.trials) >= MAX_TOTAL_TRIALS:
            print("⚠️ Batas Trial tercapai. Lanjut ke Final Training...")
            break

    # ==============================
    # TRAINING FINAL (MODEL TERBAIK)
    # ==============================
    print("\n🔥 TRAINING FINAL DENGAN PARAMETER TERBAIK (100 EPOCH)...")

    best_params = study.best_params
    print(f"Resep Pemenang: {best_params}")

    model = YOLO("yolov8m.pt")

    model.train(
        data=yaml_path,
        epochs=100,         # Final ditraining penuh 100 Epoch
        imgsz=640,
        batch=8,
        device=0,
        workers=2,
        optimizer="AdamW",
        
        # --- HASIL TUNING DIMASUKKAN KE SINI ---
        lr0=best_params["lr0"],
        momentum=best_params["momentum"],
        weight_decay=best_params["weight_decay"],
        
        # --- PARAMETER YANG DIKUNCI (DARI ARGS.YAML 88.95%) ---
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
        name="Training_Final_Holy3_Best",
        save=True,          # Diaktifkan untuk menyimpan weight best.pt
        plots=True          # Diaktifkan untuk grafik Bab 4
    )

# ==============================
# ENTRY POINT
# ==============================
if __name__ == "__main__":
    freeze_support()
    run_optuna_training()