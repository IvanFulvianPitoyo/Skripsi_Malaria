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
# TARGET
# ==============================
TARGET_MAP50 = 0.90
MAX_TOTAL_TRIALS = 100


# ==============================
# OBJECTIVE FUNCTION
# ==============================
def objective(trial):

    # ==========================
    # OPTUNA SEARCH SPACE (12)
    # ==========================
    lr0 = trial.suggest_float("lr0", 1e-5, 1e-2, log=True)
    lrf = trial.suggest_float("lrf", 0.001, 0.1, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    warmup_epochs = trial.suggest_float("warmup_epochs", 1.0, 8.0)

    momentum = trial.suggest_float("momentum", 0.8, 0.98)

    scale = trial.suggest_float("scale", 0.3, 0.7)
    degrees = trial.suggest_int("degrees", 0, 180)
    translate = trial.suggest_float("translate", 0.02, 0.25)

    mosaic = trial.suggest_float("mosaic", 0.3, 1.0)
    mixup = trial.suggest_float("mixup", 0.0, 0.4)

    hsv_s = trial.suggest_float("hsv_s", 0.4, 0.9)
    hsv_v = trial.suggest_float("hsv_v", 0.2, 0.7)

    print(f"\n🔎 Trial {trial.number} dimulai...")

    model = YOLO("yolov8m.pt")

    results = model.train(

        data=yaml_path,
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,
        workers=3,
        optimizer="AdamW",

        lr0=lr0,
        lrf=lrf,
        momentum=momentum,
        weight_decay=weight_decay,

        cos_lr=True,
        warmup_epochs=warmup_epochs,

        fliplr=0.5,
        flipud=0.5,

        degrees=degrees,
        translate=translate,
        scale=scale,

        mosaic=mosaic,
        mixup=mixup,

        hsv_s=hsv_s,
        hsv_v=hsv_v,

        project=project_save_path,
        name=f"optuna_trial_{trial.number}",

        cache=False,
        plots=False,
        verbose=False
    )

    # ==========================
    # METRICS
    # ==========================
    map50 = results.results_dict["metrics/mAP50(B)"]
    map50_95 = results.results_dict["metrics/mAP50-95(B)"]

    trial.set_user_attr("map50", map50)

    print(f"Trial {trial.number} mAP50: {map50}")
    print(f"Trial {trial.number} mAP50-95: {map50_95}")

    return map50_95


# ==============================
# MAIN OPTUNA LOOP
# ==============================
def run_optuna_training():

    print("🚀 MENJALANKAN OPTUNA SEARCH")

    yaml_content = {
        'path': dataset_path,
        'train': 'images/train',
        'val': 'images/val',
        'names': {0: 'P.falciparum', 1: 'P.vivax'}
    }

    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f)

    sampler = optuna.samplers.TPESampler(seed=42)

    study = optuna.create_study(

        study_name="malaria_yolo_study",
        direction="maximize",
        storage="sqlite:///malaria_yolo_study.db",
        load_if_exists=True,
        sampler=sampler
    )

    print(f"Total trial sebelumnya: {len(study.trials)}")

    # ==============================
    # LOOP SAMPAI mAP50 >= 90%
    # ==============================
    while True:

        study.optimize(objective, n_trials=5)

        best_trial = study.best_trial
        best_map50 = best_trial.user_attrs.get("map50", 0)

        print(f"\n🏆 Best mAP50 sekarang: {best_map50}")

        if best_map50 >= TARGET_MAP50:
            print("🎯 Target 90% tercapai!")
            break

        if len(study.trials) >= MAX_TOTAL_TRIALS:
            print("⚠️ Trial sudah mencapai batas maksimum")
            return

        print("Belum mencapai 90%, lanjut trial...\n")

    # ==============================
    # FINAL TRAINING
    # ==============================
    print("\n🔥 TRAINING FINAL MODEL")

    best_params = study.best_params
    model = YOLO("yolov8m.pt")

    model.train(

        data=yaml_path,
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,
        workers=6,
        optimizer="AdamW",

        lr0=best_params["lr0"],
        lrf=best_params["lrf"],
        momentum=best_params["momentum"],
        weight_decay=best_params["weight_decay"],

        cos_lr=True,
        warmup_epochs=best_params["warmup_epochs"],

        degrees=best_params["degrees"],
        translate=best_params["translate"],
        scale=best_params["scale"],

        mosaic=best_params["mosaic"],
        mixup=best_params["mixup"],

        hsv_s=best_params["hsv_s"],
        hsv_v=best_params["hsv_v"],

        fliplr=0.5,
        flipud=0.5,

        project=project_save_path,
        name="Training_Final_Optuna_Best",

        save=True,
        plots=True,
        patience=30
    )

    print("\n🎉 TRAINING SELESAI")
    print(f"Output: {project_save_path}")


# ==============================
# ENTRY POINT
# ==============================
if __name__ == "__main__":
    freeze_support()
    run_optuna_training()