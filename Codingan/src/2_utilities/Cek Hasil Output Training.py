import os
import pandas as pd

root = r"D:\Skripsi_Malaria_Output"

hasil = []

for folder in os.listdir(root):
    if folder.startswith("optuna_holy3_trial_"):

        csv_path = os.path.join(root, folder, "results.csv")

        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)

                # Cari kolom mAP50
                map_cols = [c for c in df.columns if "mAP50" in c]

                if map_cols:
                    best_map50 = df[map_cols[0]].max()

                    hasil.append({
                        "trial": folder,
                        "mAP50": best_map50
                    })

            except Exception as e:
                print(f"Gagal membaca {folder}: {e}")

hasil = sorted(hasil, key=lambda x: x["mAP50"], reverse=True)

print("\n===== TOP 10 =====")
for i, h in enumerate(hasil[:10]):
    print(f"{i+1}. {h['trial']} -> {h['mAP50']:.4f}")

if hasil:
    print("\n🏆 TERBAIK:")
    print(hasil[0])