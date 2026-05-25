import os
import shutil
from tqdm import tqdm

# 1. Path tempat ke-23 folder "Sudah Anotasi" berada (Sesuai screenshot)
source_base_dir = r"C:\Users\Ivan Fulvian - Personal\Documents\Skripsi_Malaria\datasets\malaria"

# 2. Path folder MASTER baru untuk menampung semua file yang sudah disatukan
target_base_dir = r"C:\Users\Ivan Fulvian - Personal\Documents\Skripsi_Malaria\Data_Asli"
target_images_dir = os.path.join(target_base_dir, "images")
target_labels_dir = os.path.join(target_base_dir, "labels")

# Buat folder target jika belum ada
os.makedirs(target_images_dir, exist_ok=True)
os.makedirs(target_labels_dir, exist_ok=True)

print("🧹 MEMULAI PENGGABUNGAN FOLDER...")

image_count = 0
label_count = 0

# Telusuri setiap subfolder di dalam source_base_dir
for folder_name in tqdm(os.listdir(source_base_dir), desc="Memproses Folder"):
    folder_path = os.path.join(source_base_dir, folder_name)
    
    # Pastikan yang diproses hanya FOLDER (mengabaikan file nyasar)
    if os.path.isdir(folder_path):
        
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            
            # Jika itu file gambar (.jpg, .png, .jpeg)
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                shutil.copy2(file_path, os.path.join(target_images_dir, file))
                image_count += 1
                
            # Jika itu file teks label YOLO (Abaikan file classes.txt bawaan LabelImg)
            elif file.lower().endswith('.txt') and file.lower() != "classes.txt":
                shutil.copy2(file_path, os.path.join(target_labels_dir, file))
                label_count += 1

print("\n✅ PENGGABUNGAN SELESAI!")
print(f"Total Gambar terkumpul : {image_count} file")
print(f"Total Label terkumpul  : {label_count} file")
print(f"File tersimpan rapi di : {target_base_dir}")