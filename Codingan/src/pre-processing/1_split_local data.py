import os
import shutil
import random
from tqdm import tqdm

# --- 1. KONFIGURASI PATH WINDOWS (LOCAL) ---
# Folder SUMBER (Data Asli 23 Folder)
# Menggunakan r"" (raw string) agar backslash '\' terbaca benar di Windows
source_root = r"C:\Users\Ivan\OneDrive\Documents\Skripsi\Actual Dataset\Dataset Malaria Hanura"

# Folder TUJUAN (Format YOLO)
# Sesuai screenshot Anda: Documents > Skripsi_Malaria > datasets > malaria
dataset_dest = r"C:\Users\Ivan\OneDrive\Documents\Skripsi_Malaria\datasets\malaria"

def split_dataset_local_mix():
    # Cek apakah folder sumber terbaca
    if not os.path.exists(source_root):
        print(f"❌ Error: Folder sumber tidak ditemukan di:")
        print(f"   {source_root}")
        print("👉 Pastikan path sudah benar dan tidak ada typo.")
        return

    print("🕵️‍♂️ Sedang menscan file dari folder OneDrive...")
    print("   (Ini mungkin butuh waktu sedikit jika OneDrive sedang sync)")

    # --- 2. MENGUMPULKAN DATA ---
    data_pairs = []
    valid_images = ('.jpg', '.jpeg', '.png', '.bmp')

    # os.walk akan menelusuri semua sub-folder (Sudah Anotasi 1 s/d 23)
    for root, dirs, files in os.walk(source_root):
        for f in files:
            if f.lower().endswith(valid_images):
                img_path = os.path.join(root, f)
                
                # Cari pasangan label .txt
                # Asumsi nama file sama, cuma beda ekstensi
                base_name = os.path.splitext(f)[0]
                txt_name = base_name + '.txt'
                txt_path = os.path.join(root, txt_name)
                
                # Hanya ambil jika pasangan labelnya ADA
                if os.path.exists(txt_path):
                    data_pairs.append((img_path, txt_path))

    total_files = len(data_pairs)
    print(f"📊 Total Pasangan Gambar+Label ditemukan: {total_files}")
    
    if total_files == 0:
        print("❌ Tidak ada data ditemukan. Cek apakah folder OneDrive Anda sudah ter-download di laptop?")
        return

    # --- 3. ACAK & BAGI (70:30) ---
    print("🔀 Mengacak (Mixing) dataset...")
    random.seed(42) # Seed 42 agar hasil acakan konsisten jika di-run ulang
    random.shuffle(data_pairs)
    
    # Hitung titik potong 70%
    split_idx = int(total_files * 0.7)
    
    train_set = data_pairs[:split_idx]
    val_set = data_pairs[split_idx:]
    
    print(f"✅ Training Set (70%) : {len(train_set)} file")
    print(f"✅ Validation Set (30%): {len(val_set)} file")

    for split in ['train', 'val']:
        os.makedirs(os.path.join(dataset_dest, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(dataset_dest, 'labels', split), exist_ok=True)

    # --- 5. EKSEKUSI COPY ---
    def copy_files(data_list, split_type):
        print(f"🚀 Menyalin ke folder {split_type}...")
        for img_src, txt_src in tqdm(data_list):
            # Nama file asli
            filename_img = os.path.basename(img_src)
            filename_txt = os.path.basename(txt_src)
            
            # Path Tujuan
            dst_img = os.path.join(dataset_dest, 'images', split_type, filename_img)
            dst_txt = os.path.join(dataset_dest, 'labels', split_type, filename_txt)
            
            # Copy file (Overwrite jika sudah ada, biar update)
            shutil.copy2(img_src, dst_img)
            shutil.copy2(txt_src, dst_txt)

    copy_files(train_set, 'train')
    copy_files(val_set, 'val')
    
    print("\n🎉 SELESAI! Dataset sudah terbagi di folder:")
    print(f"   {dataset_dest}")

if __name__ == "__main__":
    split_dataset_local_mix()