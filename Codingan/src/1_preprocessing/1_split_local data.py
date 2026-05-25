import os
import shutil
import random
from tqdm import tqdm

# --- 1. KONFIGURASI PATH WINDOWS (LOCAL) ---
source_root = r"C:\Users\Ivan\OneDrive\Documents\Skripsi\Actual Dataset\Dataset Malaria Hanura"
dataset_dest = r"C:\Users\Ivan\OneDrive\Documents\Skripsi_Malaria\dataset_final_malaria"

def split_dataset_local_mix():
    if not os.path.exists(source_root):
        print(f"❌ Error: Folder sumber tidak ditemukan di:")
        print(f"   {source_root}")
        print("👉 Pastikan path sudah benar dan tidak ada typo.")
        return

    print("🕵️‍♂️ Sedang menscan file dari folder sumber...")

    # MENGINISIALISASI VARIABEL YANG HILANG
    valid_images = ('.jpg', '.jpeg', '.png', '.bmp')
    data_pairs = []
    
    # Kata kunci file augmentasi/copy yang harus dijauhkan (Revisi Poin 1.1)    
    keywords_aug = ["_copy_", "_aug", "_bal_"]

    # --- 2. MENGUMPULKAN DATA ---
    for root, dirs, files in os.walk(source_root):
        for f in files:
            # FILTER ANTI-LEAKAGE: Diletakkan di dalam loop untuk setiap file
            if any(k in f.lower() for k in keywords_aug):
                continue  # Lewati file ini, jangan dimasukkan ke pembagian
            
            if f.lower().endswith(valid_images):
                img_path = os.path.join(root, f)
                
                base_name = os.path.splitext(f)[0]
                txt_name = base_name + '.txt'
                txt_path = os.path.join(root, txt_name)
                
                if os.path.exists(txt_path):
                    data_pairs.append((img_path, txt_path))

    total_files = len(data_pairs)
    print(f"📊 Total Pasangan Gambar+Label bersih (tanpa augmentasi): {total_files}")
    
    if total_files == 0:
        print("❌ Tidak ada data ditemukan. Cek kembali folder Anda.")
        return

    print("🔀 Mengacak (Mixing) dataset...")
    random.seed(42) 
    random.shuffle(data_pairs)
    
    # --- 3. PEMBAGIAN 80:20 (Revisi Poin 1.4) ---
    split_idx = int(total_files * 0.8)
    
    train_set = data_pairs[:split_idx]
    val_set = data_pairs[split_idx:]
    
    print(f"✅ Training Set (80%) : {len(train_set)} file")
    print(f"✅ Validation Set (20%): {len(val_set)} file")

    # --- 4. PEMBUATAN FOLDER ---
    for split in ['train', 'val']:
        os.makedirs(os.path.join(dataset_dest, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(dataset_dest, 'labels', split), exist_ok=True)

    # --- 5. EKSEKUSI COPY ---
    def copy_files(data_list, split_type):
        print(f"🚀 Menyalin ke folder {split_type}...")
        for img_src, txt_src in tqdm(data_list):
            filename_img = os.path.basename(img_src)
            filename_txt = os.path.basename(txt_src)
            
            dst_img = os.path.join(dataset_dest, 'images', split_type, filename_img)
            dst_txt = os.path.join(dataset_dest, 'labels', split_type, filename_txt)
            
            shutil.copy2(img_src, dst_img)
            shutil.copy2(txt_src, dst_txt)

    copy_files(train_set, 'train')
    copy_files(val_set, 'val')
    
    print("\n🎉 SELESAI! Dataset sudah terbagi 80:20 dengan aman tanpa kebocoran data (Data Leakage)!")

if __name__ == "__main__":
    split_dataset_local_mix()