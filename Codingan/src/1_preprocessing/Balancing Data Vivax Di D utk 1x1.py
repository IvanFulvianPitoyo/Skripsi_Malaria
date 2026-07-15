import os
import shutil
import random
from tqdm import tqdm

# ==============================
# KONFIGURASI FOLDER SUMBER & TUJUAN
# ==============================
# Folder sumber: Tempat data yang labelnya sudah benar dan Falciparum-nya sudah diaugmentasi
source_base = r"C:\Users\Ivan\OneDrive\Documents\Skripsi_Malaria\dataset_final_malaria"

# Folder tujuan: Partisi D Anda
target_base = r"D:\DatasetMalaria1x1"

def create_dir(path):
    os.makedirs(path, exist_ok=True)

def balance_dataset():
    print("🚀 MEMULAI PEMBUATAN DATASET 1:1 DI PARTISI D...")

    # 1. Buat struktur folder di D:\
    for split in ['train', 'val', 'test']:
        create_dir(os.path.join(target_base, 'images', split))
        create_dir(os.path.join(target_base, 'labels', split))

    src_train_img = os.path.join(source_base, 'images', 'train')
    src_train_lbl = os.path.join(source_base, 'labels', 'train')
    tgt_train_img = os.path.join(target_base, 'images', 'train')
    tgt_train_lbl = os.path.join(target_base, 'labels', 'train')

    falci_files = [] 
    vivax_only_files = [] 

    total_falci_boxes = 0
    vivax_terbawa = 0

    print("🔍 Menghitung total kotak (bounding box) di folder Train sumber...")
    
    if not os.path.exists(src_train_lbl):
        print(f"❌ Folder sumber tidak ditemukan: {src_train_lbl}")
        return

    label_files = [f for f in os.listdir(src_train_lbl) if f.endswith('.txt') and f != 'classes.txt']

    # 2. Proses pemilahan data (Mana yang wajib dibawa, mana yang jadi cadangan)
    for f in tqdm(label_files):
        lbl_path = os.path.join(src_train_lbl, f)
        base_name = os.path.splitext(f)[0]
        
        # Deteksi gambar pasangannya (mendukung berbagai format)
        img_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG']:
            temp_img = os.path.join(src_train_img, base_name + ext)
            if os.path.exists(temp_img):
                img_path = temp_img
                break
                
        if not img_path: 
            continue

        c_falci = 0
        c_vivax = 0
        
        with open(lbl_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if not parts: continue
                if parts[0] == '0': c_falci += 1
                elif parts[0] == '1': c_vivax += 1
                    
        # KATEGORI A: Wajib dibawa karena ada Falciparum
        if c_falci > 0:
            falci_files.append({'name': base_name, 'img_path': img_path, 'lbl_path': lbl_path})
            total_falci_boxes += c_falci
            vivax_terbawa += c_vivax
            
        # KATEGORI B: Murni Vivax (Kandidat yang akan dipotong/undersampling)
        elif c_vivax > 0:
            vivax_only_files.append({'name': base_name, 'img_path': img_path, 'lbl_path': lbl_path, 'box_count': c_vivax})

    print(f"\n✅ Total P. Falciparum saat ini : {total_falci_boxes} kotak")
    print(f"✅ P. Vivax yang ikut terbawa   : {vivax_terbawa} kotak")

    # 3. Proses Penyeimbangan (Undersampling)
    target_vivax_kurang = total_falci_boxes - vivax_terbawa
    selected_vivax_files = []
    current_added = 0

    if target_vivax_kurang > 0:
        print(f"\n✂️ Menyedot {target_vivax_kurang} kotak Vivax tambahan secara acak untuk menyeimbangkan rasio...")
        random.seed(42) # Kunci acak agar hasil tetap konsisten jika diulang
        random.shuffle(vivax_only_files)
        
        for v_data in vivax_only_files:
            if current_added >= target_vivax_kurang:
                break
            selected_vivax_files.append(v_data)
            current_added += v_data['box_count']

    final_vivax_total = vivax_terbawa + current_added

    print("\n==================================================")
    print(f"🎯 ESTIMASI HASIL AKHIR DATASET TRAIN 1:1:")
    print(f"   P. Falciparum : {total_falci_boxes} kotak")
    print(f"   P. Vivax      : {final_vivax_total} kotak")
    print("==================================================\n")

    # Gabungkan file
    final_files_to_copy = falci_files + selected_vivax_files

    # 4. Eksekusi Salin (Copy) File ke Disk D
    print("⏳ Menyalin file Training ke D:\\DatasetMalaria1x1...")
    for data in tqdm(final_files_to_copy):
        # Gunakan copy2 untuk mempertahankan metadata asli file
        shutil.copy2(data['lbl_path'], os.path.join(tgt_train_lbl, os.path.basename(data['lbl_path'])))
        shutil.copy2(data['img_path'], os.path.join(tgt_train_img, os.path.basename(data['img_path'])))

    # 5. Salin Folder Validasi dan Test (Tanpa diubah/murni)
    print("\n⏳ Menyalin folder Validation dan Test...")
    for split in ['val', 'test']:
        for folder in ['images', 'labels']:
            src_dir = os.path.join(source_base, folder, split)
            tgt_dir = os.path.join(target_base, folder, split)
            if os.path.exists(src_dir):
                for f in os.listdir(src_dir):
                    shutil.copy2(os.path.join(src_dir, f), os.path.join(tgt_dir, f))

    # 6. Buat file .yaml otomatis
    yaml_path = os.path.join(target_base, 'malaria_1x1.yaml')
    yaml_content = f"""path: {target_base}
train: images/train
val: images/val
test: images/test

nc: 2
names:
  0: P. falciparum
  1: P. vivax
"""
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print("\n🎉 SELESAI! Folder dataset Anda sudah rapi di D:\\DatasetMalaria1x1.")
    print("📝 File malaria_1x1.yaml juga sudah dibuatkan otomatis di folder tersebut.")
    print("\n👉 Langkah Selanjutnya: Anda tinggal menjalankan training YOLO dengan data tersebut.")

if __name__ == "__main__":
    balance_dataset()