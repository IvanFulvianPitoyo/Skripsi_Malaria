import os
import shutil
import random
from tqdm import tqdm

# ==============================
# KONFIGURASI FOLDER
# ==============================
# Ganti dengan folder tempat dataset 1:2 Anda berada sekarang
source_base = r"D:\DatasetMalaria1x1"
target_base = r"D:\Seharusnya1x1"

def trim_dataset_to_1x1():
    print("🚀 MEMULAI PEMANGKASAN DATASET DARI RASIO 1:2 MENJADI 1:1...")

    # 1. Hapus folder D lama agar benar-benar suci dari file hantu/cache
    if os.path.exists(target_base):
        print(f"🧹 Menghapus folder lama {target_base}...")
        shutil.rmtree(target_base)
    
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(target_base, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(target_base, 'labels', split), exist_ok=True)

    src_train_img = os.path.join(source_base, 'images', 'train')
    src_train_lbl = os.path.join(source_base, 'labels', 'train')
    tgt_train_img = os.path.join(target_base, 'images', 'train')
    tgt_train_lbl = os.path.join(target_base, 'labels', 'train')

    falci_files = [] 
    vivax_only_files = [] 

    total_falci_boxes = 0
    vivax_terbawa = 0

    print("🔍 Menganalisis folder Train 1:2 Anda...")
    label_files = [f for f in os.listdir(src_train_lbl) if f.endswith('.txt') and f != 'classes.txt']

    # 2. Klasifikasi gambar
    for f in tqdm(label_files):
        lbl_path = os.path.join(src_train_lbl, f)
        base_name = os.path.splitext(f)[0]
        
        img_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            temp_img = os.path.join(src_train_img, base_name + ext)
            if os.path.exists(temp_img):
                img_path = temp_img
                break
                
        if not img_path: continue

        c_falci, c_vivax = 0, 0
        with open(lbl_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if not parts: continue
                if parts[0] == '0': c_falci += 1
                elif parts[0] == '1': c_vivax += 1
                    
        if c_falci > 0:
            falci_files.append({'name': base_name, 'img_path': img_path, 'lbl_path': lbl_path})
            total_falci_boxes += c_falci
            vivax_terbawa += c_vivax
        elif c_vivax > 0:
            vivax_only_files.append({'name': base_name, 'img_path': img_path, 'lbl_path': lbl_path, 'box_count': c_vivax})

    print(f"\n✅ P. Falciparum (Patokan) : {total_falci_boxes} kotak")
    print(f"✅ P. Vivax (Terbawa otomatis) : {vivax_terbawa} kotak")

    # 3. Pemangkasan (Undersampling) Vivax
    # Kita lebihkan sedikit (misal targetnya 1:1.05) agar model tidak terlalu kelaparan data variasi
    target_vivax_total = int(total_falci_boxes * 1.05) 
    target_vivax_kurang = target_vivax_total - vivax_terbawa
    
    selected_vivax_files = []
    current_added = 0

    if target_vivax_kurang > 0:
        print(f"\n✂️ Mengambil sisa Vivax untuk mengejar target ~{target_vivax_total} kotak...")
        random.seed(42) # Dikunci agar bisa dipertanggungjawabkan di sidang
        random.shuffle(vivax_only_files)
        
        for v_data in vivax_only_files:
            if current_added >= target_vivax_kurang:
                break
            selected_vivax_files.append(v_data)
            current_added += v_data['box_count']

    final_vivax_total = vivax_terbawa + current_added

    print("\n==================================================")
    print(f"🎯 ESTIMASI HASIL AKHIR DATASET TRAIN:")
    print(f"   P. Falciparum : {total_falci_boxes} kotak")
    print(f"   P. Vivax      : {final_vivax_total} kotak")
    print("==================================================\n")

    # 4. Salin data Train
    final_files_to_copy = falci_files + selected_vivax_files
    print("⏳ Menyalin data Train terpilih ke D:\\DatasetMalaria1x1...")
    for data in tqdm(final_files_to_copy):
        shutil.copy2(data['lbl_path'], os.path.join(tgt_train_lbl, os.path.basename(data['lbl_path'])))
        shutil.copy2(data['img_path'], os.path.join(tgt_train_img, os.path.basename(data['img_path'])))

    # 5. Salin data Val dan Test (Murni) SECARA KETAT (Abaikan file .cache)
    print("\n⏳ Menyalin folder Validation dan Test (Anti-Cache)...")
    for split in ['val', 'test']:
        for folder in ['images', 'labels']:
            src_dir = os.path.join(source_base, folder, split)
            tgt_dir = os.path.join(target_base, folder, split)
            if os.path.exists(src_dir):
                for f in os.listdir(src_dir):
                    # KUNCI UTAMA: Jangan pernah menyalin file labels.cache!
                    if not f.endswith('.cache'):
                        shutil.copy2(os.path.join(src_dir, f), os.path.join(tgt_dir, f))

    yaml_path = os.path.join(target_base, 'malaria_1x1.yaml')
    with open(yaml_path, 'w') as f:
        f.write(f"path: {target_base}\ntrain: images/train\nval: images/val\ntest: images/test\nnc: 2\nnames:\n  0: P. falciparum\n  1: P. vivax\n")

    print("\n🎉 SELESAI! Folder dataset Anda sudah rapi dan DIJAMIN BEBAS CACHE.")

if __name__ == "__main__":
    trim_dataset_to_1x1()