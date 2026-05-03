import os
import shutil
import random
from tqdm import tqdm

# --- KONFIGURASI PATH ---
base_path = r"C:\Users\Ivan\OneDrive\Documents\Skripsi_Malaria\datasets\malaria"
images_dir = os.path.join(base_path, "images")
labels_dir = os.path.join(base_path, "labels")

def resplit_and_balance():
    print("🔄 MEMULAI OPERASI 'KOCOK ULANG' & BALANCING...")
    
    # 1. KUMPULKAN SEMUA DATA MURNI (Hapus Copy-an Lama)
    print("\n🧹 Tahap 1: Mengumpulkan & Membersihkan Data...")
    all_pairs = [] # List berisi (path_gambar, path_label)
    
    for split in ['train', 'val']:
        img_split_path = os.path.join(images_dir, split)
        lbl_split_path = os.path.join(labels_dir, split)
        
        if not os.path.exists(img_split_path): continue
        
        files = os.listdir(img_split_path)
        for f in files:
           
            if "_copy_" in f:
                os.remove(os.path.join(img_split_path, f))
                # Hapus labelnya juga
                lbl_name = os.path.splitext(f)[0] + ".txt"
                if os.path.exists(os.path.join(lbl_split_path, lbl_name)):
                    os.remove(os.path.join(lbl_split_path, lbl_name))
                continue
            
            # Jika file MURNI (Original), simpan ke list
            if f.endswith(('.jpg', '.png', '.bmp', '.jpeg', '.JPG')):
                base_name = os.path.splitext(f)[0]
                
                # Cek label (bisa .txt atau .TXT)
                lbl_name = base_name + ".txt"
                lbl_full = os.path.join(lbl_split_path, lbl_name)
                if not os.path.exists(lbl_full):
                    lbl_name = base_name + ".TXT"
                    lbl_full = os.path.join(lbl_split_path, lbl_name)
                
                if os.path.exists(lbl_full):
                    # Simpan path lengkapnya
                    all_pairs.append({
                        'img': os.path.join(img_split_path, f),
                        'lbl': lbl_full,
                        'filename': f,
                        'lbl_filename': lbl_name
                    })

    print(f"   Total Data Murni ditemukan: {len(all_pairs)} gambar")
    
    # 2. DETEKSI KELAS (FALCIPARUM vs VIVAX)
    falci_pairs = []
    vivax_pairs = []
    
    print("   Sedang memisahkan Falciparum dan Vivax...")
    for item in tqdm(all_pairs):
        with open(item['lbl'], 'r') as f:
            content = f.read()
            # Asumsi Falciparum = 0, Vivax = 1
            if '0 ' in content: 
                falci_pairs.append(item)
            else:
                vivax_pairs.append(item)

    print(f"   📊 Statistik Murni: Falciparum={len(falci_pairs)} | Vivax={len(vivax_pairs)}")
    
    # 3. BAGI ULANG (SPLIT 80:20)
    print("\n🎲 Tahap 2: Membagi Ulang (Split 80/20)...")
    random.shuffle(falci_pairs)
    random.shuffle(vivax_pairs)
    
    # Hitung porsi
    n_falci_train = int(len(falci_pairs) * 0.8)
    n_vivax_train = int(len(vivax_pairs) * 0.8)
    
    train_data = falci_pairs[:n_falci_train] + vivax_pairs[:n_vivax_train]
    val_data   = falci_pairs[n_falci_train:] + vivax_pairs[n_vivax_train:]
    
    print(f"   Target Train: {len(train_data)} | Target Val: {len(val_data)}")

    # 4. PINDAHKAN FILE KE TEMPAT BARU
    print("   Memindahkan file...")
    
    # Fungsi pembantu pindah
    def move_files(data_list, split_name):
        for item in data_list:
            # Tujuan Gambar
            dst_img = os.path.join(images_dir, split_name, item['filename'])
            # Tujuan Label
            dst_lbl = os.path.join(labels_dir, split_name, item['lbl_filename'])
            
            # Pindahkan (Move) - Hati-hati jangan tertukar source/dest
            if item['img'] != dst_img:
                shutil.move(item['img'], dst_img)
            if item['lbl'] != dst_lbl:
                shutil.move(item['lbl'], dst_lbl)

    move_files(train_data, 'train')
    move_files(val_data, 'val')
    
    print("✅ Re-split selesai!")

    # 5. BALANCING (OVERSAMPLING) HANYA DI TRAIN
    print("\n⚖️ Tahap 3: Balancing Training Set...")
    
    # Hitung ulang kondisi Train sekarang
    train_vivax_count = n_vivax_train
    train_falci_count = n_falci_train
    
    needed = train_vivax_count - train_falci_count
    
    if needed > 0:
        print(f"   Menambahkan {needed} duplikat Falciparum ke Training...")
        
        # Ambil data Falciparum yang ada di folder train
        current_falci_files = [x['filename'] for x in falci_pairs[:n_falci_train]]
        
        for i in tqdm(range(needed)):
            src_fname = random.choice(current_falci_files)
            base_name = os.path.splitext(src_fname)[0]
            ext = os.path.splitext(src_fname)[1]
            
            src_img_path = os.path.join(images_dir, 'train', src_fname)
            src_lbl_path = os.path.join(labels_dir, 'train', base_name + ".txt")
            if not os.path.exists(src_lbl_path): src_lbl_path = os.path.join(labels_dir, 'train', base_name + ".TXT")
            
            # Nama Baru
            new_base = f"{base_name}_bal_{i}"
            dst_img_path = os.path.join(images_dir, 'train', new_base + ext)
            dst_lbl_path = os.path.join(labels_dir, 'train', new_base + ".txt")
            
            shutil.copy(src_img_path, dst_img_path)
            shutil.copy(src_lbl_path, dst_lbl_path)
            
    else:
        print("   Data Training sudah cukup seimbang.")

    print("\n🎉 SELESAI TOTAL! Struktur folder sudah diperbaiki.")
    print("   Sekarang folder 'val' punya cukup banyak soal Falciparum.")

if __name__ == "__main__":
    resplit_and_balance()