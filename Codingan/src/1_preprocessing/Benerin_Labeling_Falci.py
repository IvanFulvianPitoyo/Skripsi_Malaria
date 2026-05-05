import os
from tqdm import tqdm

# --- KONFIGURASI PATH ---
# Lokasi dataset label Anda (Train dan Val)
train_label_path = r"C:\Users\Ivan\OneDrive\Documents\Skripsi_Malaria\datasets\malaria\labels\train"
val_label_path   = r"C:\Users\Ivan\OneDrive\Documents\Skripsi_Malaria\datasets\malaria\labels\val"

def fix_labels(folder_path):
    print(f"🔧 Memperbaiki label di folder: {folder_path}")
    
    if not os.path.exists(folder_path):
        print("❌ Folder tidak ditemukan!")
        return

    files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    fixed_count = 0
    
    for filename in tqdm(files):
        path = os.path.join(folder_path, filename)
        
        with open(path, 'r') as f:
            lines = f.readlines()
            
        new_lines = []
        modified = False
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 0: continue
            
            class_id = int(parts[0])
            
            # --- LOGIKA PERBAIKAN ---
            # Jika ketemu angka 15 atau 2, ubah jadi 0 (Falciparum)
            if class_id == 15 or class_id == 2:
                parts[0] = '0'  # Ubah jadi 0
                new_line = " ".join(parts) + "\n"
                new_lines.append(new_line)
                modified = True
            
            # Jika angka > 1 (selain 15 dan 2), ubah paksa ke 0 juga (Jaga-jaga)
            elif class_id > 1:
                parts[0] = '0'
                new_line = " ".join(parts) + "\n"
                new_lines.append(new_line)
                modified = True
                
            else:
                # Jika sudah benar (0 atau 1), biarkan
                new_lines.append(line)
        
        # Simpan kembali jika ada perubahan
        if modified:
            with open(path, 'w') as f:
                f.writelines(new_lines)
            fixed_count += 1

    print(f"✅ Selesai! {fixed_count} file label berhasil diperbaiki di folder ini.\n")

if __name__ == "__main__":
    # Perbaiki folder Train
    fix_labels(train_label_path)
    # Perbaiki folder Val
    fix_labels(val_label_path)
    
    print("🎉 SEMUA LABEL SUDAH DIPERBAIKI!")
    print("👉 Langkah Selanjutnya:")
    print("1. Jalankan script '6_hapus_cache.py' (WAJIB! Agar YOLO baca ulang).")
    print("2. Jalankan Training Ulang.")