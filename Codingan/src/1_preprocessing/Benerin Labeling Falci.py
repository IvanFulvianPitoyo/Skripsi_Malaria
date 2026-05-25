import os
from tqdm import tqdm

# --- KONFIGURASI PATH ---
# Lokasi dataset label Anda (Train, Val, dan Test)
train_label_path = r"C:\Users\Ivan\OneDrive\Documents\Skripsi_Malaria\dataset_final_malaria\labels\train"
val_label_path   = r"C:\Users\Ivan\OneDrive\Documents\Skripsi_Malaria\dataset_final_malaria\labels\val"
test_label_path  = r"C:\Users\Ivan\OneDrive\Documents\Skripsi_Malaria\dataset_final_malaria\labels\test"

def fix_labels(folder_path):
    print(f"🔧 Memperbaiki label di folder: {folder_path}")
    
    if not os.path.exists(folder_path):
        print(f"❌ Folder tidak ditemukan: {folder_path}")
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
            if len(parts) == 0: 
                continue
            
            try:
                class_id = int(parts[0])
            except ValueError:
                # Lewati jika baris kosong atau bukan angka
                new_lines.append(line)
                continue
            
            # --- LOGIKA PERBAIKAN ---
            # Jika angka ID kelas lebih dari 1 (misal 2, 15, dst), ubah paksa ke 0 (Falciparum)
            if class_id > 1:
                parts[0] = '0'
                new_line = " ".join(parts) + "\n"
                new_lines.append(new_line)
                modified = True
            else:
                # Jika sudah benar (0 atau 1), biarkan utuh
                new_lines.append(line)
        
        # Simpan kembali (overwrite) hanya jika ada perubahan pada file tersebut
        if modified:
            with open(path, 'w') as f:
                f.writelines(new_lines)
            fixed_count += 1

    print(f"✅ Selesai! {fixed_count} file label berhasil diperbaiki di folder ini.\n")


if __name__ == "__main__":
    # EKSEKUSI MUTLAK UNTUK KETIGA FOLDER
    fix_labels(train_label_path)
    fix_labels(val_label_path)
    fix_labels(test_label_path)
    
    print("🎉 SEMUA LABEL (TRAIN, VAL, TEST) SUDAH DIPERBAIKI DENGAN AMAN!")
    print("👉 Langkah Selanjutnya:")
    print("1. Cek folder labels Anda, hapus file 'labels.cache' (jika ada) agar YOLO tidak membaca cache lama.")
    print("2. Jalankan Final Training 80:10:10.")