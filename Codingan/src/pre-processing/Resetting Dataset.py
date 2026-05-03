import os
import glob
from tqdm import tqdm

# --- PATH ---
dataset_path = r"C:\Users\Ivan\OneDrive\Documents\Skripsi_Malaria\datasets\malaria"

def reset_to_pure():
    print("🧹 SEDANG MEMBERSIHKAN DATASET KEMBALI KE MURNI (PURE)...")
    
    # Kata kunci file sampah (hasil generate sebelumnya)
    # Sesuaikan dengan nama file yang dibuat script sebelumnya
    keywords = ["_copy", "_bal_", "_aug", "aug_"] 
    
    deleted_count = 0
    
    for split in ['train', 'val']:
        img_dir = os.path.join(dataset_path, "images", split)
        lbl_dir = os.path.join(dataset_path, "labels", split)
        
        if not os.path.exists(img_dir): continue
        
        files = os.listdir(img_dir)
        for f in tqdm(files, desc=f"Scanning {split}"):
            # Cek apakah nama file mengandung kata kunci sampah
            if any(k in f for k in keywords):
                # Hapus Gambar
                try:
                    os.remove(os.path.join(img_dir, f))
                    
                    # Hapus Label pasangannya
                    base_name = os.path.splitext(f)[0]
                    txt_path = os.path.join(lbl_dir, base_name + ".txt")
                    if os.path.exists(txt_path):
                        os.remove(txt_path)
                    
                    deleted_count += 1
                except Exception as e:
                    print(f"Gagal hapus {f}: {e}")

    print(f"\n✅ SELESAI! {deleted_count} file duplikat/sampah telah dihapus.")
    print("   Sekarang dataset Anda sudah bersih (PURE).")

if __name__ == "__main__":
    reset_to_pure()