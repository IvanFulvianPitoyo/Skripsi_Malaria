import os
from tqdm import tqdm

# ==============================
# KONFIGURASI FOLDER
# ==============================
# Masukkan path folder 'labels' yang ingin Anda cek (biasanya folder 'train')
folder_labels = r"D:\DatasetMalaria1x1\labels\train"

def check_dataset_balance():
    print(f"🔍 Memindai folder: {folder_labels}")
    
    if not os.path.exists(folder_labels):
        print("❌ Folder tidak ditemukan! Pastikan path sudah benar.")
        return

    # Ambil semua file .txt kecuali classes.txt
    txt_files = [f for f in os.listdir(folder_labels) if f.endswith('.txt') and f != 'classes.txt']
    
    if not txt_files:
        print("⚠️ Tidak ada file .txt yang ditemukan di folder ini.")
        return

    total_falci = 0
    total_vivax = 0
    anomalies = 0

    print("⏳ Menghitung jumlah kotak target (bounding box)...")
    for file in tqdm(txt_files):
        with open(os.path.join(folder_labels, file), 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                
                class_id = parts[0]
                if class_id == '0':
                    total_falci += 1
                elif class_id == '1':
                    total_vivax += 1
                else:
                    anomalies += 1 # Menangkap jika ada class_id 2, 3, dst.

    print("\n==================================================")
    print("📊 HASIL PENGECEKAN KESEIMBANGAN DATASET (KOTAK/BBOX)")
    print("==================================================")
    print(f"Total File Gambar/Label : {len(txt_files)} file")
    print(f"Total P. Falciparum (0) : {total_falci} kotak")
    print(f"Total P. Vivax (1)      : {total_vivax} kotak")
    print("--------------------------------------------------")
    
    if total_falci > 0 and total_vivax > 0:
        if total_falci > total_vivax:
            ratio = total_falci / total_vivax
            print(f"⚖️ Rasio Falci vs Vivax  : {ratio:.2f} : 1")
        else:
            ratio = total_vivax / total_falci
            print(f"⚖️ Rasio Falci vs Vivax  : 1 : {ratio:.2f}")
    elif total_falci == 0:
        print("⚖️ Rasio                : HANYA ADA VIVAX!")
    elif total_vivax == 0:
        print("⚖️ Rasio                : HANYA ADA FALCIPARUM!")
        
    if anomalies > 0:
        print(f"⚠️ PERINGATAN: Ditemukan {anomalies} kotak dengan ID kelas SALAH (bukan 0 atau 1)!")
        
    print("==================================================")

if __name__ == "__main__":
    check_dataset_balance()
