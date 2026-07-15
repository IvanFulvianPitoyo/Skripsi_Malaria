import os
import shutil
import random
from tqdm import tqdm

# --- KONFIGURASI PATH (Drive C - Dataset RGB Anda) ---
dataset_train_path = r"D:\DatasetMalaria1x1\images\train"
label_train_path   = r"D:\DatasetMalaria1x1\labels\train"

def balance_dataset_v2():
    print("🔍 DIAGNOSA PATH...")
    print(f"   Target Folder: {dataset_train_path}")
    
    # 1. Cek Apakah Folder Ada?
    if not os.path.exists(dataset_train_path):
        print("\n❌ ERROR FATAL: Folder tidak ditemukan!")
        print("   Mohon cek apakah path-nya sudah benar?")
        return

    # 2. Ambil semua file tanpa filter dulu (untuk debugging)
    all_files = os.listdir(dataset_train_path)
    print(f"   Total file di folder: {len(all_files)}")
    
    if len(all_files) > 0:
        print(f"   Contoh file ditemukan: {all_files[:3]}")
    else:
        print("❌ ERROR: Folder KOSONG melompong. Salah folder?")
        return

    # 3. Filter Gambar (Case Insensitive - Huruf Besar/Kecil dianggap sama)
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif')
    image_files = [f for f in all_files if f.lower().endswith(valid_exts)]
    
    print(f"   Jumlah Gambar Valid: {len(image_files)}")
    
    if len(image_files) == 0:
        print("❌ ERROR: Tidak ada gambar dengan akhiran .jpg/.png/.jpeg!")
        return

    print("\n⚖️ MULAI PROSES PENYEIMBANGAN DATA...")
    
    vivax_count = 0
    falciparum_count = 0
    falciparum_files = [] 

    # 4. Hitung Populasi
    print("   Sedang mengecek label...")
    for f in tqdm(image_files):
        # Cari file labelnya (juga handle huruf besar/kecil)
        base_name = os.path.splitext(f)[0]
        label_name = base_name + ".txt"
        label_file = os.path.join(label_train_path, label_name)
        
        if os.path.exists(label_file):
            with open(label_file, 'r') as txt:
                content = txt.read()
                if '0 ' in content: 
                    falciparum_count += 1
                    falciparum_files.append(f)
                elif '1 ' in content: 
                    vivax_count += 1
        else:
            # Coba cari versi .TXT huruf besar jika .txt gagal
            label_file_upper = os.path.join(label_train_path, base_name + ".TXT")
            if os.path.exists(label_file_upper):
                with open(label_file_upper, 'r') as txt:
                    content = txt.read()
                    if '0 ' in content:
                        falciparum_count += 1
                        falciparum_files.append(f)
                    elif '1 ' in content:
                        vivax_count += 1

    print(f"\n📊 STATISTIK DATA:")
    print(f"   - P.vivax      : {vivax_count}")
    print(f"   - P.falciparum : {falciparum_count}")
    
    if falciparum_count == 0:
        print("\n❌ MASALAH: Gambar ada, tapi Label '0' (Falciparum) tidak terdeteksi.")
        print("   Kemungkinan label txt Anda isinya kosong atau formatnya beda.")
        return

    # 5. Eksekusi Duplikasi
    target_count = vivax_count 
    needed = target_count - falciparum_count
    
    if needed <= 0:
        print("✅ Data sudah seimbang. Tidak perlu duplikasi.")
        return

    print(f"\n🚀 MENAMBAHKAN {needed} DATA FALCIPARUM...")
    
    for i in tqdm(range(needed)):
        src_img_name = random.choice(falciparum_files)
        base_name = os.path.splitext(src_img_name)[0]
        ext = os.path.splitext(src_img_name)[1] # Ambil ekstensi aslinya (.jpg/.JPG)
        
        # Nama Baru
        new_base_name = base_name + f"_copy_{i}"
        new_img_name = new_base_name + ext
        new_txt_name = new_base_name + ".txt"
        
        # Copy
        shutil.copy(os.path.join(dataset_train_path, src_img_name), 
                    os.path.join(dataset_train_path, new_img_name))
        
        # Copy Label (Handle txt/TXT)
        src_txt_path = os.path.join(label_train_path, base_name + ".txt")
        if not os.path.exists(src_txt_path):
            src_txt_path = os.path.join(label_train_path, base_name + ".TXT")
            
        shutil.copy(src_txt_path, os.path.join(label_train_path, new_txt_name))

    print(f"\n✅ SUKSES! Total file Falciparum sekarang: ~{vivax_count}")
    print("   Silakan jalankan Training Ulang sekarang!")

if __name__ == "__main__":
    balance_dataset_v2()