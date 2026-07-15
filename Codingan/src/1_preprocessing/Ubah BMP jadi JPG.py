import os
from PIL import Image
from tqdm import tqdm

# ==============================
# KONFIGURASI FOLDER
# ==============================
# Path menuju folder dataset mentah Anda (yang berisi folder "Sudah Anotasi 1-23")
source_dir = r"D:\DatasetMalaria1x1\images"

def convert_bmp_to_jpg():
    print("🔍 Memindai folder untuk mencari file .bmp...")
    
    bmp_files = []
    # os.walk akan menembus folder dan subfolder secara otomatis
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith('.bmp'):
                bmp_files.append(os.path.join(root, file))
                
    total_files = len(bmp_files)
    
    if total_files == 0:
        print("✅ Tidak ada file .bmp yang ditemukan. Semua gambar mungkin sudah berformat lain.")
        return

    print(f"⚠️ Ditemukan {total_files} file .bmp! Memulai proses konversi ke .jpg...")
    
    success_count = 0
    
    for bmp_path in tqdm(bmp_files):
        try:
            # 1. Buka gambar BMP
            with Image.open(bmp_path) as img:
                # Konversi ke RGB (Wajib dilakukan karena format JPEG tidak mendukung transparansi/Alpha Channel)
                rgb_im = img.convert('RGB')
                
                # 2. Tentukan nama file baru dengan ekstensi .jpg
                jpg_path = os.path.splitext(bmp_path)[0] + '.jpg'
                
                # 3. Simpan gambar dengan kualitas 100% agar data medis tidak rusak
                rgb_im.save(jpg_path, 'JPEG', quality=100)
                
            # 4. Hapus file BMP asli agar folder bersih dan YOLO tidak membaca data ganda
            os.remove(bmp_path)
            
            success_count += 1
            
        except Exception as e:
            print(f"\n❌ Gagal mengonversi file: {bmp_path}")
            print(f"Error: {e}")

    print("\n========================================")
    print(f"🎉 SELESAI! Berhasil mengonversi {success_count} dari {total_files} file.")
    print("Semua gambar sekarang berformat .jpg dan siap digunakan untuk YOLOv8.")
    print("========================================")

if __name__ == "__main__":
    convert_bmp_to_jpg()