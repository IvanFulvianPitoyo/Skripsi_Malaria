import os
import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO

# 1. Sembunyikan window utama tkinter agar hanya pop-up file yang muncul
root = tk.Tk()
root.withdraw()
# Agar jendela pop-up muncul di paling depan
root.attributes('-topmost', True) 

print("==================================================")
print("   SISTEM DETEKSI MALARIA BERBASIS YOLOv8m")
print("==================================================")
print("Menunggu Anda memilih file gambar...")

# 2. Buka jendela dialog untuk memilih gambar
sumber_gambar = filedialog.askopenfilename(
    title="Pilih Gambar Sediaan Darah Mikroskopis",
    filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
)

# 3. Cek apakah pengguna memilih gambar atau membatalkan
if not sumber_gambar:
    print("❌ Dibatalkan: Tidak ada gambar yang dipilih. Program dihentikan.")
else:
    print(f"✅ Gambar terpilih: {sumber_gambar}")
    
    # 4. Tentukan lokasi model secara dinamis (Anti-Gagal)
    # Ini akan mencari tahu secara otomatis di mana file uji_model.py ini berada
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Lalu menggabungkannya dengan folder 'models' dan file 'yolov8m.pt'
    path_model = os.path.join(script_dir, 'models', 'yolov8m.pt')
    
    if not os.path.exists(path_model):
        print(f"❌ Error: Model tidak ditemukan di lokasi absolut: '{path_model}'")
        print("Pastikan file 'yolov8m.pt' benar-benar ada di dalam folder 'models' yang sejajar dengan file skrip ini.")
    else:
        print(f"✅ Model berhasil ditemukan!")
        print("⏳ Memuat model YOLOv8m dan memproses gambar...")
        
        # 5. Eksekusi proses deteksi
        model = YOLO(path_model)
        hasil = model.predict(source=sumber_gambar, show=True, save=True)
        
        print("\n🎉 Deteksi berhasil diselesaikan!")
        print("📁 Gambar hasil inferensi otomatis disimpan di dalam folder 'runs/detect/predict'.")