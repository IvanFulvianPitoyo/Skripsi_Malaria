import os
import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO

# Menyembunyikan window utama tkinter agar tampilan rapi
root = tk.Tk()
root.withdraw()
root.attributes('-topmost', True) 

print("==================================================")
print("   SISTEM DETEKSI MALARIA BERBASIS YOLOv8m")
print("==================================================")
print("Menunggu pemilihan file citra sediaan darah...")

# Membuka dialog pemilihan file
# (Sesuai dengan kesepakatan: GUI diimplementasikan sederhana, narasi proposal Bab 3 yang akan direvisi)
sumber_gambar = filedialog.askopenfilename(
    title="Pilih Citra Sediaan Darah Mikroskopis",
    filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
)

if not sumber_gambar:
    print("❌ Dibatalkan: Tidak ada citra yang dipilih. Program dihentikan.")
else:
    print(f"✅ Citra terpilih: {sumber_gambar}")
    
    # Penentuan path relatif (Menjawab Revisi Poin 3.4: Menghindari Hardcode Path Windows)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Memuat model terlatih, BUKAN model bawaan pabrik (Menjawab Revisi Poin 2.1)
    path_model = os.path.join(script_dir, 'models', 'yolov8m_malaria.pt') 
    
    if not os.path.exists(path_model):
        print(f"❌ Error: Model tidak ditemukan di lokasi: '{path_model}'")
        print("Pastikan file 'yolov8m_malaria.pt' berada di dalam folder 'models'.")
    else:
        print("⏳ Memuat model YOLOv8m (Terlatih - Trial 36) dan memproses citra...")
        
        # Eksekusi model YOLO
        model = YOLO(path_model)
        
        # Parameter imgsz=640 dikunci agar parasit mikroskopis tidak hancur karena penyusutan resolusi
        hasil = model.predict(
            source=sumber_gambar, 
            show=True,          
            save=True,          
            conf=0.25,          
            show_labels=True,   
            show_conf=True,
            imgsz=640           
        )
        
        print("\n==================================================")
        print("🎉 DETEKSI SELESAI! BERIKUT HASILNYA:")
        print("==================================================")
        
        # Ekstraksi dan pelaporan hasil deteksi secara interaktif di Terminal
        boxes = hasil[0].boxes
        if len(boxes) == 0:
            print("⚠️ Tidak ada parasit yang terdeteksi (True Negative).")
            print("💡 Catatan: Jika citra dipastikan positif, ini merupakan False Negative.")
        else:
            print(f"🦠 Ditemukan {len(boxes)} objek parasit pada citra:")
            for i, box in enumerate(boxes):
                class_id = int(box.cls[0].item())
                nama_kelas = model.names[class_id]
                akurasi = round(box.conf[0].item() * 100, 2)
                
                print(f"   ➤ Objek ke-{i+1}: [{nama_kelas}] dengan tingkat kepercayaan {akurasi}%")
        
        print("\n📁 Citra hasil deteksi (dengan Bounding Box) tersimpan di direktori 'runs/detect/predict'.")