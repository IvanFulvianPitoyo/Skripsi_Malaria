import os
import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO

# Menyembunyikan window utama tkinter agar tampilan rapi
root = tk.Tk()
root.withdraw()
root.attributes('-topmost', True) 

# Membuka dialog pemilihan file
sumber_gambar = filedialog.askopenfilename(
    title="Pilih Citra Sediaan Darah Mikroskopis",
    filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
)

if not sumber_gambar:
    print("❌ Dibatalkan: Tidak ada citra yang dipilih. Program dihentikan.")
else:
    print(f"✅ Citra terpilih: {sumber_gambar}")
    
    # Penentuan path relatif
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Memuat model terlatih
    path_model = os.path.join(script_dir, 'models', 'yolov8m_malaria(91%)1x2.pt') 
    
    if not os.path.exists(path_model):
        print(f"❌ Error: Model tidak ditemukan di lokasi: '{path_model}'")
        print("Pastikan file 'yolov8m_malaria(91%)1x2.pt' berada di dalam folder 'models'.")
    else:
        print("⏳ Memuat model YOLOv8m (Terlatih) dan memproses citra...")
        
        # Eksekusi model YOLO
        model = YOLO(path_model)
        
       # Eksekusi Prediksi dengan memaksa menggunakan CPU
        hasil = model.predict(
            source=sumber_gambar, 
            show=True,          
            save=True,          
            conf=0.65,          
            show_labels=True,   
            show_conf=True,
            imgsz=640,
            half=False,
            device='cpu'        # <--- TAMBAHKAN BARIS INI UNTUK MEM-BYPASS ERROR CUDA
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