import cv2
import os
import numpy as np
import random
from tqdm import tqdm

# --- PASTIKAN PATH INI BENAR ---
base_path = r"D:\DatasetMalaria1x1"
images_dir = os.path.join(base_path, "images", "train")
labels_dir = os.path.join(base_path, "labels", "train")

def add_noise(image):
    """Menambahkan efek bintik-bintik (Noise)"""
    row, col, ch = image.shape
    mean = 0
    var = 0.5
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = image + gauss * 20
    return np.clip(noisy, 0, 255).astype(np.uint8)

def adjust_brightness(image):
    """Menggelapkan atau mencerahkan gambar secara acak"""
    factor = random.uniform(0.6, 1.4) 
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def augment_data():
    print("🎨 MEMULAI AUGMENTASI KHUSUS FALCIPARUM...")
    
    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        print("❌ Error: Folder train tidak ditemukan. Cek kembali path Anda.")
        return

    files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.bmp'))]
    falciparum_files = []

    # 1. MENCARI GAMBAR YANG MENGANDUNG FALCIPARUM (Kelas 0 atau Anomali > 1)
    for f in files:
        label_path = os.path.join(labels_dir, os.path.splitext(f)[0] + ".txt")
        if not os.path.exists(label_path): 
            label_path = os.path.join(labels_dir, os.path.splitext(f)[0] + ".TXT")
        
        has_falciparum = False
        if os.path.exists(label_path):
            with open(label_path, 'r') as file:
                for line in file.readlines():
                    parts = line.strip().split()
                    if not parts: continue
                    try:
                        cls_id = int(parts[0])
                        # Tangkap kelas 0, atau anomali seperti 15, 2, dll
                        if cls_id == 0 or cls_id > 1:
                            has_falciparum = True
                            break
                    except ValueError:
                        continue
                        
            if has_falciparum:
                falciparum_files.append((f, label_path))

    print(f"✅ Ditemukan {len(falciparum_files)} gambar yang mengandung Falciparum.")
    print("⏳ Sedang membuat variasi: Noise, Brightness, & Flip horizontal...")

    count = 0
    # 2. EKSEKUSI AUGMENTASI HANYA UNTUK FILE YANG TERPILIH
    for img_name, lbl_path in tqdm(falciparum_files):
        img_path = os.path.join(images_dir, img_name)   
        image = cv2.imread(img_path)
        
        if image is None: continue
        
        # Baca dan perbaiki label (ubah anomali jadi 0)
        corrected_labels = []
        with open(lbl_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if not parts: continue
                cls_id = int(parts[0])
                if cls_id > 1: 
                    cls_id = 0 # Paksa jadi 0 (Falciparum)
                parts[0] = str(cls_id)
                corrected_labels.append(" ".join(parts) + "\n")
        
        base_name = os.path.splitext(img_name)[0]
        
        # --- A. AUGMENTASI NOISE ---
        img_noise = add_noise(image)
        new_name_noise = f"{base_name}_aug_noise"
        cv2.imwrite(os.path.join(images_dir, new_name_noise + ".jpg"), img_noise)
        with open(os.path.join(labels_dir, new_name_noise + ".txt"), 'w') as f:
            f.writelines(corrected_labels) # Koordinat tetap sama
        count += 1
        
        # --- B. AUGMENTASI BRIGHTNESS ---
        img_bright = adjust_brightness(image)
        new_name_bright = f"{base_name}_aug_bright"
        cv2.imwrite(os.path.join(images_dir, new_name_bright + ".jpg"), img_bright)
        with open(os.path.join(labels_dir, new_name_bright + ".txt"), 'w') as f:
            f.writelines(corrected_labels) # Koordinat tetap sama
        count += 1

        # --- C. AUGMENTASI FLIP HORIZONTAL ---
        img_flip = cv2.flip(image, 1) 
        new_name_flip = f"{base_name}_aug_flip"
        cv2.imwrite(os.path.join(images_dir, new_name_flip + ".jpg"), img_flip)
        
        new_flip_labels = []
        for line in corrected_labels:
            parts = line.strip().split()
            cls_id = parts[0]
            x, y, w, h = map(float, parts[1:])
            new_x = 1.0 - x # Logika matematika balik koordinat X untuk YOLO
            new_flip_labels.append(f"{cls_id} {new_x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
            
        with open(os.path.join(labels_dir, new_name_flip + ".txt"), 'w') as f:
            f.writelines(new_flip_labels)
        count += 1

    print(f"\n🎉 SELESAI! Berhasil menciptakan {count} gambar variasi Falciparum baru.")
    print("👉 Dataset Train Anda sekarang sudah siap tempur. Silakan jalankan Final Training dengan Focal Loss!")

if __name__ == "__main__":
    augment_data()