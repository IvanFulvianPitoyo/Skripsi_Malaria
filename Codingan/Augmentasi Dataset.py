import cv2
import os
import numpy as np
import random
from tqdm import tqdm

base_path = r"C:\Users\Ivan\OneDrive\Documents\Skripsi_Malaria\datasets\malaria"
images_dir = os.path.join(base_path, "images", "train")
labels_dir = os.path.join(base_path, "labels", "train")

TARGET_CLASS = 0 

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
    print("🎨 MEMULAI AUGMENTASI CERDAS (VARIASI GAMBAR)...")
    
    files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.bmp'))]
    falciparum_files = []

    for f in files:
        label_path = os.path.join(labels_dir, os.path.splitext(f)[0] + ".txt")
        if not os.path.exists(label_path): label_path = os.path.join(labels_dir, os.path.splitext(f)[0] + ".TXT")
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as file:
                content = file.read()
                if f"{TARGET_CLASS} " in content:
                    falciparum_files.append((f, label_path))

    print(f"   Ditemukan {len(falciparum_files)} gambar Falciparum asli.")
    print("   Sedang membuat variasi: Noise, Brightness, & Flip...")

    count = 0
    for img_name, lbl_path in tqdm(falciparum_files):
        img_path = os.path.join(images_dir, img_name)
        image = cv2.imread(img_path)
        
        if image is None: continue
        
        # Baca Label Asli
        with open(lbl_path, 'r') as f:
            labels = f.readlines()
        
        base_name = os.path.splitext(img_name)[0]
        
        img_noise = add_noise(image)
        new_name = f"{base_name}_aug_noise"
        cv2.imwrite(os.path.join(images_dir, new_name + ".jpg"), img_noise)
        with open(os.path.join(labels_dir, new_name + ".txt"), 'w') as f:
            f.writelines(labels)
        count += 1
        
        img_bright = adjust_brightness(image)
        new_name = f"{base_name}_aug_bright"
        cv2.imwrite(os.path.join(images_dir, new_name + ".jpg"), img_bright)
        with open(os.path.join(labels_dir, new_name + ".txt"), 'w') as f:
            f.writelines(labels) 
        count += 1

        img_flip = cv2.flip(image, 1) 
        new_name = f"{base_name}_aug_flip"
        cv2.imwrite(os.path.join(images_dir, new_name + ".jpg"), img_flip)
        
        new_labels = []
        for line in labels:
            parts = line.split()
            cls = int(parts[0])
            x, y, w, h = map(float, parts[1:])
            
            new_x = 1.0 - x
            new_labels.append(f"{cls} {new_x} {y} {w} {h}\n")
            
        with open(os.path.join(labels_dir, new_name + ".txt"), 'w') as f:
            f.writelines(new_labels) Tamba
        count += 1

    print(f"\n✅ SELESAI! Berhasil membuat {count} gambar baru.")
    print("   Sekarang dataset Anda seimbang DENGAN VARIASI (Bukan Copy-Paste).")
    print("   Silakan lanjut ke Training Final.")

if __name__ == "__main__":
    augment_data()