import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# KONFIGURASI PATH
# ==========================================
IMAGE_DIR = r"C:\Users\Ivan\OneDrive\Documents\Skripsi_Malaria\datasets\malaria\images\train"
LABEL_DIR = r"C:\Users\Ivan\OneDrive\Documents\Skripsi_Malaria\datasets\malaria\labels\train"
OUTPUT_DIR = r"D:\Visualisasi_Dataset_Bab3"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

CLASS_NAMES = {0: "P. falciparum", 1: "P. vivax"}
CLASS_COLORS = {0: (0, 0, 255), 1: (0, 255, 0)}

# --- KONFIGURASI RESIZE YOLOv8 ---
TARGET_SIZE = (640, 640) 

# ==========================================
# FUNGSI PEMBANTU
# ==========================================
def read_labels(label_path, width, height):
    """Membaca file teks YOLO dan mengonversinya menjadi pixel sesuai ukuran BARU"""
    bboxes = []
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                data = line.strip().split()
                if len(data) == 5:
                    class_id = int(data[0])
                    cx, cy, w, h = map(float, data[1:])
                    # Karena cx, cy, w, h adalah rasio (0-1), 
                    # kita kalikan langsung dengan width & height (640)
                    x1 = int((cx - w / 2) * width)
                    y1 = int((cy - h / 2) * height)
                    x2 = int((cx + w / 2) * width)
                    y2 = int((cy + h / 2) * height)
                    bboxes.append((class_id, x1, y1, x2, y2))
    return bboxes

def draw_boxes(img, bboxes):
    img_drawn = img.copy()
    for box in bboxes:
        class_id, x1, y1, x2, y2 = box
        color = CLASS_COLORS.get(class_id, (0, 255, 255))
        cv2.rectangle(img_drawn, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img_drawn, CLASS_NAMES.get(class_id, "Unknown"), (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return cv2.cvtColor(img_drawn, cv2.COLOR_BGR2RGB)

# ==========================================
# FUNGSI AUGMENTASI
# ==========================================
def apply_brightness(img, factor=1.5):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def apply_horizontal_flip(img, bboxes, width):
    img_flipped = cv2.flip(img, 1)
    flipped_bboxes = []
    for box in bboxes:
        class_id, x1, y1, x2, y2 = box
        new_x2 = width - x1
        new_x1 = width - x2
        flipped_bboxes.append((class_id, new_x1, y1, new_x2, y2))
    return img_flipped, flipped_bboxes

def apply_gaussian_noise(img):
    mean = 0
    var = 500
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, img.shape)
    noisy_img = img + gauss
    return np.clip(noisy_img, 0, 255).astype(np.uint8)

# ==========================================
# PROSES UTAMA
# ==========================================
print(f"Memproses gambar (Resize otomatis ke {TARGET_SIZE[0]}x{TARGET_SIZE[1]})...")

image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.png'))]

for img_file in image_files[:3]: # Mengambil 3 gambar sebagai contoh
    img_path = os.path.join(IMAGE_DIR, img_file)
    label_path = os.path.join(LABEL_DIR, os.path.splitext(img_file)[0] + ".txt")
    
    img_original = cv2.imread(img_path)
    if img_original is None:
        continue
    
    # ----------------------------------------------------
    # PROSES RESIZING (DILAKUKAN PALING AWAL)
    # ----------------------------------------------------
    img = cv2.resize(img_original, TARGET_SIZE)
    height, width, _ = img.shape # Sekarang height=640, width=640
    
    # 1. Baca koordinat menyesuaikan ukuran 640x640
    bboxes = read_labels(label_path, width, height)
    img_raw_rgb = draw_boxes(img, bboxes)
    
    # 2. Terapkan Brightness pada gambar 640x640
    img_bright = apply_brightness(img, factor=1.5)
    img_bright_rgb = draw_boxes(img_bright, bboxes)
    
    # 3. Terapkan Flip pada gambar 640x640
    img_flipped, flipped_bboxes = apply_horizontal_flip(img, bboxes, width)
    img_flipped_rgb = draw_boxes(img_flipped, flipped_bboxes)
    
    # 4. Terapkan Noise pada gambar 640x640
    img_noisy = apply_gaussian_noise(img)
    img_noisy_rgb = draw_boxes(img_noisy, bboxes)

    # ==========================================
    # GABUNGKAN GAMBAR (1 Baris, 4 Kolom)
    # ==========================================
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(img_raw_rgb)
    axes[0].set_title(f"1. Raw (Resized {width}x{height})")
    axes[0].axis('off')
    
    axes[1].imshow(img_bright_rgb)
    axes[1].set_title("2. Brightness Enhancement")
    axes[1].axis('off')
    
    axes[2].imshow(img_flipped_rgb)
    axes[2].set_title("3. Horizontal Flip")
    axes[2].axis('off')
    
    axes[3].imshow(img_noisy_rgb)
    axes[3].set_title("4. Gaussian Noise")
    axes[3].axis('off')
    
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIR, f"perbandingan_augmentasi_640_{img_file}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Berhasil disimpan: {save_path}")

print("\nProses selesai. Cek folder D:\\Visualisasi_Dataset_Bab3")