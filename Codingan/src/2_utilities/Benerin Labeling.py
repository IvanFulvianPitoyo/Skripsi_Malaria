import os

# Sesuaikan dengan path folder label mentah Anda (sebelum di-split)
# Contoh: r"C:\Users\Ivan\OneDrive\Documents\Skripsi_Malaria\Data_Asli\labels"
labels_dir = r"C:\Users\Ivan\OneDrive\Documents\Skripsi_Malaria\datasets\malaria\labels" 

def fix_yolo_labels(label_folder):
    print(f"🔍 Memeriksa dan memperbaiki label di: {label_folder}")
    fixed_count = 0
    
    # Loop untuk membaca setiap file .txt di dalam folder
    for filename in os.listdir(label_folder):
        if not filename.endswith(".txt"):
            continue
            
        filepath = os.path.join(label_folder, filename)
        with open(filepath, 'r') as file:
            lines = file.readlines()
            
        new_lines = []
        needs_fix = False
        
        for line in lines:
            parts = line.strip().split()
            # YOLO format: class_id x_center y_center width height
            if len(parts) >= 5:
                class_id = parts[0]
                x, y, w, h = map(float, parts[1:5])
                
                # Deteksi jika ada nilai yang minus (<0) atau kelewatan batas (>1)
                if x < 0 or x > 1 or y < 0 or y > 1 or w < 0 or w > 1 or h < 0 or h > 1:
                    needs_fix = True
                    
                # KOREKSI (CLIPPING): Paksa nilai agar mentok di 0.0 atau 1.0
                x = max(0.000000, min(1.000000, x))
                y = max(0.000000, min(1.000000, y))
                w = max(0.000000, min(1.000000, w))
                h = max(0.000000, min(1.000000, h))
                
                # Masukkan kembali ke daftar baris baru dengan format 6 desimal
                new_lines.append(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
        
        # Jika file tersebut bermasalah, timpa (overwrite) dengan data yang sudah bersih
        if needs_fix:
            with open(filepath, 'w') as file:
                file.writelines(new_lines)
            fixed_count += 1
            
    print(f"✅ Selesai! Berhasil memperbaiki {fixed_count} file label yang error/out of bounds.")

if __name__ == "__main__":
    fix_yolo_labels(labels_dir)