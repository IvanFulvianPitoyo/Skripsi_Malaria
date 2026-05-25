import os

# KITA TEMBAK LANGSUNG KE MASTER DATA AGAR BERES DARI AKAR
labels_dir = r"C:\Users\Ivan\OneDrive\Documents\Skripsi_Malaria\datasets\malaria\labels"

def shift_classes_and_fix_coords(label_folder):
    print(f"🔍 Memeriksa dan MENGUBAH KELAS (1->0, 2->1) di: {label_folder}")
    fixed_files = 0
    shifted_boxes = 0

    for filename in os.listdir(label_folder):
        if not filename.endswith(".txt") or filename == "classes.txt":
            continue

        filepath = os.path.join(label_folder, filename)
        with open(filepath, "r") as file:
            lines = file.readlines()

        new_lines = []
        needs_fix = False

        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x, y, w, h = map(float, parts[1:5])

                # MENGUBAH KELAS (SHIFTING)
                if class_id == 1:
                    class_id = 0
                    needs_fix = True
                    shifted_boxes += 1
                elif class_id == 2:
                    class_id = 1
                    needs_fix = True
                    shifted_boxes += 1

                # MENGUNCI KOORDINAT AGAR TIDAK OUT OF BOUNDS
                if x < 0 or x > 1 or y < 0 or y > 1 or w < 0 or w > 1 or h < 0 or h > 1:
                    needs_fix = True
                    x = max(0.000000, min(1.000000, x))
                    y = max(0.000000, min(1.000000, y))
                    w = max(0.000000, min(1.000000, w))
                    h = max(0.000000, min(1.000000, h))

                # Simpan baris yang sudah diubah kelasnya
                new_lines.append(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

        # Timpa file dengan data yang sudah benar
        if needs_fix:
            with open(filepath, "w") as file:
                file.writelines(new_lines)
            fixed_files += 1

    print(f"✅ SELESAI! {fixed_files} file berhasil diperbaiki.")
    print(f"🔄 Total kotak yang sukses diubah kelasnya (1->0 / 2->1): {shifted_boxes}")

if __name__ == "__main__":
    shift_classes_and_fix_coords(labels_dir)