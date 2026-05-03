import os

# --- KONFIGURASI ---
# Lokasi dataset di Drive D
dataset_path = r"D:\datasets\malaria"

# Lokasi file YAML akan disimpan (Bebas, tapi saran saya taruh di D juga)
yaml_save_path = r"D:\datasets\data_malaria_canny.yaml"

# Isi File YAML
# Perhatikan: 'path' adalah alamat absolut, 'train'/'val' adalah relatif terhadap 'path'
yaml_content = f"""
path: {dataset_path}
train: images/train
val: images/val

# Jumlah Kelas (Classes)
nc: 2

# Nama Kelas (Sesuai urutan index 0 dan 1)
names:
  0: P.falciparum
  1: P.vivax
"""

# Tulis ke file
try:
    with open(yaml_save_path, 'w') as f:
        f.write(yaml_content)
    print("✅ BERHASIL!")
    print(f"File YAML sudah dibuat di: {yaml_save_path}")
    print("Isinya:")
    print("-" * 20)
    print(yaml_content)
    print("-" * 20)
except Exception as e:
    print(f"❌ Gagal membuat file: {e}")