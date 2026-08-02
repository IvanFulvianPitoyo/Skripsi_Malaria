import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO
import torch
import numpy as np
import threading

class MalariaGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistem Deteksi Malaria YOLOv8m")
        self.root.geometry("3024x4032")
        self.root.configure(bg="#e0e0e0")
        
        self.vid = None
        self.is_camera_on = False
        self.current_frame = None
        self.display_image = None
        self.zoom_factor = 1.0
        self.rotation_angle = 0
        self.camera_index = 0
        self.model = None
        self.model_path = None
        self.device = "cpu"
        self.switching_camera = False

        self.setup_ui()

    def setup_ui(self):
        self.control_frame = tk.Frame(self.root, width=250, bg="#a0a0a0", padx=10, pady=10)
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y)

        tk.Label(self.control_frame, text="Controls", bg="#808080", fg="white", font=("Arial", 12, "bold"), anchor="w").pack(fill=tk.X, pady=(0, 5))


        model_frame = tk.LabelFrame(self.control_frame, text="Model", bg="#f0f0f0", font=("Arial", 9, "bold"))
        model_frame.pack(fill=tk.X, pady=5)

        tk.Button(model_frame, text="Load Model", command=self.load_model, width=20).pack(pady=2)

        self.lbl_model_status = tk.Label(model_frame, text="No Model Loaded", fg="red", bg="#f0f0f0", font=("Arial", 8), wraplength=200, justify="left")
        self.lbl_model_status.pack(pady=5)


        cam_frame = tk.LabelFrame(self.control_frame, text="Camera", bg="#f0f0f0", font=("Arial", 9, "bold"))
        cam_frame.pack(fill=tk.X, pady=5)
        
        tk.Button(cam_frame, text="Start Camera", command=self.start_camera, width=20).pack(pady=2)
        tk.Button(cam_frame, text="Stop Camera", command=self.stop_camera, width=20).pack(pady=2)
        tk.Button(cam_frame, text="Switch Camera", command=self.switch_camera, width=20).pack(pady=2)
        tk.Button(cam_frame, text="Take Image", command=self.take_image, width=20).pack(pady=2)
        tk.Button(cam_frame, text="Check Camera", command=self.check_camera, width=20).pack(pady=2)
        
        self.lbl_cam_status = tk.Label(cam_frame, text="Camera Not Detected", fg="red", bg="#f0f0f0", font=("Arial", 8))
        self.lbl_cam_status.pack(pady=5)


        image_frame = tk.LabelFrame(self.control_frame, text="Image Input", bg="#f0f0f0", font=("Arial", 9, "bold"))
        image_frame.pack(fill=tk.X, pady=5)

        tk.Button(image_frame, text="Load Image", command=self.load_image, width=20).pack(pady=2)

        self.lbl_image_status = tk.Label(image_frame, text="No Image Loaded", fg="red", bg="#f0f0f0", font=("Arial", 8), wraplength=200, justify="left")
        self.lbl_image_status.pack(pady=5)

   
        flip_frame = tk.LabelFrame(self.control_frame, text="Flip Controls", bg="#f0f0f0", font=("Arial", 9, "bold"))
        flip_frame.pack(fill=tk.X, pady=5)

        btn_flip_frame = tk.Frame(flip_frame, bg="#f0f0f0")
        btn_flip_frame.pack(pady=5)
        tk.Button(btn_flip_frame, text="90°", command=lambda: self.set_rotation(90), width=5).grid(row=0, column=0, padx=2)
        tk.Button(btn_flip_frame, text="180°", command=lambda: self.set_rotation(180), width=5).grid(row=0, column=1, padx=2)
        tk.Button(btn_flip_frame, text="270°", command=lambda: self.set_rotation(270), width=5).grid(row=0, column=2, padx=2)

        tk.Label(flip_frame, text="Flip Slider", bg="#f0f0f0", font=("Arial", 8)).pack()
        self.flip_slider = tk.Scale(flip_frame, from_=0, to=360, orient=tk.HORIZONTAL, bg="#f0f0f0", command=self.slider_rotate)
        self.flip_slider.pack(fill=tk.X, padx=10, pady=5)


        zoom_frame = tk.LabelFrame(self.control_frame, text="Zoom Control", bg="#f0f0f0", font=("Arial", 9, "bold"))
        zoom_frame.pack(fill=tk.X, pady=5)

        tk.Button(zoom_frame, text="Zoom (+)", command=self.zoom_in, width=20).pack(pady=2)
        tk.Button(zoom_frame, text="Zoom (-)", command=self.zoom_out, width=20).pack(pady=2)
        tk.Button(zoom_frame, text="Reset", command=self.reset_view, width=20).pack(pady=2)
        tk.Label(zoom_frame, text="Mouse Wheel For Zoom\nDrag With Mouse", bg="#f0f0f0", fg="gray", font=("Arial", 8)).pack(pady=5)
        export_frame = tk.LabelFrame(self.control_frame, text="Export Image", bg="#f0f0f0", font=("Arial", 9, "bold"))
        export_frame.pack(fill=tk.X, pady=5)

        tk.Button(export_frame, text="Export Image", command=self.export_image, width=20).pack(pady=2)
    
        self.format_var = tk.StringVar(value="PNG")
        format_cb = ttk.Combobox(export_frame, textvariable=self.format_var, values=["PNG", "JPG", "BMP"], state="readonly", width=17)
        format_cb.pack(pady=2)
        
        tk.Button(export_frame, text="Save Location", command=self.export_image, width=20).pack(pady=2)

        self.view_frame = tk.Frame(self.root, bg="white")
        self.view_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        tk.Label(self.view_frame, text="Microscope View", bg="white", fg="gray", font=("Arial", 10), anchor="w").pack(fill=tk.X, padx=5, pady=2)
        self.canvas = tk.Canvas(self.view_frame, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.canvas.bind("<MouseWheel>", self.mouse_wheel_zoom)
        self.canvas.bind("<ButtonPress-1>", self.on_drag_start)
        self.canvas.bind("<B1-Motion>", self.on_drag_motion)

        self.drag_data = {"x": 0, "y": 0}

    def load_model(self):
        file_path = filedialog.askopenfilename(
            title="Pilih File Model YOLOv8 (.pt)",
            filetypes=[
                ("PyTorch Model", "*.pt"),
                ("All Files", "*.*")
            ]
        )

        if not file_path:
            return

        self.lbl_model_status.config(text="Loading model...", fg="blue")
        self.root.update_idletasks()

        try:
            self.model = YOLO(file_path)
            self.model_path = file_path
            self.device = "cpu"
            if torch.cuda.is_available():
                try:
                    dummy = np.zeros((64, 64, 3), dtype=np.uint8)
                    self.model.predict(dummy, device="cuda", verbose=False)
                    self.device = "cuda"
                except Exception:
                    self.device = "cpu"

            model_name = os.path.basename(file_path)
            device_label = "GPU" if self.device == "cuda" else "CPU"
            self.lbl_model_status.config(text=f"Model Loaded ({device_label}):\n{model_name}", fg="green")
        except Exception as e:
            self.model = None
            self.model_path = None
            self.lbl_model_status.config(text="No Model Loaded", fg="red")
            messagebox.showerror("Error", f"Gagal memuat model:\n{str(e)}")

    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Pilih Gambar untuk Dianalisis",
            filetypes=[
                ("Image Files", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif"),
                ("All Files", "*.*")
            ]
        )

        if not file_path:
            return

        if self.is_camera_on:
            self.stop_camera()

        img_bgr = cv2.imread(file_path)
        if img_bgr is None:
            messagebox.showerror("Error", "Gagal membuka file gambar. Pastikan file valid.")
            self.lbl_image_status.config(text="No Image Loaded", fg="red")
            return

        self.lbl_image_status.config(text="Processing image...", fg="blue")
        self.root.update_idletasks()

        if self.model is not None:
            try:
                results = self.model(img_bgr, verbose=False, device=self.device)
                annotated = results[0].plot()
                self.current_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            except Exception as e:
                self.current_frame = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                messagebox.showwarning("Peringatan", f"Gagal menjalankan deteksi pada gambar:\n{str(e)}")
        else:
            self.current_frame = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        self.reset_view()

        image_name = os.path.basename(file_path)
        self.lbl_image_status.config(text=f"Image Loaded:\n{image_name}", fg="green")

    def check_camera(self):
        temp_cap = cv2.VideoCapture(self.camera_index)
        if temp_cap.isOpened():
            self.lbl_cam_status.config(text=f"Camera {self.camera_index} Detected", fg="green")
            temp_cap.release()
        else:
            self.lbl_cam_status.config(text="Camera Not Detected", fg="red")

    def start_camera(self):
        if not self.is_camera_on:
            self.vid = cv2.VideoCapture(self.camera_index)
            if self.vid.isOpened():
                self.is_camera_on = True
                self.lbl_cam_status.config(text=f"Camera {self.camera_index} Active", fg="green")
                self.update_frame()
            else:
                messagebox.showerror("Error", "Kamera tidak ditemukan!")

    def stop_camera(self):
        self.is_camera_on = False
        if self.vid:
            self.vid.release()
        self.lbl_cam_status.config(text="Camera Stopped", fg="orange")
        self.canvas.delete("all")

    def switch_camera(self):
        if self.switching_camera:
            return
        self.switching_camera = True

        next_index = 1 if self.camera_index == 0 else 0

        was_on = self.is_camera_on
        self.is_camera_on = False
        if self.vid:
            self.vid.release()
            self.vid = None

        self.lbl_cam_status.config(text="Switching Camera...", fg="blue")

        threading.Thread(
            target=self._switch_camera_worker,
            args=(self.camera_index, next_index, was_on),
            daemon=True
        ).start()

    def _switch_camera_worker(self, current_index, next_index, was_on):
        test_cap = cv2.VideoCapture(next_index)
        alternate_available = test_cap.isOpened()
        test_cap.release()

        new_vid = None
        fallback_vid = None

        if alternate_available:
            if was_on:
                new_vid = cv2.VideoCapture(next_index)
                if not new_vid.isOpened():
                    new_vid.release()
                    new_vid = None
        else:
            if was_on:
                # Switch didn't happen - reopen the original camera so the feed comes back
                fallback_vid = cv2.VideoCapture(current_index)
                if not fallback_vid.isOpened():
                    fallback_vid.release()
                    fallback_vid = None

        self.root.after(0, self._switch_camera_finish, next_index, was_on, alternate_available, new_vid, fallback_vid)

    def _switch_camera_finish(self, next_index, was_on, alternate_available, new_vid, fallback_vid):
        if not alternate_available:
            self.lbl_cam_status.config(text="No other camera detected", fg="red")
            if fallback_vid is not None:
                self.vid = fallback_vid
                self.is_camera_on = True
                self.lbl_cam_status.config(text=f"Camera {self.camera_index} Active", fg="green")
                self.update_frame()
            self.switching_camera = False
            return

        self.camera_index = next_index

        if was_on:
            if new_vid is not None:
                self.vid = new_vid
                self.is_camera_on = True
                self.lbl_cam_status.config(text=f"Camera {self.camera_index} Active", fg="green")
                self.update_frame()
            else:
                self.lbl_cam_status.config(text="Camera Not Detected", fg="red")
        else:
            self.lbl_cam_status.config(text=f"Camera {self.camera_index} Selected", fg="blue")

        self.switching_camera = False

    def take_image(self):
    
        self.is_camera_on = False
        self.lbl_cam_status.config(text="Image Captured", fg="blue")

    def update_frame(self):
        if self.is_camera_on:
            ret, frame = self.vid.read()
            if ret:
                if self.model is not None:
                    try:
                        results = self.model(frame, verbose=False, device=self.device)
                        annotated_frame = results[0].plot()
                        self.current_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    except Exception:
                        # Fall back to the raw feed if inference fails on a frame
                        self.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    self.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.render_image()
            self.root.after(15, self.update_frame)
    def set_rotation(self, angle):
        self.rotation_angle = angle
        self.flip_slider.set(angle)
        self.render_image()

    def slider_rotate(self, val):
        self.rotation_angle = int(val)
        self.render_image()

    def zoom_in(self):
        self.zoom_factor *= 1.2
        self.render_image()

    def zoom_out(self):
        self.zoom_factor /= 1.2
        self.render_image()

    def reset_view(self):
        self.zoom_factor = 1.0
        self.rotation_angle = 0
        self.flip_slider.set(0)
        self.render_image()

    def mouse_wheel_zoom(self, event):
        if event.delta > 0:
            self.zoom_in()
        else:
            self.zoom_out()

    def on_drag_start(self, event):
        self.drag_data["x"] = event.x
        self.drag_data["y"] = event.y

    def on_drag_motion(self, event):
        dx = event.x - self.drag_data["x"]
        dy = event.y - self.drag_data["y"]
        self.canvas.move("all", dx, dy)
        self.drag_data["x"] = event.x
        self.drag_data["y"] = event.y

    def render_image(self):
        if self.current_frame is not None:
            img = Image.fromarray(self.current_frame)
            if self.rotation_angle != 0:
                img = img.rotate(-self.rotation_angle, expand=True)
            new_width = int(img.width * self.zoom_factor)
            new_height = int(img.height * self.zoom_factor)
            img = img.resize((new_width, new_height), Image.LANCZOS)
            self.display_image = ImageTk.PhotoImage(img)
            self.canvas.delete("all")
            canvas_w = self.canvas.winfo_width()
            canvas_h = self.canvas.winfo_height()
            self.canvas.create_image(canvas_w//2, canvas_h//2, anchor=tk.CENTER, image=self.display_image)

    def export_image(self):
        if self.current_frame is None:
            messagebox.showwarning("Peringatan", "Tidak ada gambar untuk disimpan. Mulai kamera atau ambil gambar terlebih dahulu.")
            return

        fmt = self.format_var.get().lower()
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=f".{fmt}",
            title="Lokasi Penyimpanan File",
            filetypes=[
                ("PNG File", "*.png"),
                ("JPEG File", "*.jpg"),
                ("BMP File", "*.bmp"),
                ("All Files", "*.*")
            ],
            initialfile="Screenshot_Malaria_Deteksi"
        )

        if file_path:
            try:
                img = Image.fromarray(self.current_frame)
                if self.rotation_angle != 0:
                    img = img.rotate(-self.rotation_angle, expand=True)
                
                img.save(file_path)
                messagebox.showinfo("Berhasil", f"Gambar berhasil disimpan di:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Gagal menyimpan gambar:\n{str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = MalariaGUI(root)
    root.mainloop()