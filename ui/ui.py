import customtkinter as ctk
from tkinter import filedialog
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import run


class PathPlannerGUI:
    def __init__(self):
        self.img = ""
        self.useGPS = False
        self.outline_path = None
        self.transformer = None

        ctk.set_appearance_mode("dark")
        self.app = ctk.CTk()
        self.app.title("Path Planner")
        self.app.geometry("1920x1000")
        self.app.resizable(False, False)

        self.create_widgets()

    def create_widgets(self):
        self.imgButton = ctk.CTkButton(
            self.app, text="Choose image", command=self.imageButton
        )
        self.imgButton.grid(row=0, column=0, padx=20, pady=20)

        self.scanButton = ctk.CTkButton(
            self.app, text="Scan Image", command=self.scanImage, state="disabled"
        )
        self.scanButton.grid(row=0, column=1, padx=20, pady=20)

        self.startButton = ctk.CTkButton(
            self.app, text="Continue", command=self.startRun, state="disabled"
        )
        self.startButton.grid(row=0, column=2, padx=20, pady=20)

        self.gpsBox = ctk.CTkCheckBox(
            self.app, text="Use GPS coordinates", command=self.checkGPS
        )
        self.gpsBox.grid(row=1, column=0, padx=20, pady=20)

        self.progressBar = ctk.CTkProgressBar(self.app, width=135)
        self.progressBar.grid(row=2, column=0, padx=20, pady=20)
        self.progressBar.set(0)

        self.progressStep = ctk.CTkLabel(self.app, text="")
        self.progressStep.grid(row=2, column=2, padx=20, pady=20)

        self.gps1_label = ctk.CTkLabel(self.app, text="Top-left GPS (lat,lon):")
        self.gps1_label.grid(row=1, column=1, padx=5, pady=20, sticky="e")

        self.gps1 = ctk.CTkEntry(
            self.app,
            placeholder_text="53.889408,9.547150",
            width=200,
            height=30,
        )
        self.gps1.grid(row=1, column=2, padx=5, pady=20, sticky="w")

        self.gps2_label = ctk.CTkLabel(self.app, text="Top-right GPS (lat,lon):")
        self.gps2_label.grid(row=1, column=3, padx=5, pady=20, sticky="e")

        self.gps2 = ctk.CTkEntry(
            self.app,
            placeholder_text="54.886907,9.546503",
            width=200,
            height=30,
        )
        self.gps2.grid(row=1, column=4, padx=5, pady=20, sticky="w")

        self.corner_frame = ctk.CTkFrame(self.app)
        self.corner_frame.grid(
            row=3, column=0, columnspan=5, padx=20, pady=20, sticky="ew"
        )

        self.corner_label = ctk.CTkLabel(
            self.corner_frame, text="Detected Corner Points:"
        )
        self.corner_label.pack(pady=10)

        self.corner_info = ctk.CTkLabel(
            self.corner_frame, text="Scan image first to see corner points"
        )
        self.corner_info.pack(pady=5)

        self.image_frame = ctk.CTkFrame(self.app, width=1850, height=780)
        self.image_frame.grid(row=4, column=0, columnspan=5, padx=35, pady=20)
        self.image_frame.grid_propagate(False)

    def updateProgress(self, progress_value):
        if progress_value > 1.0:
            progress_value = max(0, min(100, progress_value)) / 100
        else:
            progress_value = max(0.0, min(1.0, progress_value))

        self.app.after(0, lambda: self.progressBar.set(progress_value))

    def updateStep(self, step):
        self.progressStep.configure(text=step)

    def imageButton(self):
        file_path = filedialog.askopenfilename(
            title="Choose an image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.gif *.bmp")],
        )
        if file_path:
            print(f"Selected: {file_path}")
            self.img = file_path
            self.scanButton.configure(state="normal")

    def scanImage(self):
        if not self.img:
            return

        threading.Thread(target=self._scan_image_thread, daemon=True).start()

    def _scan_image_thread(self):
        try:
            try:
                from aerialMapping.runModel import run_model_and_get_outlines
                from aerialMapping.utils import save_outlines_to_json
            except ImportError:
                parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                if parent_dir not in sys.path:
                    sys.path.append(parent_dir)
                from aerialMapping.runModel import run_model_and_get_outlines
                from aerialMapping.utils import save_outlines_to_json

            self.updateStep("Running detection model...")
            self.updateProgress(20)

            outline = run_model_and_get_outlines(self.img)
            self.updateProgress(50)

            self.outline_path = save_outlines_to_json(outline, self.img)
            self.updateProgress(70)

            try:
                from pathing.mapTransform2 import SimpleGPSTransformer
            except ImportError:
                import json
                import numpy as np

                class SimpleGPSTransformer:
                    def __init__(self, json_file_path):
                        with open(json_file_path, "r") as f:
                            self.data = json.load(f)

                        all_points = []
                        for detection in self.data["detections"]:
                            all_points.extend(detection["outline_points"])
                        all_points = np.array(all_points)

                        min_x, min_y = np.min(all_points, axis=0)
                        max_x, max_y = np.max(all_points, axis=0)

                        top_left_target = [min_x, min_y]
                        top_right_target = [max_x, min_y]

                        distances_tl = np.sum(
                            (all_points - top_left_target) ** 2, axis=1
                        )
                        closest_tl_idx = np.argmin(distances_tl)

                        distances_tr = np.sum(
                            (all_points - top_right_target) ** 2, axis=1
                        )
                        closest_tr_idx = np.argmin(distances_tr)

                        self.corners = {
                            "top_left": tuple(all_points[closest_tl_idx]),
                            "top_right": tuple(all_points[closest_tr_idx]),
                        }

            self.transformer = SimpleGPSTransformer(self.outline_path)

            self.app.after(0, self._update_corner_display)

            self.updateProgress(100)
            self.updateStep("Scan complete - Set GPS coordinates and click Continue")

            self.app.after(0, lambda: self.startButton.configure(state="normal"))

        except Exception as e:
            print(f"Error during scanning: {e}")
            import traceback

            traceback.print_exc()
            self.app.after(0, lambda: self.updateStep(f"Error: {e}"))

    def _update_corner_display(self):
        if not self.transformer:
            return

        corners = self.transformer.corners
        corner_text = f"Top-left corner: {corners['top_left']}\nTop-right corner: {corners['top_right']}"
        corner_text += (
            f"\n\nEnter GPS coordinates for these points above, then click Continue"
        )

        self.corner_info.configure(text=corner_text)

        self._show_corner_visualization()

    def _show_corner_visualization(self):
        if not self.transformer:
            return

        for widget in self.image_frame.winfo_children():
            widget.destroy()

        fig, ax = plt.subplots(1, 1, figsize=(18, 9))
        fig.patch.set_facecolor("#2b2b2b")
        ax.set_facecolor("#2b2b2b")

        colors = {
            "bunker": "#8B4513",
            "green": "#228B22",
            "fairway": "#90EE90",
            "rough": "#556B2F",
            "tee": "#FFD700",
        }

        for detection in self.transformer.data["detections"]:
            feature_class = detection["class"]
            outline_points = detection["outline_points"]

            from matplotlib.patches import Polygon

            polygon = Polygon(
                outline_points,
                facecolor=colors.get(feature_class, "#808080"),
                edgecolor="white",
                alpha=0.7,
                linewidth=1,
            )
            ax.add_patch(polygon)

        corner_colors = ["red", "blue"]
        corner_labels = ["TOP-LEFT", "TOP-RIGHT"]

        for i, (name, (x, y)) in enumerate(self.transformer.corners.items()):
            circle = plt.Circle(
                (x, y),
                radius=25,
                color=corner_colors[i],
                fill=True,
                alpha=0.8,
                zorder=10,
            )
            ax.add_patch(circle)

            ax.text(
                x,
                y - 50,
                f"{corner_labels[i]}\n({x}, {y})",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="white",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor=corner_colors[i],
                    edgecolor="white",
                    linewidth=2,
                ),
            )

        all_points = [
            p
            for detection in self.transformer.data["detections"]
            for p in detection["outline_points"]
        ]
        if all_points:
            xs, ys = zip(*all_points)
            ax.set_xlim(min(xs) - 50, max(xs) + 50)
            ax.set_ylim(min(ys) - 50, max(ys) + 50)

        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_title(
            "Detected Features - Enter GPS for Red/Blue Corners",
            fontsize=12,
            fontweight="bold",
            color="white",
        )
        ax.grid(True, alpha=0.3, color="white")
        ax.tick_params(colors="white")

        canvas = FigureCanvasTkAgg(fig, self.image_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(padx=10, pady=10)

    def startRun(self):
        if not self.outline_path:
            print("No scanned data available")
            return

        run(self.img, self, existing_outline_path=self.outline_path)

    def checkGPS(self):
        self.useGPS = not self.useGPS
        print(f"GPS is now: {'ON' if self.useGPS else 'OFF'}")

    def run(self):
        self.app.mainloop()

    def get_gps_setting(self):
        return self.useGPS

    def get_image_path(self):
        return self.img


def begin():
    gui = PathPlannerGUI()
    gui.run()


if __name__ == "__main__":
    begin()
