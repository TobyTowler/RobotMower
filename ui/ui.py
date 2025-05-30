import customtkinter as ctk
from tkinter import filedialog
import sys
import os

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import run


class PathPlannerGUI:
    def __init__(self):
        self.img = ""
        self.useGPS = False

        ctk.set_appearance_mode("dark")
        self.app = ctk.CTk()
        self.app.title("Path Planner")
        self.app.geometry("1800x1000")

        self.create_widgets()

    def create_widgets(self):
        """Create and layout all GUI widgets"""
        self.imgButton = ctk.CTkButton(
            self.app, text="Choose image", command=self.imageButton
        )
        self.imgButton.grid(row=0, column=0, padx=20, pady=20)

        self.startButton = ctk.CTkButton(self.app, text="Run", command=self.startRun)
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

        self.gps1 = ctk.CTkEntry(
            self.app,
            placeholder_text="First coord",
            width=200,
            height=30,
        )
        self.gps1.grid(row=1, column=2, padx=20, pady=20, sticky="w")

        self.gps2 = ctk.CTkEntry(
            self.app,
            placeholder_text="Second coord",
            width=200,
            height=30,
        )
        self.gps2.grid(row=1, column=4, padx=20, pady=20, sticky="w")

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

    def startRun(self):
        run(self.img, self)

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
