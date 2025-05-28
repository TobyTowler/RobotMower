import customtkinter as ctk
from tkinter import filedialog

import sys
import os


# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import run


img = ""


def imageButton():
    global img
    file_path = filedialog.askopenfilename(
        title="Choose an image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.gif *.bmp")],
    )
    if file_path:
        print(f"Selected: {file_path}")
        img = file_path


def startRun():
    global img
    run(img)


def begin():
    ctk.set_appearance_mode("dark")
    app = ctk.CTk()
    app.title("Path Planner")
    app.geometry("1800 x 1000")

    imgButton = ctk.CTkButton(app, text="Choose image", command=imageButton)
    imgButton.grid(row=0, column=0, padx=20, pady=20)

    startButton = ctk.CTkButton(app, text="Run", command=startRun)
    startButton.grid(row=2, column=0, padx=20, pady=20)

    app.mainloop()
