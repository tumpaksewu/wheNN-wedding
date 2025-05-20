import tkinter as tk
from tkinter import filedialog


def select_folder():
    e = None
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes("-topmost", 1)

    try:
        folder = filedialog.askdirectory(title="Выберите папку с видео")
        if folder:
            return folder, e
    except Exception as e:
        return None, e
