import tkinter as tk
from tkinter import ttk
from config_loader import load_config, save_config

def launch_settings_ui(on_change = None):
    config = load_config()

    def update_config(key, val):
        config[key] = val
        save_config(config)
        if on_change:
            on_change(config)

    def apply_tkinter_theme(theme):
        bg = "#1e1e1e" if theme == "dark" else "#ffffff"
        fg = "#f0f0f0" if theme == "dark" else "#000000"
        hover_bg = "#333333" if theme == "dark" else "#dddddd"
        active_fg = "#ffffff" if theme == "dark" else "#000000"
        active_bg = "#444444" if theme == "dark" else "#cccccc"

        root.configure(bg=bg)

        style = ttk.Style()
        style.theme_use("default")

        style.configure("TLabel", background = bg, foreground = fg)
        style.configure("TFrame", background = bg)
        style.configure("TCheckbutton",
                        background = bg,
                        foreground = fg,
                        focuscolor = bg)
        style.map("TCheckbutton",
                  background = [("active", hover_bg)],
                  foreground = [("active", active_fg)])

        style.configure("TButton", background = bg, foreground = fg)

        style.map("TButton",
                background = [("active", active_bg)],
                foreground = [("active", active_fg)])

        style.configure("TScale", background = bg, troughcolor = hover_bg)
        style.configure("Horizontal.TScale", background = bg, troughcolor = hover_bg)

    root = tk.Tk()
    root.title("SmartSight Settings")
    root.geometry("300x420")
    apply_tkinter_theme(config.get("theme", "dark"))

    toggles = {
        "enable_depth": "Enable Depth",
        "enable_audio_panning": "Audio Panning",
        "show_fps": "Show FPS",
    }

    for key, label in toggles.items():
        var = tk.BooleanVar(value = config.get(key, True))
        cb = ttk.Checkbutton(root, text = label, variable = var,
                             command = lambda k = key, v = var: update_config(k, v.get()))
        cb.pack(anchor = "w", padx = 20, pady = 6)

    ttk.Label(root, text = "Frame Skip Rate").pack(pady = (20, 5))
    skip_var = tk.IntVar(value = config.get("skip_rate", 2))
    ttk.Scale(root, from_ = 1, to = 10, variable = skip_var,
              command=lambda e: update_config("skip_rate", int(skip_var.get()))
              ).pack(fill = "x", padx = 20)

    ttk.Label(root, text = "Select Theme").pack(pady = (20, 5))
    theme_var = tk.StringVar(value = config.get("theme", "dark"))
    ttk.OptionMenu(
        root, theme_var, config.get("theme", "dark"),
        "dark", "light",
        command = lambda v: [apply_tkinter_theme(v), update_config("theme", v)]
    ).pack(fill = "x", padx = 20)

    ttk.Button(root, text = "Close", command = root.destroy).pack(pady = 30)
    root.mainloop()