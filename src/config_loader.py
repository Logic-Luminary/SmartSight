import json
import os
import platform

if platform.system() == "Windows":
    import winreg

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../config.json")

DEFAULT_CONFIG = {
    "enable_depth": False,
    "enable_audio_panning": False,
    "show_fps": False,
    "skip_rate": 1,
    "theme": "dark",
    "enable_gamma_correction": True,
    "enable_brightness`_boost": True,
    "motion_confidence_fallback": True,
    "distance_tolerance": 0.5
}

def detect_windows_theme():
    try:
        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize"
        )
        value, _ = winreg.QueryValueEx(key, "AppUseLightTheme")
        return "dark" if value == 0 else "light"
    except Exception:
        return "dark"

def load_config():
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError("[WARNING⚠️] config.json file not found. Creating default config. . .")
        DEFAULT_CONFIG["theme"] = detect_windows_theme()
        save_config(DEFAULT_CONFIG)
        return DEFAULT_CONFIG

    try:
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        print("[ERROR] config.json is corrupted - resetting to default. . .")
        save_config(DEFAULT_CONFIG)
        return DEFAULT_CONFIG


def save_config(data):
    with open(CONFIG_PATH, "w") as f:
        json.dump(data, f, indent=4)