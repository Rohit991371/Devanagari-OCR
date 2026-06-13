from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent

ASSETS_DIR = BASE_DIR / "assets"
FONTS_DIR = BASE_DIR / "fonts"

DATA_ROOT = Path(
    os.environ.get(
        "DATA_ROOT",
        str(BASE_DIR.parent / "Data")
    )
).resolve()

if __name__ == "__main__":
    print("BASE_DIR :", BASE_DIR)
    print("ASSETS_DIR:", ASSETS_DIR)
    print("FONTS_DIR :", FONTS_DIR)
    print("DATA_ROOT :", DATA_ROOT)