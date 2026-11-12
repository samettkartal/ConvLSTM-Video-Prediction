"""Görsel özeti tek sayfalık PDF yapar (img2pdf veya Pillow fallback)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CANDIDATES = [
    ROOT / "layers_ozet_page.jpg",
    Path(__file__).resolve().parent.parent.parent.parent
    / ".cursor"
    / "projects"
    / "C-Users-samet-AppData-Local-Temp-colleague-git-suqzs1pv"
    / "assets"
    / "layers_ozet_page.jpg",
]
OUTPUT = ROOT / "layers_ozet.pdf"


def find_image() -> Path:
    for p in CANDIDATES:
        if p.is_file():
            return p
    raise FileNotFoundError("layers_ozet_page.jpg bulunamadi; once GenerateImage veya HTML kullanin.")


def main() -> None:
    src = find_image()
    try:
        import img2pdf
    except ImportError:
        img2pdf = None
    if img2pdf is not None:
        OUTPUT.write_bytes(img2pdf.convert(str(src)))
    else:
        from PIL import Image

        im = Image.open(src).convert("RGB")
        im.save(OUTPUT, "PDF", resolution=150.0)
    print(f"PDF yazildi: {OUTPUT}")


if __name__ == "__main__":
    main()
