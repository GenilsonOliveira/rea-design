"""Rebuild this reference with editable text and geometry plus the extracted subject.

Run after the segment command described in DESIGN.md. Font substitutions are expected.
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent


def font(size, names):
    for name in names:
        for candidate in (name, str(Path("C:/Windows/Fonts") / name)):
            try:
                return ImageFont.truetype(candidate, size)
            except OSError:
                pass
    return ImageFont.load_default(size=size)


def main():
    canvas = Image.new("RGBA", (415, 568), "#fce9da")
    d = ImageDraw.Draw(canvas)
    d.ellipse((4, 25, 167, 188), fill="#1660a5")
    d.ellipse((8, 29, 163, 184), outline="white", width=3)
    d.ellipse((13, 34, 158, 179), outline="#78b6db", width=1)
    bold = ["impact.ttf", "DejaVuSans-Bold.ttf"]
    italic = ["segoescb.ttf", "DejaVuSans-BoldOblique.ttf"]
    regular = ["arial.ttf", "DejaVuSans.ttf"]
    d.text((56, 39), "Tome", font=font(24, italic), fill="#ffc300", stroke_width=0)
    d.text((25, 73), "INCA", font=font(35, bold), fill="white")
    d.text((51, 107), "KOLA", font=font(35, bold), fill="white")
    d.text((79, 147), "MARCA REG.", font=font(6, bold), fill="white")
    d.text((168, 35), "mini", font=font(22, italic), fill="#1c3038")
    d.text((239, 35), "N O T A S", font=font(25, regular), fill="#1c3038")
    canvas.alpha_composite(Image.open(ROOT / "segments/subject.png").convert("RGBA"))
    d = ImageDraw.Draw(canvas)
    d.text((20, 502), "¡De sabor nacional!", font=font(33, italic), fill="#ec277a", stroke_width=0)
    canvas.convert("RGB").save(ROOT / "reconstruction.png")


if __name__ == "__main__":
    main()
