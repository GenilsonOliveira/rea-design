"""Generate visual evidence; semantic interpretation belongs to the agent."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps, ImageStat, UnidentifiedImageError


def load(path):
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        image.load()
        return image.convert("RGBA")


def flatten(image):
    background = Image.new("RGBA", image.size, "white")
    return Image.alpha_composite(background, image).convert("RGB")


def inspect(source, output):
    source, output = Path(source), Path(output)
    image = load(source)
    w, h = image.size
    if w < 3 or h < 3:
        raise ValueError("Image must be at least 3 × 3 pixels")
    output.mkdir(parents=True, exist_ok=True)
    rgb = flatten(image)
    sample = rgb.copy()
    sample.thumbnail((512, 512))
    quantized = sample.quantize(colors=8, method=Image.Quantize.MEDIANCUT)
    colors = quantized.getcolors()
    palette = quantized.getpalette()
    total = sample.width * sample.height
    colors = sorted(colors, reverse=True)
    swatches = [{"hex": "#%02x%02x%02x" % tuple(palette[i*3:i*3+3]),
                 "fraction": round(count / total, 4)} for count, i in colors]
    alpha = image.getchannel("A")
    metadata = {"source_name": source.name, "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "width": w, "height": h, "aspect_ratio": round(w/h, 4),
                "alpha_extrema": list(alpha.getextrema()), "palette": swatches,
                "palette_method": "8-color median-cut quantization of a max 512px image, composited on white"}
    (output / "evidence.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    rgb.save(output / "normalized.png")
    rgb.convert("L").filter(ImageFilter.FIND_EDGES).save(output / "edges.png")
    grid = rgb.copy()
    draw = ImageDraw.Draw(grid)
    line = max(1, round(min(w, h) / 500))
    for n in (1, 2):
        draw.line((round(w*n/3), 0, round(w*n/3), h), fill="#ff00b7", width=line)
        draw.line((0, round(h*n/3), w, round(h*n/3)), fill="#ff00b7", width=line)
    grid.save(output / "grid.png")
    crops = output / "crops"
    crops.mkdir(exist_ok=True)
    # Overlapping halves preserve context at boundaries; names indicate position, not semantics.
    for row in range(2):
        for col in range(2):
            box = (int(col*w*0.4), int(row*h*0.4), int((col*0.4+0.6)*w), int((row*0.4+0.6)*h))
            rgb.crop(box).save(crops / f"r{row+1}-c{col+1}.png")
    return metadata


def compare(reference, candidate, output):
    ref, cand = flatten(load(reference)), flatten(load(candidate))
    if ref.size != cand.size:
        raise ValueError(f"Dimensions differ: {ref.size} vs {cand.size}; align the canvas before comparing")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    difference = ImageChops.difference(ref, cand)
    difference.save(output / "difference.png")
    Image.blend(ref, cand, 0.5).save(output / "overlay.png")
    mae = sum(ImageStat.Stat(difference).mean) / 3
    result = {"width": ref.width, "height": ref.height, "mean_absolute_rgb_error": round(mae, 4),
              "note": "Pixel error (0–255) is alignment-sensitive and is not a perceptual quality score."}
    (output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(prog="rea-design", description="Generate evidence for an agent-led design breakdown.")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("inspect", help="Extract measurements, grid, edges and context crops")
    p.add_argument("image")
    p.add_argument("--out", required=True)
    p = sub.add_parser("compare", help="Compare aligned images of the same size")
    p.add_argument("reference")
    p.add_argument("candidate")
    p.add_argument("--out", required=True)
    p = sub.add_parser("ocr", help="Detect text and polygons using local RapidOCR")
    p.add_argument("image")
    p.add_argument("--out", required=True)
    p = sub.add_parser("segment", help="Generate color region masks and optional GrabCut subject")
    p.add_argument("image")
    p.add_argument("--out", required=True)
    p.add_argument("--rect", type=int, nargs=4, metavar=("X", "Y", "WIDTH", "HEIGHT"))
    p.add_argument("--colors", type=int, default=8)
    p.add_argument("--min-area", type=float, default=0.001)
    p = sub.add_parser("semantic", help="Run FCN semantic segmentation with a local ONNX model")
    p.add_argument("image")
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--min-area", type=float, default=0.002)
    p = sub.add_parser("scene", help="Build a reviewable layer map from OCR and semantic evidence")
    p.add_argument("image")
    p.add_argument("--semantic", required=True)
    p.add_argument("--ocr", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("export-svg", help="Export a reviewed scene as editable SVG")
    p.add_argument("image")
    p.add_argument("--scene", required=True)
    p.add_argument("--out", required=True)
    args = parser.parse_args()
    try:
        if args.command == "inspect":
            result = inspect(args.image, args.out)
        elif args.command == "compare":
            result = compare(args.reference, args.candidate, args.out)
        elif args.command in ("ocr", "segment"):
            from .vision import ocr, segment
            result = ocr(args.image, args.out) if args.command == "ocr" else segment(args.image, args.out, args.rect, args.colors, args.min_area)
        elif args.command == "semantic":
            from .semantic import semantic
            result = semantic(args.image, args.out, args.model, args.min_area)
        else:
            from .scene import make_scene, export_svg
            result = make_scene(args.image, args.semantic, args.ocr, args.out) if args.command == "scene" else export_svg(args.image, args.scene, args.out)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        print(f"rea-design: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
