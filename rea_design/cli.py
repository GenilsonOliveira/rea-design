"""Generate visual evidence; semantic interpretation belongs to the agent."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps, ImageStat, UnidentifiedImageError

from . import __version__


def load(path):
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        image.load()
        return image.convert("RGBA")


def flatten(image):
    background = Image.new("RGBA", image.size, "white")
    return Image.alpha_composite(background, image).convert("RGB")


def protect_inputs(outputs, *inputs):
    for output in outputs:
        output = Path(output)
        for source in map(Path, inputs):
            if output.resolve() == source.resolve() or (
                output.exists() and source.exists() and output.samefile(source)
            ):
                raise ValueError(f"Output would overwrite an input: {output}")


def inspect(source, output):
    source, output = Path(source), Path(output)
    paths = [output / name for name in ("evidence.json", "normalized.png", "edges.png", "grid.png")]
    paths.extend(output / "crops" / f"r{row}-c{col}.png" for row in (1, 2) for col in (1, 2))
    protect_inputs(paths, source)
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
    ref, cand = load(reference), load(candidate)
    if ref.size != cand.size:
        raise ValueError(f"Dimensions differ: {ref.size} vs {cand.size}; align the canvas before comparing")
    output = Path(output)
    protect_inputs((output / name for name in ("difference.png", "overlay.png", "comparison.json", "alpha-difference.png")), reference, candidate)
    output.mkdir(parents=True, exist_ok=True)
    alpha_difference = ImageChops.difference(ref.getchannel("A"), cand.getchannel("A"))
    alpha_difference.save(output / "alpha-difference.png")
    alpha_mae = ImageStat.Stat(alpha_difference).mean[0]
    ref, cand = flatten(ref), flatten(cand)
    difference = ImageChops.difference(ref, cand)
    difference.save(output / "difference.png")
    Image.blend(ref, cand, 0.5).save(output / "overlay.png")
    mae = sum(ImageStat.Stat(difference).mean) / 3
    result = {"width": ref.width, "height": ref.height, "mean_absolute_rgb_error": round(mae, 4),
              "mean_absolute_alpha_error": round(alpha_mae, 4),
              "note": "Pixel error (0–255) is alignment-sensitive and is not a perceptual quality score."}
    (output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(prog="rea-design", description="Generate evidence for an agent-led design breakdown.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
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
    p = sub.add_parser("fetch-prompt-models", help="Download pinned model weights for local text segmentation")
    p.add_argument("--models-dir", default="models")
    p = sub.add_parser("segment-text", help="Find one object from a text description and save its mask")
    p.add_argument("image")
    p.add_argument("--prompt", required=True, action="append", help="Object description; repeat for more objects")
    p.add_argument("--models-dir", default="models")
    p.add_argument("--threshold", type=float, default=0.25)
    p.add_argument("--out", required=True)
    p = sub.add_parser("geometry", help="Detect thin grid lines and solid color cells")
    p.add_argument("image")
    p.add_argument("--out", required=True)
    p = sub.add_parser("typography", help="Measure and split the dominant OCR title")
    p.add_argument("image")
    p.add_argument("--ocr", required=True, metavar="OCR_JSON")
    p.add_argument("--out", required=True)
    p = sub.add_parser("scene", help="Build a reviewable layer map from OCR and masks")
    p.add_argument("image")
    p.add_argument("--semantic")
    p.add_argument("--ocr")
    p.add_argument("--geometry", metavar="GEOMETRY_JSON", help="Import detected lines and blocks as SVG rectangles")
    p.add_argument("--typography", metavar="TYPOGRAPHY_JSON", help="Replace OCR fragments with an editable title block")
    p.add_argument("--grounded", metavar="GROUNDED_JSON", help="Import a mask found by segment-text")
    p.add_argument("--grounded-source", help="Aligned photographic source used by segment-text, if different from the reference")
    p.add_argument("--grabcut", metavar="SEGMENTS_JSON", action="append", default=[], help="Import a subject generated by segment --rect")
    p.add_argument("--mask", nargs=2, metavar=("LABEL", "PNG"), action="append", default=[], help="Add a full-canvas grayscale or alpha mask")
    p.add_argument("--replace-mask", nargs=2, metavar=("ID", "PNG"), action="append", default=[], help="Replace a detected object mask and keep its ID")
    p.add_argument("--out", required=True)
    p = sub.add_parser("prepare-repair", help="Write an overlap crop and inpainting mask for an image generator")
    p.add_argument("image")
    p.add_argument("--scene", required=True)
    p.add_argument("--clean-plate")
    p.add_argument("--layer", help="Receiving object ID, required when multiple layers have overlaps")
    p.add_argument("--margin", type=int, help="Feather width outside the overlap, in source pixels")
    p.add_argument("--out", required=True)
    p = sub.add_parser("export-svg", help="Export a reviewed scene as editable SVG")
    p.add_argument("image")
    p.add_argument("--scene", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--clean-plate", help="Use an aligned photograph without the graphics as the pixel source")
    p.add_argument("--background-plate", help="Use an aligned subject-free background instead of inpainting")
    repair = p.add_mutually_exclusive_group()
    repair.add_argument("--repair-plate", help="Aligned donor image for the pixels under foreground objects")
    repair.add_argument("--repair-model", metavar="LAMA_ONNX", help="Recompose inferred overlaps using local LaMa")
    repair.add_argument("--generative-repair", nargs=2, metavar=("REPAIR_JSON", "GENERATED_PNG"),
                        action="append", default=[], help="Apply a masked generative edit. Repeat for distinct layers.")
    p.add_argument("--move", nargs=3, metavar=("LAYER_ID", "DX", "DY"), action="append", default=[])
    p.add_argument("--title-line", action="append", help="Replace successive lines of the single editable title")
    p.add_argument("--title-only", action="store_true", help="Clean and replace the title while preserving the rest of the source")
    p.add_argument("--title-behind", action="append", default=[], metavar="LAYER_ID",
                   help="Explicit foreground object for --title-only; repeat for multiple layers")
    p.add_argument("--title-line-over", nargs=2, action="append", default=[], metavar=("LINE", "LAYER_ID"),
                   help="Paint a numbered title line over a foreground object in --title-only mode")
    p.add_argument("--title-paper", action="store_true", help="Reconstruct paper grain beneath the old title ink")
    p.add_argument("--title-overlay", nargs=2, action="append", default=[], metavar=("TEXT_LAYER_ID", "LABEL"),
                   help="Lift ink from an OCR layer and paint it above the replacement title")
    p.add_argument("--title-font", metavar="FONT_FILE", help="Measure the title using a local TTF or OTF")
    p = sub.add_parser("assemble", help="Detect objects, suggest layers and build an editable SVG in one command")
    p.add_argument("image")
    p.add_argument("--prompt", action="append", required=True, help="Object description; repeat for more objects")
    p.add_argument("--models-dir", default="models")
    p.add_argument("--threshold", type=float, default=0.25)
    p.add_argument("--clean-plate")
    repair = p.add_mutually_exclusive_group()
    repair.add_argument("--repair-plate")
    repair.add_argument("--repair-model", metavar="LAMA_ONNX", help="Recompose inferred overlaps using local LaMa")
    p.add_argument("--background-plate")
    p.add_argument("--title-font")
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
        elif args.command == "fetch-prompt-models":
            from .grounded import fetch_prompt_models
            result = fetch_prompt_models(args.models_dir)
        elif args.command == "segment-text":
            from .grounded import segment_texts
            result = segment_texts(args.image, args.prompt, args.out, models_dir=args.models_dir, threshold=args.threshold)
        elif args.command == "geometry":
            from .geometry import detect_geometry
            result = detect_geometry(args.image, args.out)
        elif args.command == "typography":
            from .typography import detect_typography
            result = detect_typography(args.image, args.ocr, args.out)
        elif args.command == "scene":
            from .scene import make_scene
            result = make_scene(args.image, args.semantic, args.ocr, args.out, geometry_json=args.geometry,
                                typography_json=args.typography, grounded_json=args.grounded, grounded_source=args.grounded_source,
                                grabcut=args.grabcut, manual=args.mask, replacements=args.replace_mask)
        elif args.command == "prepare-repair":
            from .generative import prepare_repair
            result = prepare_repair(args.image, args.scene, args.out, clean_plate=args.clean_plate,
                                    layer=args.layer, margin=args.margin)
        elif args.command == "assemble":
            from .assemble import assemble
            result = assemble(args.image, args.prompt, args.out, models_dir=args.models_dir, threshold=args.threshold,
                              clean_plate=args.clean_plate, repair_plate=args.repair_plate, repair_model=args.repair_model,
                              background_plate=args.background_plate, title_font=args.title_font)
        else:
            from .scene import export_svg
            result = export_svg(args.image, args.scene, args.out, clean_plate=args.clean_plate,
                                background_plate=args.background_plate, title_font=args.title_font,
                                repair_plate=args.repair_plate, repair_model=args.repair_model,
                                generative_repairs=args.generative_repair, moves=args.move, title_lines=args.title_line,
                                title_only=args.title_only, title_behind=args.title_behind,
                                title_overlays=args.title_overlay, title_line_over=args.title_line_over,
                                title_paper=args.title_paper)
        # JSON remains lossless even when Windows redirects stdout to a legacy code page.
        print(json.dumps(result, indent=2, ensure_ascii=True))
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        print(f"rea-design: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
