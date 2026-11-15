"""Optional local OCR and image segmentation. Coordinates are original pixels."""
import json
from pathlib import Path

from PIL import Image, ImageDraw
from .cli import load, flatten


def _deps():
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise ValueError("Vision dependencies missing; install rea-design[vision]") from exc
    return cv2, np


def ocr(source, output):
    cv2, np = _deps()
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError as exc:
        raise ValueError("OCR dependencies missing; install rea-design[vision]") from exc
    image = flatten(load(source))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # Pass an array rather than a path so EXIF orientation and Unicode paths work.
    bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    results, _ = RapidOCR()(bgr)
    words = []
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    for polygon, text, confidence in results or []:
        points = [[round(float(x), 2), round(float(y), 2)] for x, y in polygon]
        xs, ys = zip(*points)
        words.append({"id": f"text-{len(words)+1:03d}", "text": text,
                      "confidence": round(float(confidence), 4), "polygon": points,
                      "bbox": [round(min(xs), 2), round(min(ys), 2), round(max(xs)-min(xs), 2), round(max(ys)-min(ys), 2)]})
        draw.line([tuple(p) for p in points] + [tuple(points[0])], fill="#ff00b7", width=2)
    data = {"width": image.width, "height": image.height, "engine": "RapidOCR ONNX Runtime", "items": words,
            "note": "Text and confidence are model estimates. Check stylized, small or rotated text visually."}
    (output / "ocr.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    overlay.save(output / "ocr-overlay.png")
    return data


def segment(source, output, rect=None, colors=8, min_area=0.001):
    cv2, np = _deps()
    if not 2 <= colors <= 16 or not 0 <= min_area <= 1:
        raise ValueError("colors must be 2–16 and min-area must be 0–1")
    image = flatten(load(source))
    w, h = image.size
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    masks = output / "masks"
    masks.mkdir(exist_ok=True)
    rgb = np.array(image)
    # Deterministic median-cut provides connected same-color regions, not semantic objects.
    q = image.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
    indices = np.array(q)
    palette = q.getpalette()
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    regions = []
    for index in sorted(int(i) for i in np.unique(indices)):
        mask = (indices == index).astype("uint8")
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for component in range(1, count):
            x, y, rw, rh, area = (int(v) for v in stats[component])
            if area < max(1, w*h*min_area):
                continue
            rid = f"region-{len(regions)+1:03d}"
            Image.fromarray(((labels == component) * 255).astype("uint8")).save(masks / f"{rid}.png")
            color = "#%02x%02x%02x" % tuple(palette[index*3:index*3+3])
            regions.append({"id": rid, "bbox": [x, y, rw, rh], "area_pixels": area, "color": color, "mask": f"masks/{rid}.png"})
            draw.rectangle((x, y, x+rw-1, y+rh-1), outline="#ff00b7", width=2)
            draw.text((x, y), rid, fill="#ff00b7")
    subject = None
    if rect is not None:
        x, y, rw, rh = rect
        if x < 0 or y < 0 or rw < 2 or rh < 2 or x+rw > w or y+rh > h or rw*rh == w*h:
            raise ValueError("rect must fit inside the image, be at least 2x2, and leave some background outside")
        mask = np.zeros((h, w), np.uint8)
        bg, fg = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), mask, (x, y, rw, rh), bg, fg, 5, cv2.GC_INIT_WITH_RECT)
        except cv2.error as exc:
            raise ValueError(f"GrabCut failed for this rectangle: {exc}") from exc
        foreground = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype("uint8")
        Image.fromarray(foreground).save(output / "subject-mask.png")
        rgba = image.convert("RGBA")
        rgba.putalpha(Image.fromarray(foreground))
        rgba.save(output / "subject.png")
        subject = {"method": "GrabCut initialized by user rectangle", "rect": list(rect), "mask": "subject-mask.png", "cutout": "subject.png", "pixels": int(np.count_nonzero(foreground))}
    data = {"width": w, "height": h, "method": "median-cut colors and connected components", "regions": regions,
            "subject": subject, "note": "Regions are color components, not object or original layer detection; masks may fragment or merge objects."}
    (output / "segments.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    overlay.save(output / "segments-overlay.png")
    return data
