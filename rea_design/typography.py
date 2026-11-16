"""Measure the dominant title block and split its visible lines before OCR."""

import hashlib
import json
import math
from pathlib import Path

from PIL import ImageDraw

from .cli import flatten, load, protect_inputs
from .vision import _deps


def title_items(items):
    eligible = [item for item in items if item["bbox"][2] >= 12 and item["bbox"][3] >= 8]
    if not eligible:
        return []
    anchor = max(eligible, key=lambda item: item["bbox"][3] * math.sqrt(item["bbox"][2]))
    group = [anchor]
    x, _, width, height = anchor["bbox"]
    peers = [item for item in eligible if item is not anchor and
             .55 * height <= item["bbox"][3] <= 1.45 * height and
             abs(item["bbox"][0] - x) <= max(8, width * .09) and item["bbox"][2] >= width * .44]
    # Work outward from the anchor and stop when the size or vertical rhythm changes.
    for direction in (-1, 1):
        current = anchor
        while True:
            _, y, _, h = current["bbox"]
            candidates = []
            for peer in peers:
                if peer in group:
                    continue
                _, py, _, ph = peer["bbox"]
                gap = y - (py + ph) if direction < 0 else py - (y + h)
                if -.18 * height <= gap <= .5 * height:
                    candidates.append((gap, peer))
            if not candidates:
                break
            _, current = min(candidates, key=lambda entry: abs(entry[0]))
            group.append(current)
    return sorted(group, key=lambda item: item["bbox"][1])


def measure_lines(image, item):
    cv2, np = _deps()
    x, y, w, h = item["bbox"]
    left, top = max(0, math.floor(x)), max(0, math.floor(y))
    right, bottom = min(image.width, math.ceil(x + w)), min(image.height, math.ceil(y + h))
    if right <= left or bottom <= top:
        return []
    crop = np.asarray(image.crop((left, top, right, bottom))).astype(np.float32)
    pixels = crop.reshape(-1, 3)
    if len(pixels) < 4:
        return []
    cv2.setRNGSeed(13)
    sample = pixels[::max(1, len(pixels) // 12000)]
    _, _, centers = cv2.kmeans(sample, 2, None,
                               (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, .5), 2,
                               cv2.KMEANS_PP_CENTERS)
    if np.linalg.norm(centers[0] - centers[1]) < 32:
        return []
    border = np.concatenate((crop[0], crop[-1], crop[:, 0], crop[:, -1]))
    background = np.median(border, axis=0)
    ink = int(np.argmax(np.linalg.norm(centers - background, axis=1)))
    axis = centers[ink] - centers[1 - ink]
    foreground = crop @ axis > (centers.mean(axis=0) @ axis)
    rows = np.flatnonzero(foreground.mean(axis=1) >= .16)
    lines = []
    for run in np.split(rows, np.flatnonzero(np.diff(rows) > 1) + 1):
        if len(run) < max(5, crop.shape[0] * .12):
            continue
        y1, y2 = int(run[0]), int(run[-1]) + 1
        columns = np.flatnonzero(foreground[y1:y2].mean(axis=0) >= .10)
        if not len(columns):
            continue
        x1, x2 = int(columns[0]), int(columns[-1]) + 1
        mask = foreground[y1:y2, x1:x2]
        rgb = np.rint(np.median(crop[y1:y2, x1:x2][mask], axis=0)).astype(int)
        lines.append(({"bbox": [left+x1, top+y1, x2-x1, y2-y1], "fill": "#%02x%02x%02x" % tuple(rgb),
                       "ink_fraction": round(float(mask.mean()), 3)}, mask))
    return lines


def detect_typography(source, ocr_json, output):
    cv2, np = _deps()
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError as exc:
        raise ValueError("OCR dependencies missing; install rea-design[vision]") from exc

    output = Path(output)
    manifest, overlay_file = output / "typography.json", output / "typography-overlay.png"
    protect_inputs((manifest, overlay_file), source, ocr_json)
    image = flatten(load(source))
    source_hash = hashlib.sha256(Path(source).read_bytes()).hexdigest()
    evidence = json.loads(Path(ocr_json).read_text(encoding="utf-8-sig"))
    if evidence.get("source_sha256") != source_hash or (evidence.get("width"), evidence.get("height")) != image.size:
        raise ValueError("OCR evidence does not match the source size or hash")
    selected = title_items(evidence["items"])
    title = []
    engine = None
    for item in selected:
        measured = measure_lines(image, item)
        if not measured:
            # Keep existing OCR editable when pixel analysis cannot measure the line.
            x, y, w, h = (round(value) for value in item["bbox"])
            title.append({"bbox": [x, y, w, h], "fill": "#222222", "text": item["text"],
                          "confidence": item["confidence"], "ink_fraction": None})
            continue
        for line, mask in measured:
            if engine is None:
                engine = RapidOCR()
            padded = np.pad(np.where(mask, 0, 255).astype(np.uint8), 12, constant_values=255)
            result, _ = engine(cv2.cvtColor(padded, cv2.COLOR_GRAY2BGR), use_det=False, use_cls=False, use_rec=True)
            text, confidence = result[0] if result else (item["text"] if len(measured) == 1 else "", 0)
            title.append({**line, "text": text, "confidence": round(float(confidence), 4)})
    items = []
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    if title:
        left = min(line["bbox"][0] for line in title)
        top = min(line["bbox"][1] for line in title)
        right = max(x+w for x, _, w, _ in (line["bbox"] for line in title))
        bottom = max(y+h for _, y, _, h in (line["bbox"] for line in title))
        coverages = [line["ink_fraction"] for line in title if line["ink_fraction"] is not None]
        heavy = coverages and (sum(coverages) / len(coverages) >= .55)
        for line in title:
            x, y, w, h = line["bbox"]
            draw.rectangle((x, y, x+w-1, y+h-1), outline="#19f6e6", width=2)
        items.append({"id": "title-001", "kind": "text-block", "bbox": [left, top, right-left, bottom-top],
                      "source_ids": [item["id"] for item in selected], "lines": title,
                      "style": "heavy display" if heavy else "unclassified", "font_family": "Arial Black" if heavy else "sans-serif",
                      "font_weight": 900 if heavy else 400})
    data = {"width": image.width, "height": image.height, "source_sha256": source_hash, "items": items,
            "note": "The main OCR title is a proposal. Glyph boxes, weight and font family are estimates; review the transcription."}
    output.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    overlay.save(overlay_file)
    return data
