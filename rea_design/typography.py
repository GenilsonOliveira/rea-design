"""Measure the dominant title block and split its visible lines before OCR."""

import hashlib
import json
import math
from pathlib import Path

from PIL import ImageColor, ImageDraw

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
                center_step = direction * ((py + ph / 2) - (y + h / 2))
                if center_step >= .40 * min(h, ph) and -.43 * min(h, ph) <= gap <= .5 * height:
                    candidates.append((center_step, peer))
            if not candidates:
                break
            _, current = min(candidates, key=lambda entry: entry[0])
            group.append(current)
    return sorted(group, key=lambda item: item["bbox"][1])


def title_crop_boxes(items):
    """Limit overlapping OCR envelopes at the midpoint between their edges."""
    boxes = [item["bbox"].copy() for item in items]
    for first, second in zip(boxes, boxes[1:]):
        bottom, top = first[1] + first[3], second[1]
        if top < bottom:
            boundary = round((bottom + top) / 2)
            if first[1] < boundary < top + second[3]:
                first[3] = boundary - first[1]
                second[3] -= boundary - top
                second[1] = boundary
    return boxes


def letters(text):
    return "".join(char for char in text.casefold() if char.isalnum())


def read_title_rows(item, rows):
    original, original_score = item["text"].strip(), item["confidence"]
    if len(rows) > 1 and original_score >= .88 and letters("".join(line["text"] for line in rows)) != letters(original):
        # A wide OCR envelope can catch the edge of a neighboring word or photograph.
        rows = [max(rows, key=lambda line: line["bbox"][3] * math.sqrt(line["bbox"][2]))]
    if len(rows) == 1:
        line = rows[0]
        refined, score = line["text"].strip(), line["confidence"]
        same_letters = letters(refined) == letters(original)
        more_formatting = len(refined) > len(original)
        keep_refined = refined and (
            (same_letters and (more_formatting and score >= original_score - .06
                               or len(refined) >= len(original) and score > original_score))
            or (not same_letters and original_score < .85 and score > original_score + .08)
        )
        line["text"], line["confidence"] = (refined, score) if keep_refined else (original, original_score)
    return rows


def common_title_ink(measurements):
    colors = []
    for lines in measurements:
        if lines:
            strongest, _ = max(lines, key=lambda pair: pair[0]["bbox"][3] * pair[0]["bbox"][2])
            rgb = ImageColor.getrgb(strongest["fill"])
            if max(rgb) - min(rgb) >= max(45, .35 * max(rgb)):
                colors.append(rgb)
    counts = [sum(math.dist(rgb, other) <= 38 for other in colors) for rgb in colors]
    if not counts:
        return None
    support = max(counts)
    best = colors[counts.index(support)]
    if support >= max(2, len(measurements) / 2) and not any(
        count == support and math.dist(rgb, best) > 38 for rgb, count in zip(colors, counts)
    ):
        return best
    return None


def measure_lines(image, item, *, ink_hint=None):
    cv2, np = _deps()
    x, y, w, h = item["bbox"]
    left, top = max(0, math.floor(x)), max(0, math.floor(y))
    right, bottom = min(image.width, math.ceil(x + w)), min(image.height, math.ceil(y + h))
    if right <= left or bottom <= top:
        return []
    crop = np.asarray(image.crop((left, top, right, bottom))).astype(np.float32)
    if crop.shape[0] * crop.shape[1] < 4:
        return []
    if ink_hint is None:
        pixels = crop.reshape(-1, 3)
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
    else:
        # Use the shared ink hue where a photograph or texture flips local contrast.
        hsv = cv2.cvtColor(crop.astype(np.uint8), cv2.COLOR_RGB2HSV)
        ref_hue, ref_sat, _ = cv2.cvtColor(np.array([[ink_hint]], np.uint8), cv2.COLOR_RGB2HSV)[0, 0]
        hue_delta = np.abs(hsv[:, :, 0].astype(np.int16) - int(ref_hue))
        foreground = (np.minimum(hue_delta, 180 - hue_delta) <= 11) & (hsv[:, :, 1] >= max(55, ref_sat * .58))
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
    clipped = [{**item, "bbox": box} for item, box in zip(selected, title_crop_boxes(selected), strict=True)]
    measurements = [measure_lines(image, item) for item in clipped]
    ink_hint = common_title_ink(measurements)
    for item, measured in zip(clipped, measurements, strict=True):
        box = item["bbox"]
        agrees = ink_hint and any(math.dist(ImageColor.getrgb(line["fill"]), ink_hint) <= 38 for line, _ in measured)
        if ink_hint and not agrees and item["confidence"] >= .88:
            # Require a line-sized patch before replacing local color evidence.
            alternate = measure_lines(image, item, ink_hint=ink_hint)
            if len(alternate) == 1:
                line = alternate[0][0]
                if line["bbox"][2] >= .65 * box[2] and line["bbox"][3] >= .45 * box[3]:
                    measured = alternate
        if not measured:
            # Keep existing OCR editable when pixel analysis cannot measure the line.
            x, y, w, h = (round(value) for value in box)
            title.append({"bbox": [x, y, w, h], "fill": "#222222", "text": item["text"],
                          "confidence": item["confidence"], "ink_fraction": None})
            continue
        rows = []
        for line, mask in measured:
            if engine is None:
                engine = RapidOCR()
            padded = np.pad(np.where(mask, 0, 255).astype(np.uint8), 12, constant_values=255)
            result, _ = engine(cv2.cvtColor(padded, cv2.COLOR_GRAY2BGR), use_det=False, use_cls=False, use_rec=True)
            text, confidence = result[0] if result else (item["text"] if len(measured) == 1 else "", 0)
            rows.append({**line, "text": text.strip(), "confidence": round(float(confidence), 4)})
        title.extend(read_title_rows(item, rows))
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
