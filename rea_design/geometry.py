"""Detect long orthogonal strokes and solid blocks bounded by them."""
import hashlib
import json
from itertools import pairwise
from pathlib import Path

from PIL import ImageDraw

from .cli import flatten, load, protect_inputs
from .vision import _deps


def axis_lines(gray, *, vertical):
    cv2, np = _deps()
    oriented = gray if vertical else gray.T
    length, width = oriented.shape
    if min(length, width) < 16:
        return []
    pad = np.pad(oriented, ((0, 0), (2, 2)), mode="edge")
    before, after = pad[:, :width], pad[:, 4:]
    bright = np.minimum(oriented - before, oriented - after)
    dark = np.minimum(before - oriented, after - oriented)
    strength = np.maximum(bright, dark)
    # Opening removes short marks such as thin type and isolated texture edges.
    kernel = np.ones((max(11, round(length * .028)), 1), dtype=np.uint8)
    supported = cv2.morphologyEx((strength >= 6).astype(np.uint8), cv2.MORPH_OPEN, kernel)
    support = supported.sum(axis=0)
    positions = np.flatnonzero(support >= max(length * .34, support.max() * .54))
    groups = np.split(positions, np.flatnonzero(np.diff(positions) > 1) + 1)
    lines = []
    for group in groups:
        if not len(group):
            continue
        if len(group) > 6:
            continue
        position = int(max(group, key=lambda x: (support[x], strength[:, x][supported[:, x] > 0].sum())))
        rows = np.flatnonzero(supported[:, group].any(axis=1))
        start, stop = int(rows[0]), int(rows[-1]) + 1
        if start <= 3:
            start = 0
        if stop >= length - 3:
            stop = length
        mask = supported[:, position] > 0
        foreground = oriented[mask, position]
        baseline = (before[mask, position] + after[mask, position]) / 2
        light = np.median(foreground - baseline) >= 0
        distance = (255 - baseline) if light else baseline
        opacity = np.median(abs(foreground - baseline) / np.maximum(distance, 1))
        lines.append({"position": position, "start": start, "stop": stop,
                      "stripe": (int(group[0]), int(group[-1]) + 1), "light": light,
                      "opacity": round(float(np.clip(opacity, .05, 1)), 3),
                      "support": round(float(support[position] / (stop - start)), 3)})
    return lines


def color_blocks(rgb, vertical, horizontal):
    _, np = _deps()
    blocks = []
    height, width = rgb.shape[:2]
    for left, right in pairwise(vertical):
        x0, x1 = left["position"], right["position"]
        for top, bottom in pairwise(horizontal):
            y0, y1 = top["position"], bottom["position"]
            if x1 - x0 < max(18, width * .035) or y1 - y0 < max(18, height * .035):
                continue
            if any(line["start"] > y0 + 5 or line["stop"] < y1 - 5 for line in (left, right)):
                continue
            if any(line["start"] > x0 + 5 or line["stop"] < x1 - 5 for line in (top, bottom)):
                continue
            inner = rgb[y0+4:y1-4, x0+4:x1-4]
            color = np.median(inner[::2, ::2], axis=(0, 1)).astype(int)
            solid = float(np.mean(np.max(abs(inner - color), axis=2) <= 6))
            if solid < .88:
                continue
            strips = (rgb[y0+4:y1-4, max(0, x0-8):max(0, x0-4)], rgb[y0+4:y1-4, x1+4:x1+8],
                      rgb[max(0, y0-8):max(0, y0-4), x0+4:x1-4], rgb[y1+4:y1+8, x0+4:x1-4])
            contrasted = sum(np.median(np.max(abs(strip - color), axis=2)) >= 12 for strip in strips if strip.size)
            if contrasted < 2:
                continue
            blocks.append({"kind": "rect", "role": "color-block", "bbox": [x0, y0, x1-x0, y1-y0],
                           "source_bbox": [x0, y0, x1-x0+1, y1-y0+1],
                           "fill": "#%02x%02x%02x" % tuple(color), "solid_fraction": round(solid, 3),
                           "contrasting_sides": int(contrasted)})
    return blocks


def detect_geometry(source, output):
    cv2, np = _deps()
    output = Path(output)
    manifest, overlay_file = output / "geometry.json", output / "geometry-overlay.png"
    protect_inputs((manifest, overlay_file), source)
    image = flatten(load(source))
    rgb = np.asarray(image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    vertical, horizontal = axis_lines(gray, vertical=True), axis_lines(gray, vertical=False)
    items = color_blocks(rgb, vertical, horizontal)
    for index, item in enumerate(items, 1):
        item["id"] = f"block-{index:03d}"
    for direction, lines in (("vertical", vertical), ("horizontal", horizontal)):
        for index, line in enumerate(lines, 1):
            position, start, stop = line["position"], line["start"], line["stop"]
            low, high = line["stripe"]
            bbox = [position, start, 1, stop-start] if direction == "vertical" else [start, position, stop-start, 1]
            source_bbox = [low, start, high-low, stop-start] if direction == "vertical" else [start, low, stop-start, high-low]
            items.append({"id": f"grid-{direction}-{index:03d}", "kind": "rect", "role": "grid-line",
                          "orientation": direction, "bbox": bbox, "source_bbox": source_bbox,
                          "fill": "#ffffff" if line["light"] else "#000000", "opacity": line["opacity"], "support": line["support"]})
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    for item in items:
        x, y, w, h = item["bbox"]
        if item["role"] == "grid-line":
            draw.rectangle((x, y, x+w-1, y+h-1), outline="#22f8dc")
        else:
            draw.rectangle((x+3, y+3, x+w-4, y+h-4), outline="#f8ee44", width=2)
            draw.text((x+8, y+8), item["id"], fill="#f8ee44", stroke_width=1, stroke_fill="#111111")
    data = {"width": image.width, "height": image.height,
            "source_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(), "items": items,
            "note": "Orthogonal ridge and fully bounded solid-cell proposals; verify visually. Stroke opacity is estimated."}
    output.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    overlay.save(overlay_file)
    return data
