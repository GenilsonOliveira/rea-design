"""Create a reviewable layer proposal and an editable SVG approximation."""
import base64
import hashlib
import io
import json
import math
import os
from pathlib import Path
from xml.etree import ElementTree as ET

from PIL import Image
from .cli import flatten, load
from .vision import _deps


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def text_mask(item, size):
    cv2, np = _deps()
    mask = np.zeros((size[1], size[0]), dtype=np.uint8)
    points = np.rint(np.asarray(item["polygon"], dtype=np.float32)).astype(np.int32)
    if points.shape != (4, 2):
        raise ValueError("OCR polygon must contain four points")
    cv2.fillPoly(mask, [points], 255)
    return mask


def estimate_text_color(image, mask):
    cv2, np = _deps()
    values = np.asarray(image)[mask > 0]
    if len(values) < 2:
        return "#222222"
    # Two approximate color clusters: the larger is assumed to be the background.
    values = values[::max(1, len(values) // 10000)].astype(np.float32)
    cv2.setRNGSeed(42)
    _, assignments, centers = cv2.kmeans(values, 2, None,
        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5), 2, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(assignments.ravel(), minlength=2)
    foreground = np.clip(np.rint(centers[int(np.argmin(counts))]), 0, 255).astype(int)
    return "#%02x%02x%02x" % tuple(foreground)


def make_scene(source, semantic_json, ocr_json, output):
    cv2, np = _deps()
    image = flatten(load(source))
    w, h = image.size
    sem_path, output = Path(semantic_json).resolve(), Path(output).resolve()
    sem = json.loads(sem_path.read_text(encoding="utf-8"))
    ocr = json.loads(Path(ocr_json).read_text(encoding="utf-8"))
    if any((data["width"], data["height"]) != (w, h) for data in (sem, ocr)):
        raise ValueError("Source, semantic and OCR dimensions differ")
    if sem.get("source_sha256") != sha(source) or ocr.get("source_sha256") != sha(source):
        raise ValueError("Evidence does not match the source hash; rerun semantic and OCR")
    output.mkdir(parents=True, exist_ok=True)
    layers = []
    masks = []
    for item in sem["items"]:
        mask_path = (sem_path.parent / item["mask"]).resolve()
        if not mask_path.is_relative_to(sem_path.parent):
            raise ValueError("Semantic mask must stay within the evidence directory")
        mask = np.asarray(Image.open(mask_path).convert("L"))
        if mask.shape != (h, w):
            raise ValueError("Semantic mask dimensions differ from the source")
        layers.append({"id": item["id"], "kind": "object", "label": item["class"], "bbox": item["bbox"],
                       "mask": os.path.relpath(mask_path, output).replace("\\", "/"), "evidence": "semantic mask", "depth": "unknown"})
        masks.append(mask > 0)
    for item in ocr["items"]:
        region_mask = text_mask(item, (w, h))
        layers.append({"id": item["id"], "kind": "text", "text": item["text"], "bbox": item["bbox"],
                       "polygon": item["polygon"], "confidence": item["confidence"], "depth": "unknown",
                       "fill": estimate_text_color(image, region_mask), "fill_evidence": "two-color estimate; verify visually"})
        masks.append(region_mask > 0)
    overlaps = []
    for i in range(len(layers)):
        for j in range(i+1, len(layers)):
            count = int(np.count_nonzero(masks[i] & masks[j]))
            if count:
                overlaps.append({"layers": [layers[i]["id"], layers[j]["id"]], "intersection_pixels": count,
                                 "ordering": "unknown", "evidence": "mask intersection; not proof of occlusion"})
    data = {"version": "0.1", "width": w, "height": h, "source_sha256": sha(source), "layers": layers,
            "overlaps": overlaps, "render_order": [layer["id"] for layer in layers],
            "note": "Proposed paint order: objects then text. It is a rendering assumption, not inferred depth; review and edit render_order."}
    (output / "scene.json").write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return data


def data_uri(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def export_svg(source, scene_path, output):
    cv2, np = _deps()
    scene_path = Path(scene_path).resolve()
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    image = flatten(load(source))
    w, h = image.size
    if (scene["width"], scene["height"]) != (w, h) or scene["source_sha256"] != sha(source):
        raise ValueError("Scene does not match source dimensions or hash")
    layers = scene["layers"]
    by_id = {item["id"]: item for item in layers}
    order = scene["render_order"]
    if len(by_id) != len(layers) or len(order) != len(layers) or set(order) != set(by_id):
        raise ValueError("render_order must contain each unique layer ID exactly once")
    text_union = np.zeros((h, w), np.uint8)
    for item in layers:
        if item["kind"] == "text":
            text_union = cv2.bitwise_or(text_union, text_mask(item, (w, h)))
    objects = {}
    removal = text_union.copy()
    order_index = {ident: index for index, ident in enumerate(order)}
    for item in layers:
        if item["kind"] != "object":
            continue
        path = (scene_path.parent / item["mask"]).resolve()
        mask = np.asarray(Image.open(path).convert("L"))
        if mask.shape != (h, w):
            raise ValueError("Mask dimensions differ from source")
        mask = np.where(mask > 0, 255, 0).astype(np.uint8)
        removal = cv2.bitwise_or(removal, mask)
        front_text = np.zeros((h, w), np.uint8)
        for text in layers:
            if text["kind"] == "text" and order_index[text["id"]] > order_index[item["id"]]:
                front_text = cv2.bitwise_or(front_text, text_mask(text, (w, h)))
        objects[item["id"]] = cv2.bitwise_and(mask, cv2.bitwise_not(front_text))
    # This is an approximate background; source pixels behind opaque objects are unavailable.
    removal = cv2.dilate(removal, np.ones((3, 3), np.uint8), iterations=1)
    rgb = np.asarray(image)
    background = Image.fromarray(cv2.inpaint(rgb, removal, 3, cv2.INPAINT_TELEA))
    ns = "http://www.w3.org/2000/svg"
    ET.register_namespace("", ns)
    svg = ET.Element(f"{{{ns}}}svg", {"width": str(w), "height": str(h), "viewBox": f"0 0 {w} {h}", "version": "1.1"})
    ET.SubElement(svg, f"{{{ns}}}title").text = "Editable reconstruction proposal; verify layers and text"
    ET.SubElement(svg, f"{{{ns}}}image", {"id": "background-inpainted", "width": str(w), "height": str(h), "href": data_uri(background)})
    for ident in order:
        item = by_id[ident]
        group = ET.SubElement(svg, f"{{{ns}}}g", {"id": ident, "data-kind": item["kind"]})
        if item["kind"] == "object":
            cutout = image.convert("RGBA")
            cutout.putalpha(Image.fromarray(objects[ident]))
            bounds = cutout.getbbox()
            if bounds:
                left, top, right, bottom = bounds
                ET.SubElement(group, f"{{{ns}}}image", {"x": str(left), "y": str(top), "width": str(right-left),
                              "height": str(bottom-top), "href": data_uri(cutout.crop(bounds))})
        elif item["kind"] == "text":
            x, y, bw, bh = item["bbox"]
            # Color is a rough local estimate and font is a generic guess.
            polygon = item["polygon"]
            px, py = polygon[0]
            angle = math.degrees(math.atan2(polygon[1][1] - py, polygon[1][0] - px))
            line_height = math.hypot(polygon[3][0] - px, polygon[3][1] - py)
            line_width = math.hypot(polygon[1][0] - px, polygon[1][1] - py)
            font_size = max(1, min(line_height * .8, line_width / max(1, len(item["text"]) * .65)))
            ET.SubElement(group, f"{{{ns}}}text", {"x": str(px), "y": str(py + line_height*.82), "font-family": "sans-serif",
                          "font-size": str(font_size), "fill": item.get("fill", "#222222"), "textLength": str(max(1, line_width)),
                          "transform": f"rotate({angle:.3f} {px} {py})", "lengthAdjust": "spacingAndGlyphs"}).text = item["text"]
        else:
            raise ValueError(f"Unsupported layer kind: {item['kind']}")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(svg).write(output, encoding="utf-8", xml_declaration=True)
    return {"svg": str(output), "layers": len(layers), "note": "Editable text and raster cutouts over an inpainted raster background; font, depth and hidden pixels are estimates."}
