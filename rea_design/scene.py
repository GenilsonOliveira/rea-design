"""Create a reviewable layer proposal and an editable SVG approximation."""
import base64
import hashlib
import io
import json
import math
from pathlib import Path
from xml.etree import ElementTree as ET

from PIL import Image
from .cli import flatten, load, protect_inputs
from .vision import _deps


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_mask(item, size):
    cv2, np = _deps()
    mask = np.zeros((size[1], size[0]), dtype=np.uint8)
    points = np.rint(np.asarray(item["polygon"], dtype=np.float32)).astype(np.int32)
    if points.shape != (4, 2):
        raise ValueError("OCR polygon must contain four points")
    cv2.fillPoly(mask, [points], 255)
    return mask


def text_colors(image, mask):
    cv2, np = _deps()
    values = np.asarray(image)[mask > 0]
    if len(values) < 2:
        return None
    # Two approximate color clusters: the larger is assumed to be the background.
    values = values[::max(1, len(values) // 10000)].astype(np.float32)
    cv2.setRNGSeed(42)
    _, assignments, centers = cv2.kmeans(values, 2, None,
        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5), 2, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(assignments.ravel(), minlength=2)
    minority = int(np.argmin(counts))
    return centers[minority], centers[1 - minority]


def glyph_mask(image, region):
    cv2, np = _deps()
    colors = text_colors(image, region)
    if colors is None:
        return np.zeros_like(region)
    first, second = colors
    if np.linalg.norm(first - second) < 30:
        return np.zeros_like(region)
    # Use only pixels closer to the foreground cluster. The OCR box itself is never cut out.
    inside = region > 0
    pixels = np.asarray(image)[inside].astype(np.float32)
    closer = np.sum((pixels - first) ** 2, axis=1) < np.sum((pixels - second) ** 2, axis=1)
    kernel = np.ones((3, 3), np.uint8)
    border = (region != cv2.erode(region, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0))[inside]
    first_at_border = closer[border].mean()
    if first_at_border > .7:
        closer = ~closer
    elif first_at_border >= .3:
        return np.zeros_like(region)
    if closer.mean() > .7:
        return np.zeros_like(region)
    mask = np.zeros_like(region)
    mask[inside] = closer.astype(np.uint8) * 255
    # Include edge pixels from the letter's antialiasing within the OCR bounds.
    return cv2.dilate(mask, kernel) & region


def make_scene(source, semantic_json, ocr_json, output):
    cv2, np = _deps()
    image = flatten(load(source))
    w, h = image.size
    sem_path, output = Path(semantic_json).resolve(), Path(output).resolve()
    sem = json.loads(sem_path.read_text(encoding="utf-8-sig"))
    ocr = json.loads(Path(ocr_json).read_text(encoding="utf-8-sig"))
    scene_file = output / "scene.json"
    protect_inputs((scene_file,), source, sem_path, ocr_json)
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
        original_mask = mask_path.read_bytes()
        mask_hash = hashlib.sha256(original_mask).hexdigest()
        with Image.open(io.BytesIO(original_mask)) as decoded:
            mask = np.asarray(decoded.convert("L"))
        if mask.shape != (h, w):
            raise ValueError("Semantic mask dimensions differ from the source")
        snapshot_path = f"scene-masks/{mask_hash}.png"
        snapshot = output / snapshot_path
        protect_inputs((scene_file, snapshot), source, sem_path, ocr_json, mask_path)
        snapshot.parent.mkdir(exist_ok=True)
        if not snapshot.exists() or sha(snapshot) != mask_hash:
            snapshot.write_bytes(original_mask)
        layers.append({"id": item["id"], "kind": "object", "label": item["class"], "bbox": item["bbox"],
                       "mask": snapshot_path, "mask_sha256": mask_hash, "evidence": "semantic mask", "depth": "unknown"})
        masks.append(mask > 0)
    for item in ocr["items"]:
        region_mask = text_mask(item, (w, h))
        colors = text_colors(image, region_mask)
        color = "#222222" if colors is None else "#%02x%02x%02x" % tuple(np.rint(colors[0]).astype(int))
        layers.append({"id": item["id"], "kind": "text", "text": item["text"], "bbox": item["bbox"],
                       "polygon": item["polygon"], "confidence": item["confidence"], "depth": "unknown",
                       "fill": color, "fill_evidence": "two-color estimate; verify visually"})
        masks.append(region_mask > 0)
    overlaps = []
    for i in range(len(layers)):
        for j in range(i+1, len(layers)):
            count = int(np.count_nonzero(masks[i] & masks[j]))
            if count:
                overlaps.append({"layers": [layers[i]["id"], layers[j]["id"]], "intersection_pixels": count,
                                 "ordering": "unknown", "evidence": "mask intersection; not proof of occlusion"})
    data = {"version": "0.2", "width": w, "height": h, "source_sha256": sha(source), "layers": layers,
            "overlaps": overlaps, "render_order": [layer["id"] for layer in layers],
            "note": "Proposed paint order: objects then text. It is a rendering assumption, not inferred depth; review and edit render_order."}
    scene_file.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return data


def data_uri(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def inpaint_rgba(pixels, removal):
    cv2, np = _deps()
    if not removal.any():
        return pixels.copy()
    # Reconstruct premultiplied color to keep transparent RGB from bleeding into edges.
    alpha = pixels[:, :, 3].copy()
    premultiplied = np.rint(pixels[:, :, :3] * (alpha[:, :, None] / 255.0)).astype(np.uint8)
    filled_alpha = cv2.inpaint(alpha, removal, 3, cv2.INPAINT_NS)
    filled_rgb = cv2.inpaint(premultiplied, removal, 3, cv2.INPAINT_NS)
    result = pixels.copy()
    selected = removal > 0
    a = filled_alpha[selected, None].astype(np.float32)
    result[selected, :3] = np.rint(np.clip(filled_rgb[selected] * (255.0 / np.maximum(a, 1)), 0, 255)).astype(np.uint8)
    result[selected, 3] = filled_alpha[selected]
    return result


def export_svg(source, scene_path, output):
    cv2, np = _deps()
    scene_path = Path(scene_path).resolve()
    scene = json.loads(scene_path.read_text(encoding="utf-8-sig"))
    output = Path(output)
    protect_inputs((output,), source, scene_path)
    image = load(source)
    w, h = image.size
    if (scene["width"], scene["height"]) != (w, h) or scene["source_sha256"] != sha(source):
        raise ValueError("Scene does not match source dimensions or hash")
    layers = scene["layers"]
    by_id = {item["id"]: item for item in layers}
    order = scene["render_order"]
    if len(by_id) != len(layers) or len(order) != len(layers) or set(order) != set(by_id):
        raise ValueError("render_order must contain each unique layer ID exactly once")
    pixels = np.asarray(image)
    display = flatten(image)
    warnings = []
    objects = {}
    removal = np.zeros((h, w), np.uint8)
    front_text = np.zeros_like(removal)
    for ident in reversed(order):
        item = by_id[ident]
        if item["kind"] == "text":
            glyphs = glyph_mask(display, text_mask(item, (w, h)))
            if not glyphs.any():
                warnings.append(f"{ident}: could not isolate glyphs; source text may remain visible")
            front_text |= glyphs
            removal |= glyphs
            continue
        if item["kind"] != "object":
            raise ValueError(f"Unsupported layer kind: {item['kind']}")
        path = (scene_path.parent / item["mask"]).resolve()
        protect_inputs((output,), path)
        mask_bytes = path.read_bytes()
        expected = item.get("mask_sha256")
        if expected is None and scene.get("version") != "0.2":
            warnings.append(f"{ident}: legacy mask without an integrity hash; rerun scene to snapshot it")
        elif hashlib.sha256(mask_bytes).hexdigest() != expected:
            raise ValueError(f"Mask integrity check failed for {ident}: {path}")
        with Image.open(io.BytesIO(mask_bytes)) as decoded:
            mask = np.asarray(decoded.convert("L"))
        if mask.shape != (h, w):
            raise ValueError("Mask dimensions differ from source")
        coverage = np.where((mask > 0) & (pixels[:, :, 3] > 0), 255, 0).astype(np.uint8)
        removal |= coverage
        bounds = Image.fromarray(coverage).getbbox()
        if bounds is None:
            continue
        left, top, right, bottom = bounds
        box = np.s_[top:bottom, left:right]
        glyphs = coverage[box] & front_text[box]
        donors = (coverage[box] > 0) & (glyphs == 0)
        if glyphs.any() and donors.any():
            cutout = inpaint_rgba(pixels[box], glyphs | (255 - coverage[box]))
        else:
            cutout = pixels[box].copy()
            if glyphs.any():
                warnings.append(f"{ident}: insufficient object pixels to remove overlapping text")
        cutout[:, :, 3] = np.minimum(pixels[box][:, :, 3], mask[box])
        objects[ident] = (bounds, Image.fromarray(cutout))
    # This is an approximate background; source pixels behind opaque objects are unavailable.
    if removal.all():
        warnings.append("No uncovered background pixels; the background layer is transparent")
        background = Image.new("RGBA", image.size)
    else:
        background = Image.fromarray(inpaint_rgba(pixels, removal))
    ns = "http://www.w3.org/2000/svg"
    ET.register_namespace("", ns)
    svg = ET.Element(f"{{{ns}}}svg", {"width": str(w), "height": str(h), "viewBox": f"0 0 {w} {h}", "version": "1.1"})
    ET.SubElement(svg, f"{{{ns}}}title").text = "Editable reconstruction proposal; verify layers and text"
    ET.SubElement(svg, f"{{{ns}}}image", {"id": "background-inpainted", "width": str(w), "height": str(h), "href": data_uri(background)})
    for ident in order:
        item = by_id[ident]
        group = ET.SubElement(svg, f"{{{ns}}}g", {"id": ident, "data-kind": item["kind"]})
        if item["kind"] == "object":
            if ident in objects:
                bounds, cutout = objects[ident]
                left, top, right, bottom = bounds
                ET.SubElement(group, f"{{{ns}}}image", {"x": str(left), "y": str(top), "width": str(right-left),
                              "height": str(bottom-top), "href": data_uri(cutout)})
        elif item["kind"] == "text":
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
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(svg).write(output, encoding="utf-8", xml_declaration=True)
    return {"svg": str(output), "layers": len(layers), "warnings": warnings,
            "note": "Editable text and raster cutouts over an inpainted raster background; font, depth and hidden pixels are estimates."}
