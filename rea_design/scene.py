"""Create a reviewable layer proposal and an editable SVG approximation."""
import base64
import hashlib
import io
import json
import math
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET

from PIL import Image, ImageFont
from .cli import flatten, load, protect_inputs
from .generative import apply_repair, read_repair_jobs
from .layering import propose_groups, suggest_overlap, suggested_paint_order
from .repair import LamaRepair, expand_overlap, validate_repair_options
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


def read_mask(mask_bytes, size):
    with Image.open(io.BytesIO(mask_bytes)) as image:
        if image.size != size:
            raise ValueError(f"Mask dimensions {image.size} differ from canvas {size}")
        if "A" in image.getbands() or "transparency" in image.info:
            alpha = image.convert("RGBA").getchannel("A")
            if alpha.getextrema()[0] < 255:
                return alpha
        return image.convert("L")


def make_scene(source, semantic_json, ocr_json, output, *, geometry_json=None, typography_json=None,
               grounded_json=None, grounded_source=None, grabcut=(), manual=(), replacements=()):
    _, np = _deps()
    image = flatten(load(source))
    w, h = image.size
    source_hash = sha(source)
    if grounded_source and not grounded_json:
        raise ValueError("--grounded-source requires --grounded")
    if grounded_source and load(grounded_source).size != image.size:
        raise ValueError("Grounded source must be aligned to the scene canvas")
    grounded_source_hash = sha(grounded_source) if grounded_source else source_hash
    output = Path(output).resolve()
    scene_file = output / "scene.json"
    if not (semantic_json or ocr_json or geometry_json or typography_json or grounded_json or grabcut or manual):
        raise ValueError("Provide at least one scene input (--semantic, --ocr, --geometry, --typography, --grounded, --grabcut or --mask)")
    evidence_files = [path for path in (semantic_json, ocr_json, geometry_json, typography_json, grounded_json, *grabcut) if path is not None]

    def read_evidence(path, expected_hash=source_hash):
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        if (data.get("width"), data.get("height")) != (w, h) or data.get("source_sha256") != expected_hash:
            raise ValueError(f"Evidence does not match source size or hash: {path}; rerun the analysis command")
        return data

    def mask_from_evidence(path, relative):
        directory = Path(path).resolve().parent
        mask = (directory / relative).resolve()
        if not mask.is_relative_to(directory):
            raise ValueError("Mask must stay within its evidence directory")
        return mask

    candidates = []
    grounded_hashes = {}
    if semantic_json:
        for item in read_evidence(semantic_json)["items"]:
            path = mask_from_evidence(semantic_json, item["mask"])
            candidates.append((item["id"], item["class"], path, "semantic mask"))
    if grounded_json:
        for item in read_evidence(grounded_json, grounded_source_hash)["items"]:
            path = mask_from_evidence(grounded_json, item["mask"])
            grounded_hashes[item["id"]] = item["mask_sha256"]
            candidates.append((item["id"], item["label"], path, "text-prompt mask"))
    for index, path in enumerate(grabcut, 1):
        subject = read_evidence(path).get("subject")
        if not subject:
            raise ValueError(f"No GrabCut subject in {path}; run segment with --rect")
        candidates.append((f"grabcut-{index:03d}", "subject", mask_from_evidence(path, subject["mask"]), "GrabCut mask"))
    for index, (label, path) in enumerate(manual, 1):
        candidates.append((f"manual-{index:03d}", label, Path(path), "manual mask"))
    replacement_map = dict(replacements)
    if len(replacement_map) != len(replacements) or set(replacement_map).difference(item[0] for item in candidates):
        raise ValueError("Each --replace-mask must reference a unique existing object ID")
    inputs = (source, *([grounded_source] if grounded_source else []), *evidence_files,
              *(item[2] for item in candidates), *replacement_map.values())
    protect_inputs((scene_file,), *inputs)
    ocr_items = read_evidence(ocr_json)["items"] if ocr_json else []
    output.mkdir(parents=True, exist_ok=True)
    layers = []
    masks = []
    for ident, label, mask_path, evidence in candidates:
        if ident in replacement_map:
            mask_path = Path(replacement_map[ident])
            evidence = f"manual correction of {evidence}"
        original = mask_path.read_bytes()
        if ident in grounded_hashes and ident not in replacement_map:
            if hashlib.sha256(original).hexdigest() != grounded_hashes[ident]:
                raise ValueError(f"Mask integrity check failed for {ident}: {mask_path}")
        mask = read_mask(original, image.size)
        bounds = mask.getbbox()
        if bounds is None:
            raise ValueError(f"Mask is empty for {ident}: {mask_path}")
        left, top, right, bottom = bounds
        buffer = io.BytesIO()
        mask.save(buffer, format="PNG")
        original_mask = buffer.getvalue()
        mask_hash = hashlib.sha256(original_mask).hexdigest()
        snapshot_path = f"scene-masks/{mask_hash}.png"
        snapshot = output / snapshot_path
        snapshot.parent.mkdir(exist_ok=True)
        if not snapshot.exists() or sha(snapshot) != mask_hash:
            protect_inputs((snapshot,), *inputs)
            snapshot.write_bytes(original_mask)
        layers.append({"id": ident, "kind": "object", "label": label, "bbox": [left, top, right-left, bottom-top],
                       "mask": snapshot_path, "mask_sha256": mask_hash, "evidence": evidence, "depth": "unknown"})
        masks.append(np.asarray(mask) > 0)
    if geometry_json:
        for item in read_evidence(geometry_json)["items"]:
            if item.get("kind") != "rect":
                raise ValueError("Geometry evidence must contain rectangles")
            layers.append({**item, "evidence": "detected raster geometry", "depth": "unknown"})
            mask = np.zeros((h, w), dtype=bool)
            mask[box_slice(item["bbox"], image.size)] = True
            masks.append(mask)
    title_blocks = read_evidence(typography_json)["items"] if typography_json else []
    grouped_ids = {ident for block in title_blocks for ident in block["source_ids"]}
    for item in ocr_items:
        if item["id"] in grouped_ids:
            continue
        region_mask = text_mask(item, (w, h))
        colors = text_colors(image, region_mask)
        color = "#222222" if colors is None else "#%02x%02x%02x" % tuple(np.rint(colors[0]).astype(int))
        layers.append({"id": item["id"], "kind": "text", "text": item["text"], "bbox": item["bbox"],
                       "polygon": item["polygon"], "confidence": item["confidence"], "depth": "unknown",
                       "fill": color, "fill_evidence": "two-color estimate; verify visually"})
        masks.append(region_mask > 0)
    for block in title_blocks:
        layers.append({**block, "evidence": "OCR and pixel measurements", "depth": "unknown"})
        mask = np.zeros((h, w), dtype=bool)
        for line in block["lines"]:
            mask[box_slice(line["bbox"], image.size)] = True
        masks.append(mask)
    if len({layer["id"] for layer in layers}) != len(layers):
        raise ValueError("Evidence contains duplicate layer IDs")
    overlaps = []
    areas = [int(np.count_nonzero(mask)) for mask in masks]
    for i in range(len(layers)):
        for j in range(i+1, len(layers)):
            count = int(np.count_nonzero(masks[i] & masks[j]))
            if count:
                overlaps.append(suggest_overlap(layers[i], layers[j], areas[i], areas[j], count))
    propose_groups(layers)
    data = {"version": "0.2", "width": w, "height": h, "source_sha256": source_hash, "layers": layers,
            "overlaps": overlaps, "render_order": suggested_paint_order(layers, overlaps),
            "note": "Paint order is a proposal. Review containment and reorder layers if needed."}
    if grounded_source_hash != source_hash:
        data["grounded_source_sha256"] = grounded_source_hash
    scene_file.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return data


def data_uri(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def box_slice(bbox, size):
    x, y, width, height = bbox
    if any(type(value) is not int for value in bbox) or width <= 0 or height <= 0:
        raise ValueError(f"Raster geometry needs a positive integer bounding box: {bbox}")
    return slice(max(0, y), max(0, min(size[1], y + height))), slice(max(0, x), max(0, min(size[0], x + width)))


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


def blend_overlap(pixels, donor, mask, radius):
    cv2, np = _deps()
    kernel_size = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    expanded = cv2.dilate(mask, kernel)
    softened = cv2.GaussianBlur(expanded, (kernel_size, kernel_size), max(1, radius / 3))
    alpha = np.maximum(softened, mask).astype(np.float32)[:, :, None] / 255
    # Keep the source alpha/outline of the receiving layer; only its colors are repaired.
    result = pixels.copy()
    result[:, :, :3] = np.rint(pixels[:, :, :3] * (1 - alpha) + donor[:, :, :3] * alpha).astype(np.uint8)
    return result


def title_attributes(line, block, font_file):
    x, y, width, height = line.get("layout_bbox", line["bbox"])
    font_size = height * 1.37
    attrs = {"x": str(x), "y": str(y + height), "font-family": block.get("font_family", "sans-serif"),
             "font-weight": str(block.get("font_weight", 400)), "font-size": str(font_size),
             "fill": line["fill"], "textLength": str(width), "lengthAdjust": "spacingAndGlyphs"}
    if font_file and line["text"].strip():
        font = ImageFont.truetype(str(font_file), 256)
        mask, (offset_x, offset_y) = font.getmask2(line["text"], anchor="ls")
        ink_bounds = mask.getbbox()
        if ink_bounds:
            left, top, right, bottom = ink_bounds
            left, right = left + offset_x, right + offset_x
            top, bottom = top + offset_y, bottom + offset_y
            factor = height / (bottom - top)
            scale = width / ((right - left) * factor)
            family, style = font.getname()
            style = style.lower()
            weight = 900 if "black" in style or "heavy" in style else 700 if "bold" in style else 400
            attrs.update({"x": f"{-left * factor:.5f}", "y": "0", "font-family": family,
                          "font-weight": str(weight), "font-style": "italic" if "italic" in style or "oblique" in style else "normal",
                          "font-size": f"{256*factor:.5f}", "textLength": f"{font.getlength(line['text']) * factor:.5f}",
                          "transform": f"translate({x} {y-top*factor:.5f}) scale({scale:.7f} 1)"})
    return attrs


def read_scene_input(source, scene_path, clean_plate):
    scene = json.loads(scene_path.read_text(encoding="utf-8-sig"))
    image = load(source)
    w, h = image.size
    if (scene["width"], scene["height"]) != (w, h) or scene["source_sha256"] != sha(source):
        raise ValueError("Scene does not match source dimensions or hash")
    if clean_plate:
        image = load(clean_plate)
        if image.size != (w, h):
            raise ValueError(f"Clean plate must be aligned to the scene canvas ({w} × {h})")
    pixel_hash = sha(clean_plate or source)
    if scene.get("grounded_source_sha256", pixel_hash) != pixel_hash:
        raise ValueError("Use the same aligned photo for --clean-plate that was used to make the text-prompt masks")
    return scene, image, pixel_hash


def scene_layers(scene):
    layers = scene["layers"]
    by_id = {item["id"]: item for item in layers}
    order = scene["render_order"]
    if len(by_id) != len(layers) or len(order) != len(layers) or set(order) != set(by_id):
        raise ValueError("render_order must contain each unique layer ID exactly once")
    return by_id, order


def find_occluders(scene):
    occluders = {}
    for overlap in scene.get("overlaps", []):
        if overlap.get("ordering") in {"suggested", "confirmed"}:
            occluders.setdefault(overlap["behind"], []).append(overlap["in_front"])
    return occluders


def read_object_layer(scene, scene_path, item, pixels, outputs, warnings):
    _, np = _deps()
    path = (scene_path.parent / item["mask"]).resolve()
    protect_inputs(outputs, path)
    mask_bytes = path.read_bytes()
    expected = item.get("mask_sha256")
    if expected is None and scene.get("version") != "0.2":
        warnings.append(f"{item['id']}: legacy mask without an integrity hash; rerun scene to snapshot it")
    elif hashlib.sha256(mask_bytes).hexdigest() != expected:
        raise ValueError(f"Mask integrity check failed for {item['id']}: {path}")
    height, width = pixels.shape[:2]
    mask = np.asarray(read_mask(mask_bytes, (width, height)))
    coverage = np.where((mask > 0) & (pixels[:, :, 3] > 0), 255, 0).astype(np.uint8)
    return mask, coverage


def object_overlap(coverage, occluders, seen_masks, box):
    hidden = coverage[box].copy()
    hidden.fill(0)
    for front in occluders:
        if front in seen_masks:
            hidden |= seen_masks[front][box] & coverage[box]
    return hidden


def export_svg(source, scene_path, output, *, clean_plate=None, background_plate=None, title_font=None,
               repair_plate=None, repair_model=None, generative_repairs=(), moves=(), title_lines=None):
    cv2, np = _deps()
    validate_repair_options(repair_plate, repair_model)
    if generative_repairs and (repair_plate or repair_model):
        raise ValueError("Choose generative repair, a repair plate, or LaMa")
    scene_path = Path(scene_path).resolve()
    output = Path(output)
    protect_inputs((output,), source, scene_path,
                   *(path for path in (clean_plate, title_font, repair_model) if path))
    scene, image, pixel_hash = read_scene_input(source, scene_path, clean_plate)
    w, h = image.size
    if background_plate:
        protect_inputs((output,), background_plate)
        supplied_background = load(background_plate)
        if supplied_background.size != (w, h):
            raise ValueError(f"Background plate must be aligned to the scene canvas ({w} × {h})")
    layers = scene["layers"]
    by_id, order = scene_layers(scene)
    moves_by_id = {}
    for ident, dx, dy in moves:
        if ident not in by_id or ident in moves_by_id:
            raise ValueError("Each --move must name a unique existing layer")
        dx, dy = float(dx), float(dy)
        if not all(math.isfinite(value) for value in (dx, dy)):
            raise ValueError("Layer offsets must be finite numbers")
        moves_by_id[ident] = f"translate({dx:g} {dy:g})"
    if title_lines is not None:
        titles = [item for item in layers if item["kind"] == "text-block"]
        if len(titles) != 1 or len(title_lines) != len(titles[0]["lines"]):
            raise ValueError("Repeat --title-line once per line of the single editable title")
        for line, text in zip(titles[0]["lines"], title_lines, strict=True):
            line["text"] = text
    repair = None
    if repair_plate:
        protect_inputs((output,), repair_plate)
        donor = load(repair_plate)
        if donor.size != (w, h):
            raise ValueError(f"Repair plate must be aligned to the scene canvas ({w} × {h})")
        repair = np.asarray(donor)
    occluders = find_occluders(scene)
    pixels = np.asarray(image)
    display = flatten(image)
    warnings = []
    objects = {}
    seen_masks = {}
    repaired_overlaps = []
    repair_jobs = read_repair_jobs(generative_repairs, output)
    generated_repairs = []
    automatic = None
    removal = np.zeros((h, w), np.uint8)
    front_graphics = np.zeros_like(removal)
    for ident in reversed(order):
        item = by_id[ident]
        if item["kind"] == "rect":
            if not clean_plate and "source_bbox" in item:
                region = box_slice(item["source_bbox"], (w, h))
                front_graphics[region] = 255
                removal[region] = 255
            continue
        if item["kind"] == "text" and clean_plate:
            continue
        if item["kind"] == "text-block":
            if not clean_plate:
                for line in item["lines"]:
                    region = np.zeros_like(removal)
                    region[box_slice(line["bbox"], (w, h))] = 255
                    glyphs = glyph_mask(display, region)
                    if not glyphs.any():
                        warnings.append(f"{ident}: could not isolate title glyphs; source text may remain visible")
                    front_graphics |= glyphs
                    removal |= glyphs
            continue
        if item["kind"] == "text":
            glyphs = glyph_mask(display, text_mask(item, (w, h)))
            if not glyphs.any():
                warnings.append(f"{ident}: could not isolate glyphs; source text may remain visible")
            front_graphics |= glyphs
            removal |= glyphs
            continue
        if item["kind"] != "object":
            raise ValueError(f"Unsupported layer kind: {item['kind']}")
        mask, coverage = read_object_layer(scene, scene_path, item, pixels, (output,), warnings)
        removal |= coverage
        bounds = Image.fromarray(coverage).getbbox()
        if bounds is None:
            continue
        left, top, right, bottom = bounds
        box = np.s_[top:bottom, left:right]
        hidden = object_overlap(coverage, occluders.get(ident, ()), seen_masks, box)
        seen_masks[ident] = coverage
        occluded = coverage[box] & front_graphics[box]
        source_pixels = pixels[box]
        if hidden.any():
            donor_pixels = repair[box] if repair is not None else None
            radius = max(3, min(16, round(min(w, h) * .018)))
            generated_job = repair_jobs.pop(ident, None)
            if generated_job is not None:
                overlap_mask = np.zeros_like(removal)
                overlap_mask[box] = hidden
                job, candidate = generated_job
                generated, diagnostics = apply_repair(pixels, pixel_hash, ident, overlap_mask, job, candidate)
                source_pixels = generated[box]
                removal |= expand_overlap(overlap_mask)
                generated_repairs.append(diagnostics)
            elif repair_model:
                if automatic is None:
                    automatic = LamaRepair(repair_model)
                overlap_mask = np.zeros_like(removal)
                overlap_mask[box] = hidden
                generated, repair_mask = automatic.repair(pixels, overlap_mask)
                donor_pixels = generated[box]
                # Clear residual edge pixels where the foreground crosses the portrait outline.
                removal |= repair_mask
                radius = 4
            if donor_pixels is not None:
                source_pixels = blend_overlap(source_pixels, donor_pixels, hidden, radius)
            if donor_pixels is not None or generated_job is not None:
                repaired_overlaps.append({"layer": ident, "pixels": int(np.count_nonzero(hidden))})
            else:
                occluded |= hidden
                warnings.append(f"{ident}: used classical inpainting; an explicit repair can improve the result")
        donors = (coverage[box] > 0) & (occluded == 0)
        if occluded.any() and donors.any():
            cutout = inpaint_rgba(source_pixels, occluded | (255 - coverage[box]))
        else:
            cutout = source_pixels.copy()
            if occluded.any():
                warnings.append(f"{ident}: insufficient object pixels to remove overlapping graphics")
        cutout[:, :, 3] = np.minimum(pixels[box][:, :, 3], mask[box])
        objects[ident] = (bounds, Image.fromarray(cutout))
    if repair_jobs:
        raise ValueError("Generative repair does not match an ordered object overlap: " + ", ".join(repair_jobs))
    if repair_model and automatic is None:
        warnings.append("No ordered object overlaps to recompose with LaMa")
    # This is an approximate background; source pixels behind opaque objects are unavailable.
    if background_plate:
        background = supplied_background
    elif removal.all():
        warnings.append("No uncovered background pixels; the background layer is transparent")
        background = Image.new("RGBA", image.size)
    else:
        background = Image.fromarray(inpaint_rgba(pixels, removal))
    ns = "http://www.w3.org/2000/svg"
    ET.register_namespace("", ns)
    svg = ET.Element(f"{{{ns}}}svg", {"width": str(w), "height": str(h), "viewBox": f"0 0 {w} {h}", "version": "1.1"})
    ET.SubElement(svg, f"{{{ns}}}title").text = "Editable reconstruction proposal; verify layers and text"
    background_id = "background-supplied" if background_plate else "background-inpainted"
    ET.SubElement(svg, f"{{{ns}}}image", {"id": background_id, "width": str(w), "height": str(h), "href": data_uri(background)})
    group_runs = Counter()
    previous = None
    for ident in order:
        item = by_id[ident]
        category = item.get("group")
        if category and category != previous:
            group_runs[category] += 1
            suffix = "" if group_runs[category] == 1 else f"-{group_runs[category]}"
            bundle = ET.SubElement(svg, f"{{{ns}}}g", {"id": f"group-{category}{suffix}", "data-role": category})
        previous = category
        parent = bundle if category else svg
        group = ET.SubElement(parent, f"{{{ns}}}g", {"id": ident, "data-kind": item["kind"]})
        if ident in moves_by_id:
            group.set("transform", moves_by_id[ident])
        if "role" in item:
            group.set("data-role", item["role"])
        if item["kind"] == "object":
            if ident in objects:
                if "label" in item:
                    ET.SubElement(group, f"{{{ns}}}title").text = item["label"]
                bounds, cutout = objects[ident]
                left, top, right, bottom = bounds
                ET.SubElement(group, f"{{{ns}}}image", {"x": str(left), "y": str(top), "width": str(right-left),
                              "height": str(bottom-top), "href": data_uri(cutout)})
        elif item["kind"] == "rect":
            x, y, width, height = item["bbox"]
            opacity = item.get("opacity", 1)
            if not all(math.isfinite(value) for value in (x, y, width, height, opacity)) or width <= 0 or height <= 0 or not 0 <= opacity <= 1:
                raise ValueError(f"Invalid rectangle geometry or opacity for {ident}")
            ET.SubElement(group, f"{{{ns}}}rect", {"x": str(x), "y": str(y), "width": str(width), "height": str(height),
                       "fill": item["fill"], "fill-opacity": str(opacity)})
        elif item["kind"] == "text-block":
            for index, line in enumerate(item["lines"], 1):
                attrs = title_attributes(line, item, title_font)
                attrs["id"] = f"{ident}-line-{index:03d}"
                ET.SubElement(group, f"{{{ns}}}text", attrs).text = line["text"]
        elif item["kind"] == "text":
            # Color is a rough local estimate and font is a generic guess.
            polygon = item["polygon"]
            px, py = polygon[0]
            angle = math.degrees(math.atan2(polygon[1][1] - py, polygon[1][0] - px))
            line_height = math.hypot(polygon[3][0] - px, polygon[3][1] - py)
            line_width = math.hypot(polygon[1][0] - px, polygon[1][1] - py)
            font_size = max(1, min(line_height * .8, line_width / max(1, len(item["text"]) * .65)))
            ET.SubElement(group, f"{{{ns}}}text", {"x": str(px), "y": str(py + line_height*.82), "font-family": item.get("font_family", "sans-serif"),
                          "font-size": str(item.get("font_size", font_size)), "fill": item.get("fill", "#222222"), "textLength": str(max(1, line_width)),
                          "transform": f"rotate({angle:.3f} {px} {py})", "lengthAdjust": "spacingAndGlyphs"}).text = item["text"]
        else:
            raise ValueError(f"Unsupported layer kind: {item['kind']}")
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(svg).write(output, encoding="utf-8", xml_declaration=True)
    return {"svg": str(output), "layers": len(layers), "warnings": warnings,
            "pixel_source_sha256": pixel_hash, "repaired_overlaps": repaired_overlaps,
            "repair_plate_sha256": sha(repair_plate) if repair_plate else None,
            "automatic_repair": {"method": "lama-onnx", "model_sha256": sha(repair_model)} if automatic else None,
            "generative_repairs": generated_repairs,
            "background_plate_sha256": sha(background_plate) if background_plate else None,
            "note": "Editable text and raster cutouts over a raster background; font, depth and hidden pixels are estimates."}
