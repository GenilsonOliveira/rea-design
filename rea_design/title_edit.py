"""Ink masks and a localized title edit for flattened images."""

from xml.etree import ElementTree as ET

from PIL import Image, ImageColor

from .vision import _deps


def title_ink(image, line, *, margin=2):
    """Return the title region and pixels close to its measured ink color."""
    _, np = _deps()
    from .scene import box_slice

    width, height = image.size
    x, y, w, h = line["bbox"]
    box = box_slice((x - margin, y - margin, w + 2 * margin, h + 2 * margin), image.size)
    region = np.zeros((height, width), np.uint8)
    region[box] = 255
    ink = np.zeros_like(region)
    if line.get("ink_fraction") is not None:
        color = np.asarray(ImageColor.getrgb(line["fill"]), dtype=np.float32)
        pixels = np.asarray(image)[box].astype(np.float32)
        ink[box] = (np.square(pixels - color).sum(axis=2) <= 24**2).astype(np.uint8) * 255
    return region, ink


def title_overlap_evidence(region, ink, coverage):
    """Look for visible title-colored strokes that stop at an object boundary."""
    cv2, np = _deps()
    overlap = (region > 0) & (coverage > 0)
    count = int(np.count_nonzero(overlap))
    if count < max(32, .02 * np.count_nonzero(region)):
        return None
    radius = max(3, min(12, round(cv2.boundingRect(region)[3] / 22)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    adjacent = (region > 0) & (coverage == 0) & (cv2.dilate(coverage, kernel) > 0)
    surrounding = int(np.count_nonzero(adjacent))
    if surrounding < 32:
        return None
    ink_inside = float(np.count_nonzero(ink[overlap])) / count
    outside_count = int(np.count_nonzero(ink[adjacent]))
    ink_outside = outside_count / surrounding
    if outside_count < 32 or ink_outside < .35 or ink_inside > min(.18, ink_outside * .40):
        return None
    return {"ink_near_object": round(ink_outside, 3), "ink_inside_object": round(ink_inside, 3)}


def infer_title_foregrounds(image, title, coverages):
    result = {}
    for index, line in enumerate(title["lines"], 1):
        region, ink = title_ink(image, line)
        if not ink.any():
            continue
        for ident, coverage in coverages.items():
            if ident not in result:
                evidence = title_overlap_evidence(region, ink, coverage)
                if evidence is not None:
                    result[ident] = {"title_line": index, **evidence}
    return result


def fit_visible_height(line, ink, protected):
    """Tighten a measured line if its old box extends into visible non-ink."""
    _, np = _deps()
    x, y, width, height = line["bbox"]
    if "layout_bbox" in line:
        return
    rows = np.flatnonzero(np.count_nonzero(ink & ~protected, axis=1) >= max(3, width * .075))
    if not len(rows) or rows[-1] + 1 - rows[0] < .65 * height:
        return
    # Keep the estimated height where almost all of the missing edge is occluded.
    top, bottom = int(rows[0]), int(rows[-1]) + 1
    columns = slice(max(0, x), max(0, x + width))
    if top > y and protected[max(0, y):top, columns].mean() > .55:
        top = y
    if bottom < y + height and protected[bottom:y + height, columns].mean() > .55:
        bottom = y + height
    line["layout_bbox"] = [x, top, width, bottom - top]


def title_cleanup_mask(image, title, protection, warnings, *, fit_layout=False):
    """Mask visible ink and antialiasing, leaving the foreground cutouts intact."""
    cv2, np = _deps()
    from .scene import glyph_mask

    removal = np.zeros_like(protection)
    kernel = np.ones((5, 5), np.uint8)
    for index, line in enumerate(title["lines"], 1):
        margin = max(4, min(12, round(line["bbox"][3] * .055)))
        region, ink = title_ink(image, line, margin=margin)
        visible = (region > 0) & (protection == 0)
        area = int(np.count_nonzero(visible))
        if not area:
            continue
        selected = int(np.count_nonzero(ink[visible]))
        if max(3, area * .09) <= selected <= area * .86:
            glyphs = cv2.dilate(ink, kernel) & region
            if fit_layout:
                fit_visible_height(line, ink > 0, protection > 0)
        else:
            original, _ = title_ink(image, line, margin=0)
            glyphs = glyph_mask(image, original & (255 - protection))
        glyphs[~visible] = 0
        if not glyphs.any():
            warnings.append(f"{title['id']} line {index}: could not isolate visible title ink")
        removal |= glyphs
    return removal


def export_title_edit(image, pixel_hash, scene, scene_path, output, title, title_font, title_behind):
    from .cli import flatten
    from .scene import append_title_lines, data_uri, inpaint_rgba, read_object_layer, scene_layers, svg_document

    _, np = _deps()
    by_id, order = scene_layers(scene)
    pixels = np.asarray(image)
    warnings = []
    coverages = {}
    for ident in order:
        if by_id[ident]["kind"] == "object":
            _, coverage = read_object_layer(scene, scene_path, by_id[ident], pixels, (output,), warnings)
            coverages[ident] = coverage
    if len(set(title_behind)) != len(title_behind) or set(title_behind).difference(coverages):
        raise ValueError("Each --title-behind must identify a unique object layer")
    display = flatten(image)
    evidence = infer_title_foregrounds(display, title, coverages) if not title_behind else {}
    if title_behind:
        evidence = {ident: {"ordering": "explicit"} for ident in title_behind}
    else:
        excluded = set()
        for overlap in scene.get("overlaps", ()):
            if overlap.get("ordering") != "confirmed":
                continue
            if overlap["behind"] == title["id"] and overlap["in_front"] in coverages:
                evidence[overlap["in_front"]] = {"ordering": "confirmed"}
            elif overlap["in_front"] == title["id"]:
                evidence.pop(overlap["behind"], None)
                excluded.add(overlap["behind"])
        # A cap or glasses already ordered above the protected subject should stay above the title too.
        for ident in order:
            if ident not in coverages or ident in evidence or ident in excluded:
                continue
            for overlap in scene.get("overlaps", ()):
                if (overlap.get("ordering") in {"suggested", "confirmed"} and overlap["in_front"] == ident
                        and overlap["behind"] in evidence):
                    evidence[ident] = {"ordering": "inherited", "above": overlap["behind"]}
                    break

    # Partition shared pixels into the topmost cutout so the original is painted exactly once.
    protection = np.zeros(pixels.shape[:2], np.uint8)
    foreground = []
    for ident in reversed(order):
        if ident not in evidence:
            continue
        coverage = coverages[ident] & (255 - protection)
        bounds = Image.fromarray(coverage).getbbox()
        if bounds is not None:
            foreground.append((ident, coverage, bounds))
        protection |= coverages[ident]
    removal = title_cleanup_mask(display, title, protection, warnings, fit_layout=True)
    if not removal.any():
        raise ValueError("Could not isolate visible title ink. Review the title boxes and colors in the scene.")
    # Exclude the portrait from the donors, then copy back only the title cleanup.
    filled = inpaint_rgba(pixels, removal | protection)
    base = pixels.copy()
    base[removal > 0] = filled[removal > 0]
    base[protection > 0] = 0

    namespace, svg = svg_document(image.size, "Localized editable title")
    width, height = image.size
    ET.SubElement(svg, f"{namespace}image", {"id": "background-title-cleaned", "width": str(width),
                                            "height": str(height), "href": data_uri(Image.fromarray(base))})
    group = ET.SubElement(svg, f"{namespace}g", {"id": title["id"], "data-kind": "text-block"})
    append_title_lines(group, namespace, title, title_font)
    for ident, coverage, bounds in reversed(foreground):
        left, top, right, bottom = bounds
        box = np.s_[top:bottom, left:right]
        cutout = pixels[box].copy()
        cutout[coverage[box] == 0] = 0
        layer = ET.SubElement(svg, f"{namespace}g", {"id": ident, "data-kind": "object",
                                                   "data-role": "title-foreground"})
        ET.SubElement(layer, f"{namespace}title").text = by_id[ident].get("label", ident)
        ET.SubElement(layer, f"{namespace}image", {"x": str(left), "y": str(top), "width": str(right - left),
                                                  "height": str(bottom - top), "href": data_uri(Image.fromarray(cutout))})
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(svg).write(output, encoding="utf-8", xml_declaration=True)
    front_ids = [ident for ident, _, _ in reversed(foreground)]
    return {"svg": str(output), "mode": "title-only", "pixel_source_sha256": pixel_hash,
            "render_order": ["background-title-cleaned", title["id"], *front_ids],
            "title_foreground": [{"layer": ident, **evidence[ident]} for ident in front_ids],
            "cleaned_title_pixels": int(np.count_nonzero(removal)),
            "protected_foreground_pixels": int(np.count_nonzero(protection)), "warnings": warnings}
