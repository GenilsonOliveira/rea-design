"""Prepare bounded, mask-guided edits for an agent's image generator."""

import hashlib
import json
from pathlib import Path

from PIL import Image

from .cli import flatten, load, protect_inputs
from .repair import expand_overlap
from .vision import _deps


def repair_region(hidden, margin):
    cv2, _ = _deps()
    height, width = hidden.shape
    x, y, w, h = cv2.boundingRect(hidden)
    if margin is None:
        margin = max(4, min(32, round(max(w, h) * .065)))
    if type(margin) is not int or not 0 <= margin <= 128:
        raise ValueError("Repair margin must be an integer from 0 to 128 pixels")
    area = expand_overlap(hidden, margin)
    side = max(512, max(w, h) + 2 * margin, round(1.5 * max(w, h)))
    cw, ch = min(width, side), min(height, side)
    left = min(max(0, x + (w - cw) // 2), width - cw)
    top = min(max(0, y + (h - ch) // 2), height - ch)
    return (left, top, left + cw, top + ch), area, margin


def job_identity(layer, pixel_sha256, hidden, margin):
    bounds, area, margin = repair_region(hidden, margin)
    return {
        "version": 1,
        "layer": layer,
        "pixel_source_sha256": pixel_sha256,
        "crop_xyxy": list(bounds),
        "margin": margin,
        "overlap_sha256": hashlib.sha256(hidden.tobytes()).hexdigest(),
    }, area


def prepare_repair(source, scene_file, output, *, clean_plate=None, layer=None, margin=None):
    from .scene import find_occluders, object_overlap, read_object_layer, read_scene_input, scene_layers

    _, np = _deps()
    scene_path = Path(scene_file).resolve()
    scene, image, pixel_hash = read_scene_input(source, scene_path, clean_plate)
    by_id, order = scene_layers(scene)
    if layer is not None and (layer not in by_id or by_id[layer]["kind"] != "object"):
        raise ValueError("--layer must identify an object receiving the repair")
    output = Path(output)
    context, mask_file, manifest = (output / file for file in ("context.png", "mask.png", "repair.json"))
    outputs = (context, mask_file, manifest)
    protect_inputs(outputs, source, scene_path, *(path for path in (clean_plate,) if path))
    pixels = np.asarray(image)
    occluders = find_occluders(scene)
    seen_masks, candidates, warnings = {}, {}, []
    for ident in reversed(order):
        item = by_id[ident]
        if item["kind"] != "object":
            continue
        _, coverage = read_object_layer(scene, scene_path, item, pixels, outputs, warnings)
        if layer is None or ident == layer:
            hidden = object_overlap(coverage, occluders.get(ident, ()), seen_masks, np.s_[:, :])
            if hidden.any():
                candidates[ident] = hidden
        seen_masks[ident] = coverage
    if not candidates:
        raise ValueError("No ordered object overlap to prepare for repair")
    if len(candidates) != 1:
        raise ValueError("Choose the receiving --layer from: " + ", ".join(candidates))
    ident, hidden = candidates.popitem()
    job, area = job_identity(ident, pixel_hash, hidden, margin)
    bounds = tuple(job["crop_xyxy"])
    output.mkdir(parents=True, exist_ok=True)
    flatten(image).crop(bounds).save(context)
    Image.fromarray(area).crop(bounds).save(mask_file)
    manifest.write_text(json.dumps(job, indent=2) + "\n", encoding="utf-8")
    return {**job, "job": str(manifest), "context": str(context), "mask": str(mask_file),
            "masked_pixels": int(np.count_nonzero(hidden)), "warnings": warnings}


def read_repair_jobs(pairs, output):
    jobs = {}
    for manifest, candidate in pairs:
        protect_inputs((output,), manifest, candidate)
        job = json.loads(Path(manifest).read_text(encoding="utf-8-sig"))
        ident = job.get("layer") if isinstance(job, dict) else None
        if not isinstance(ident, str) or job.get("version") != 1:
            raise ValueError(f"Invalid generative repair manifest: {manifest}")
        if ident in jobs:
            raise ValueError(f"Multiple generative repairs for layer {ident}")
        jobs[ident] = (job, candidate)
    return jobs


def align_candidate(context, candidate, area):
    """Register generated pixels against the unchanged context outside the edit mask."""
    cv2, np = _deps()
    height, width = area.shape
    # The draft may be higher-resolution than the source. Estimate the transform at modest size.
    scale = min(1.0, 512 / max(height, width))
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    reference = cv2.cvtColor(cv2.resize(context, size, interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
    draft = cv2.cvtColor(cv2.resize(candidate, size, interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
    valid = 255 - cv2.resize(area, size, interpolation=cv2.INTER_NEAREST)
    valid = cv2.erode(valid, np.ones((17, 17), np.uint8))
    selected = valid > 0
    if np.count_nonzero(selected) < 128:
        return candidate, {"alignment": "resize"}
    original_mae = float(cv2.absdiff(reference, draft)[selected].mean())
    details = {"alignment": "resize", "context_mae": round(original_mae, 2)}
    if original_mae < 0.5:
        return candidate, details
    try:
        _, warp = cv2.findTransformECC(reference, draft, np.eye(2, 3, dtype=np.float32),
                                      cv2.MOTION_AFFINE, (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS,
                                                          60, 1e-5), valid, 5)
    except cv2.error:
        return candidate, details
    center = np.array([(size[0] - 1) / 2, (size[1] - 1) / 2])
    stretch = np.linalg.svd(warp[:, :2], compute_uv=False)
    if (np.linalg.det(warp[:, :2]) <= 0 or np.abs(stretch - 1).max() > .15
            or np.linalg.norm(warp[:, :2] @ center + warp[:, 2] - center) > .12 * min(size)):
        return candidate, details
    flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
    aligned = cv2.warpAffine(draft, warp, size, flags=flags, borderMode=cv2.BORDER_REFLECT)
    aligned_mae = float(cv2.absdiff(reference, aligned)[selected].mean())
    if aligned_mae >= original_mae:
        return candidate, details
    factors = np.array([width / size[0], height / size[1]])
    warp[:, :2] *= factors[:, None] / factors[None, :]
    warp[:, 2] *= factors
    corrected = cv2.warpAffine(candidate, warp, (width, height), flags=flags, borderMode=cv2.BORDER_REFLECT)
    return corrected, {"alignment": "affine", "context_mae": round(aligned_mae, 2)}


def apply_repair(pixels, pixel_hash, ident, hidden, job, candidate_file):
    from .scene import sha

    cv2, np = _deps()
    expected, area = job_identity(ident, pixel_hash, hidden, job.get("margin"))
    if job != expected:
        raise ValueError(f"Generative repair for {ident} does not match the pixel source or overlap. Prepare it again.")
    candidate = load(candidate_file)
    x, y, right, bottom = expected["crop_xyxy"]
    width, height = right - x, bottom - y
    if abs((candidate.width / candidate.height) / (width / height) - 1) > .06:
        raise ValueError(f"Keep the crop aspect ratio for generative repair of {ident}")
    if candidate.getchannel("A").getextrema() != (255, 255):
        raise ValueError(f"Generative repair of {ident} needs an opaque RGB edit of the crop")
    region = np.s_[y:bottom, x:right]
    source = pixels[region]
    context = np.asarray(flatten(Image.fromarray(source)))
    rgb = np.asarray(candidate.convert("RGB").resize((width, height), Image.Resampling.LANCZOS))
    support = area[region]
    rgb, diagnostics = align_candidate(context, rgb, support)
    # The generator can redraw the whole crop. Only the mask and its feather may enter the SVG.
    feather = cv2.distanceTransform(support, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    feather = np.minimum((feather - 1).clip(0) / max(expected["margin"] - 1, 1), 1)
    feather = feather * feather * (3 - 2 * feather)
    feather[hidden[region] > 0] = 1
    selected = support > 0
    alpha = feather[selected][:, None]
    mixed = np.rint(source[:, :, :3][selected] * (1 - alpha) + rgb[selected] * alpha).astype(np.uint8)
    result = pixels.copy()
    rgb_out = result[region][:, :, :3]
    rgb_out[selected] = mixed
    return result, {"layer": ident, "candidate_sha256": sha(candidate_file), **diagnostics}
