"""Recover a visible ink layer from a flattened title and its surrounding paper."""

import math

from PIL import Image, ImageColor

from .vision import _deps


def local_background(crop, line_height):
    cv2, np = _deps()
    pixels = crop.reshape(-1, 3)
    _, groups, counts = np.unique((pixels.astype(np.uint8) // 16), axis=0,
                                  return_inverse=True, return_counts=True)
    color = np.median(pixels[groups.ravel() == counts.argmax()], axis=0)
    known = (np.linalg.norm(crop - color, axis=2) < 21).astype(np.float32)
    sigma = max(2, line_height / 3)
    support = cv2.GaussianBlur(known, (0, 0), sigma)[:, :, None]
    expected = cv2.GaussianBlur(crop * known[:, :, None], (0, 0), sigma) / np.maximum(support, 1e-5)
    return color, np.where(support >= .02, expected, color)


def lift_title_ink(image, item, title):
    """Use an OCR seed to recover connected strokes, with alpha separated from the original title."""
    cv2, np = _deps()
    x, y, width, height = item["bbox"]
    pad_x, pad_y = max(4, round(height * 1.55)), max(4, round(height * .79))
    left, top = max(0, math.floor(x) - pad_x), max(0, math.floor(y) - pad_y)
    right = min(image.width, math.ceil(x + width) + pad_x)
    bottom = min(image.height, math.ceil(y + height) + pad_y)
    bounds = (left, top, right, bottom)
    if right - left < 3 or bottom - top < 3:
        raise ValueError(f"Text layer {item['id']} has no usable ink region")
    crop = np.asarray(image.crop(bounds), dtype=np.float32)
    seed = np.zeros(crop.shape[:2], np.uint8)
    polygon = np.asarray(item["polygon"], dtype=np.float32)
    cv2.fillPoly(seed, [np.rint(polygon - [left, top]).astype(np.int32)], 1)
    paper_color, paper = local_background(crop, height)
    measured = np.asarray(ImageColor.getrgb(item["fill"]), dtype=np.float32)
    samples = crop[seed > 0]
    samples = samples[np.linalg.norm(samples - measured, axis=1) < 24]
    if len(samples) < 8:
        raise ValueError(f"Could not isolate ink for {item['id']}; review its OCR box and fill")
    contrast = np.linalg.norm(samples - paper_color, axis=1)
    ink_color = np.median(samples[contrast >= np.quantile(contrast, .35)], axis=0)

    # Compositing mixes paper and big-title ink at their edges. Remove that direction
    # before estimating how much of each pixel comes from the handwritten ink.
    axis = ink_color - paper
    crossing = np.zeros(crop.shape[:2], dtype=bool)
    for line in title["lines"]:
        x, y, w, h = line["bbox"]
        x0, y0, x1, y1 = max(left, x - 2), max(top, y - 2), min(right, x + w + 2), min(bottom, y + h + 2)
        if x0 >= x1 or y0 >= y1:
            continue
        region = np.s_[y0 - top:y1 - top, x0 - left:x1 - left]
        base_ink = np.asarray(ImageColor.getrgb(line["fill"]), dtype=np.float32)
        direction = base_ink - paper[region]
        vector = ink_color - paper[region]
        projection = (vector * direction).sum(axis=2) / np.maximum(np.square(direction).sum(axis=2), 1)
        separated = vector - projection[:, :, None] * direction
        if np.median(np.square(separated).sum(axis=2)) < 12**2:
            raise ValueError(f"Ink color for {item['id']} is too close to the title color to separate")
        axis[region] = separated
        crossing[region] = True
    power = np.square(axis).sum(axis=2)
    if np.median(power[seed > 0]) < 12**2:
        raise ValueError(f"Ink color for {item['id']} is too close to the underlying colors to separate")
    alpha = ((crop - paper) * axis).sum(axis=2) / np.maximum(power, 1)
    core = (alpha > .52) & (np.linalg.norm(crop - ink_color, axis=2) < 37)
    radius = max(1, min(5, round(height * .06)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    joined = cv2.dilate(core.astype(np.uint8), kernel)
    _, labels = cv2.connectedComponents(joined, connectivity=8)
    marks = np.unique(labels[core & (seed > 0)])
    if not len(marks):
        raise ValueError(f"No ink strokes connected to the OCR seed for {item['id']}")
    connected = core & np.isin(labels, marks)
    edge = cv2.dilate(connected.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    floor = np.where(crossing, .13, .04)
    opacity = np.where(edge, np.clip((alpha - floor) / (1 - floor), 0, 1), 0).astype(np.float32)
    # Suppress isolated compression speckles while preserving connected pen edges.
    opacity[crossing] = (.65 * opacity + .35 * cv2.medianBlur(opacity, 3))[crossing]
    rgba = np.empty((*opacity.shape, 4), np.uint8)
    rgba[:, :, :3] = np.rint(ink_color).astype(np.uint8)
    solid = (opacity > .92) & (np.linalg.norm(crop - ink_color, axis=2) < 24)
    rgba[:, :, :3][solid] = np.rint(.65 * crop[solid] + .35 * ink_color).astype(np.uint8)
    rgba[:, :, 3] = np.rint(opacity * 255).astype(np.uint8)
    result = Image.fromarray(rgba)
    x0, y0, x1, y1 = result.getbbox()
    return (left + x0, top + y0, left + x1, top + y1), result.crop((x0, y0, x1, y1))
