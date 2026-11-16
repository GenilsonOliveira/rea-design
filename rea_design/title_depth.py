"""Plan mixed title depth and recover underlying photographic strips."""

from PIL import ImageColor

from .paper_repair import restore_paper
from .vision import _deps


def title_runs(lines, foreground, requested):
    overrides = {}
    for index, ident in requested:
        index = int(index)
        if not 1 <= index <= len(lines) or ident not in foreground:
            raise ValueError("--title-line-over needs a title line number and a selected foreground object")
        chosen = overrides.setdefault(index, set())
        if ident in chosen:
            raise ValueError("Repeated --title-line-over rule")
        chosen.add(ident)

    runs = [[] for _ in range(len(foreground) + 1)]
    for index in range(1, len(lines) + 1):
        below = overrides.get(index, set())
        slot = len(below)
        if set(foreground[:slot]) != below:
            raise ValueError("The objects below a title line must follow the foreground paint order")
        runs[slot].append(index)
    return runs


def supported_print_coverage(display, lines, mask, expanded, blocked, radius):
    from .title_edit import title_ink

    cv2, np = _deps()
    chroma = np.ptp(np.asarray(display), axis=2)
    near_radius = max(4, round(radius * .65))
    footprint = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * near_radius + 1, 2 * near_radius + 1))
    near = cv2.dilate(mask, footprint) > 0
    selected = np.zeros_like(mask)
    for line in lines:
        color_span = np.ptp(ImageColor.getrgb(line["fill"]))
        if color_span < 48:
            continue
        _, ink = title_ink(display, line, margin=12, chromatic=True)
        ink[:max(0, line["bbox"][1])] = 0
        # Use saturated cores so degraded red halos cannot join adjacent letters.
        strong = (ink > 0) & (chroma >= max(36, .30 * color_span)) & (blocked == 0)
        cores = cv2.erode(strong.astype(np.uint8), np.ones((3, 3), np.uint8))
        _, labels, stats, _ = cv2.connectedComponentsWithStats(cores, connectivity=8)
        areas = stats[:, cv2.CC_STAT_AREA]
        inside = np.bincount(labels[expanded > 0], minlength=len(areas))
        around = np.bincount(labels[near], minlength=len(areas))
        candidates = (areas >= 96) & (inside >= .20 * areas) & (around >= .60 * areas)
        candidates[0] = False
        selected[candidates[labels]] = 255
    return cv2.dilate(selected, np.ones((11, 11), np.uint8)) & (255 - blocked)


def recover_photo_ink(pixels, display, title, ids, coverages, runs):
    """Extend visible object masks across old ink only where a line is explicitly above them."""
    from .scene import inpaint_rgba
    from .title_edit import title_cleanup_mask

    cv2, np = _deps()
    corrected = {}
    photo_pixels = 0
    for position, ident in enumerate(ids, 1):
        lines = [title["lines"][index - 1] for run in runs[position:] for index in run]
        if not lines:
            continue
        mask = coverages[ident]
        blocked = np.zeros_like(mask)
        for upper in ids[position:]:
            blocked |= coverages[upper]
        ink = title_cleanup_mask(display, {**title, "lines": lines}, blocked, [], chromatic=True)
        radius = max(6, min(64, round(max(line["bbox"][3] for line in lines) * .29)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        expanded = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        # Closing bridges the painted notches in a visible silhouette. Clip the extension to old ink.
        printed = supported_print_coverage(display, lines, mask, expanded, blocked, radius)
        coverage = mask | ((expanded | printed) & ink)
        repair = coverage & ink
        if not repair.any():
            continue
        x, y, w, h = cv2.boundingRect(repair)
        pad = 72
        region = np.s_[max(0, y - pad):y + h + pad, max(0, x - pad):x + w + pad]
        excluded = repair[region] | (255 - coverage[region]) | blocked[region]
        old = pixels[region]
        generated = inpaint_rgba(old, excluded)
        restore_paper(old, generated, repair[region], excluded)
        adjusted = pixels.copy()
        selected = repair[region] > 0
        adjusted[region][selected] = generated[selected]
        corrected[ident] = adjusted
        coverages[ident] = coverage
        photo_pixels += int(np.count_nonzero(repair))
    return corrected, photo_pixels
