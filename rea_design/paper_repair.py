"""Restore paper beneath lifted ink using tone and grain from nearby original pixels."""

from .ink_overlay import dominant_rgb
from .vision import _deps


def restore_paper(pixels, filled, mask, excluded):
    """Refine an inpainted RGBA image in place. Return the number of textured pixels."""
    cv2, np = _deps()
    x, y, width, height = cv2.boundingRect(mask)
    if not width or not height:
        return 0
    pad = 48
    left, top = max(0, x - pad), max(0, y - pad)
    region = np.s_[top:y + height + pad, left:x + width + pad]
    source = pixels[region]
    rgb = source[:, :, :3].astype(np.float32)
    interior = cv2.erode((excluded[region] == 0).astype(np.uint8), np.ones((13, 13), np.uint8)) > 0
    available = interior & (source[:, :, 3] == 255)
    if np.count_nonzero(available) < 64:
        return 0
    paper_color = dominant_rgb(rgb[available])
    donor = available & (np.linalg.norm(rgb - paper_color, axis=2) <= 18)
    if np.count_nonzero(donor) < 64:
        return 0

    # Reconstruct the slow variation in the paper's tone independently of its grain.
    known = donor.astype(np.float32)
    support = cv2.GaussianBlur(known, (0, 0), 18)
    tone = cv2.GaussianBlur(rgb * known[:, :, None], (0, 0), 18) / np.maximum(support[:, :, None], 1e-5)
    stable = cv2.erode(known, np.ones((9, 9), np.uint8)) > 0
    if np.count_nonzero(stable) < 16:
        return 0
    grain = rgb - cv2.GaussianBlur(rgb, (0, 0), 1.25)
    low, high = np.quantile(grain[stable], [.035, .965], axis=0)
    grain = grain.clip(low, high)
    repaired_rgb = filled[region][:, :, :3]
    target = (mask[region] > 0) & (support >= .035) & (source[:, :, 3] == 255)
    target &= np.linalg.norm(repaired_rgb.astype(np.float32) - tone, axis=2) <= 40
    ys, xs = np.nonzero(target)
    sample, matched = sample_paper_grain(ys, xs, stable, grain)
    target[ys[~matched], xs[~matched]] = False
    ys, xs, sample = ys[matched], xs[matched], sample[matched]
    weight = cv2.distanceTransform(target.astype(np.uint8), cv2.DIST_L2, 3)
    alpha = np.minimum(weight[ys, xs, None] / 2, 1)
    blended = repaired_rgb[ys, xs] * (1 - alpha) + (tone[ys, xs] + sample) * alpha
    repaired_rgb[ys, xs] = np.rint(blended.clip(0, 255)).astype(np.uint8)
    return len(ys)


def sample_paper_grain(ys, xs, stable, grain):
    _, np = _deps()
    height, width = stable.shape
    result = np.zeros((len(xs), 3), np.float32)
    pending = np.arange(len(xs))
    # Whole-block shifts also preserve the phase of low-amplitude JPEG texture.
    offsets = sorted(((dy, dx) for dy in range(-48, 49, 8) for dx in range(-48, 49, 8) if dy or dx),
                     key=lambda offset: offset[0]**2 + offset[1]**2)
    for dy, dx in offsets:
        if not len(pending):
            break
        row, col = ys[pending] + dy, xs[pending] + dx
        inside = (row >= 0) & (row < height) & (col >= 0) & (col < width)
        row, col = row.clip(0, height - 1), col.clip(0, width - 1)
        usable = inside & stable[row, col]
        result[pending[usable]] = grain[row[usable], col[usable]]
        pending = pending[~usable]
    matched = np.ones(len(xs), dtype=bool)
    matched[pending] = False
    return result, matched
