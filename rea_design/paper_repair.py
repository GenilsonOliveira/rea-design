"""Restore paper beneath lifted ink using tone and grain from nearby original pixels."""

from .ink_overlay import dominant_rgb
from .vision import _deps


def restore_paper(pixels, filled, mask, excluded, *, wide=False):
    """Refine an inpainted RGBA image in place. Return the number of textured pixels."""
    cv2, np = _deps()
    x, y, width, height = cv2.boundingRect(mask)
    if not width or not height:
        return 0
    pad, sigma = (96, 46) if wide else (48, 18)
    left, top = max(0, x - pad), max(0, y - pad)
    region = np.s_[top:y + height + pad, left:x + width + pad]
    source = pixels[region]
    rgb = source[:, :, :3].astype(np.float32)
    guard = 5 if wide else 13
    interior = cv2.erode((excluded[region] == 0).astype(np.uint8), np.ones((guard, guard), np.uint8)) > 0
    available = interior & (source[:, :, 3] == 255)
    if np.count_nonzero(available) < 64:
        return 0
    paper_color = dominant_rgb(rgb[available])
    donor = available & (np.linalg.norm(rgb - paper_color, axis=2) <= (32 if wide else 18))
    if np.count_nonzero(donor) < 64:
        return 0

    # Reconstruct the slow variation in the paper's tone independently of its grain.
    known = donor.astype(np.float32)
    support = cv2.GaussianBlur(known, (0, 0), sigma)
    tone = cv2.GaussianBlur(rgb * known[:, :, None], (0, 0), sigma) / np.maximum(support[:, :, None], 1e-5)
    grain_guard = 5 if wide else 9
    stable = cv2.erode(known, np.ones((grain_guard, grain_guard), np.uint8)) > 0
    if np.count_nonzero(stable) < 16:
        return 0
    grain = rgb - cv2.GaussianBlur(rgb, (0, 0), 1.25)
    if wide:
        # Take a wider band of the original paper texture without dragging nearby ink into the filter.
        weight = cv2.GaussianBlur(known, (0, 0), 4)
        local = cv2.GaussianBlur(rgb * known[:, :, None], (0, 0), 4)
        grain = rgb - local / np.maximum(weight[:, :, None], 1e-5)
        tone[support < .012] = paper_color
    low, high = np.quantile(grain[stable], [.035, .965], axis=0)
    grain = grain.clip(low, high)
    repaired_rgb = filled[region][:, :, :3]
    target = (mask[region] > 0) & (source[:, :, 3] == 255)
    if not wide:
        target &= (support >= .035) & (np.linalg.norm(repaired_rgb.astype(np.float32) - tone, axis=2) <= 40)
    ys, xs = np.nonzero(target)
    sample, matched = sample_paper_grain(ys, xs, stable, grain, reach=pad)
    if wide and not matched.all():
        remainder = np.zeros_like(target)
        remainder[ys[~matched], xs[~matched]] = True
        tiled, available_tiles = cover_with_paper_patches(remainder, stable, grain)
        sample[~matched] = tiled
        matched[~matched] = available_tiles
    target[ys[~matched], xs[~matched]] = False
    ys, xs, sample = ys[matched], xs[matched], sample[matched]
    weight = cv2.distanceTransform(target.astype(np.uint8), cv2.DIST_L2, 3)
    alpha = np.minimum(weight[ys, xs, None] / 2, 1)
    blended = repaired_rgb[ys, xs] * (1 - alpha) + (tone[ys, xs] + sample) * alpha
    repaired_rgb[ys, xs] = np.rint(blended.clip(0, 255)).astype(np.uint8)
    return len(ys)


def cover_with_paper_patches(target, stable, grain):
    """Fill large strokes beyond nearby paper with overlapping clean paper patches."""
    cv2, np = _deps()
    size = 16
    patches = cv2.erode(stable.astype(np.uint8), np.ones((size, size), np.uint8),
                        anchor=(0, 0), borderType=cv2.BORDER_CONSTANT, borderValue=0)
    donors = np.argwhere(patches > 0)
    if not len(donors):
        return np.zeros((target.sum(), 3), np.float32), np.zeros(target.sum(), dtype=bool)
    height, width = target.shape
    band = np.hanning(size + 2)[1:-1].astype(np.float32)
    feather = band[:, None] * band[None, :]
    result = np.zeros((height, width, 3), np.float32)
    weight = np.zeros((height, width), np.float32)
    # Fix the donor selection so repeated exports reconstruct identical pixels.
    rng = np.random.default_rng(83)
    step = size // 2
    for y in range(0, height, step):
        for x in range(0, width, step):
            h, w = min(size, height - y), min(size, width - x)
            target_box = np.s_[y:y+h, x:x+w]
            if not target[target_box].any():
                continue
            sy, sx = donors[rng.integers(len(donors))]
            sw = feather[:h, :w]
            result[target_box] += grain[sy:sy+h, sx:sx+w] * sw[:, :, None]
            weight[target_box] += sw
    weights = weight[target]
    return result[target] / np.maximum(weights[:, None], 1e-5), weights > 0


def sample_paper_grain(ys, xs, stable, grain, *, reach):
    _, np = _deps()
    height, width = stable.shape
    result = np.zeros((len(xs), 3), np.float32)
    pending = np.arange(len(xs))
    # Whole-block shifts also preserve the phase of low-amplitude JPEG texture.
    offsets = sorted(((dy, dx) for dy in range(-reach, reach + 1, 8) for dx in range(-reach, reach + 1, 8) if dy or dx),
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
