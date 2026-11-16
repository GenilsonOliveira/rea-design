import importlib.util
import unittest

from rea_design.paper_repair import restore_paper


@unittest.skipUnless(importlib.util.find_spec("cv2") and importlib.util.find_spec("numpy"),
                     "install rea-design[vision]")
class PaperRepairTests(unittest.TestCase):
    def test_repair_reuses_nearby_grain_without_touching_unmasked_pixels(self):
        import numpy as np

        y, x = np.indices((90, 120))
        tone = np.rint(np.array([222, 204, 178]) + (x * .02 - y * .03)[:, :, None]).astype(np.uint8)
        rng = np.random.default_rng(19)
        noise = rng.integers(-4, 5, size=(*x.shape, 1))
        paper = (tone + noise).astype(np.uint8)
        pixels = np.dstack((paper, np.full(x.shape, 255, np.uint8)))
        removal = (35 <= y) & (y < 55) & (49 <= x) & (x < 67)
        pixels[removal, :3] = [117, 79, 47]
        blocked = (14 <= y) & (y < 28) & (x > 80)
        pixels[blocked, :3] = [36, 25, 12]
        mask = removal.astype(np.uint8) * 255
        excluded = (removal | blocked).astype(np.uint8) * 255
        initial = pixels.copy()
        initial[removal, :3] = tone[removal] - 6

        restored = initial.copy()
        again = initial.copy()
        count = restore_paper(pixels, restored, mask, excluded)
        self.assertEqual(count, int(removal.sum()))
        self.assertEqual(restore_paper(pixels, again, mask, excluded), count)
        np.testing.assert_array_equal(again, restored)
        np.testing.assert_array_equal(restored[~removal], initial[~removal])
        np.testing.assert_array_equal(restored[:, :, 3], initial[:, :, 3])

        # Noise is drawn from neighboring paper, so compare tone and variance, not exact pixels.
        center = (40 <= y) & (y < 50) & (53 <= x) & (x < 63)
        residual = restored[center, :3].astype(np.float32) - tone[center]
        self.assertLess(abs(float(residual.mean())), 1.0)
        self.assertGreater(float(residual.std()), 1.0)

    def test_wide_repair_can_reuse_patches_beyond_the_local_grain_search(self):
        import numpy as np

        rng = np.random.default_rng(19)
        noise = rng.integers(-6, 7, (320, 320, 1))
        paper = (np.array([222, 221, 215]) + noise).astype(np.uint8)
        pixels = np.dstack((paper, np.full((320, 320), 255, np.uint8)))
        mask = np.zeros(paper.shape[:2], np.uint8)
        mask[42:278, 42:278] = 255
        pixels[mask > 0, :3] = [188, 23, 14]
        filled = pixels.copy()
        filled[mask > 0, :3] = [208, 159, 142]
        self.assertEqual(restore_paper(pixels, filled, mask, mask, wide=True), int(np.count_nonzero(mask)))
        center = filled[145:175, 145:175, :3].astype(np.float32)
        self.assertLess(float(np.abs(center.mean(axis=(0, 1)) - [222, 221, 215]).max()), 2)
        self.assertGreater(float(center.std(axis=(0, 1)).mean()), 1.0)
        np.testing.assert_array_equal(filled[mask == 0], pixels[mask == 0])
