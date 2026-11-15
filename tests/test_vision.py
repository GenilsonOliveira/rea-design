import tempfile
import unittest
import importlib.util
from pathlib import Path
from PIL import Image, ImageDraw

from rea_design.vision import segment


@unittest.skipUnless(importlib.util.find_spec("cv2") and importlib.util.find_spec("numpy"), "install rea-design[vision]")
class SegmentationTests(unittest.TestCase):
    def test_color_regions_and_mask_coordinates(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image = Image.new("RGB", (100, 80), "white")
            ImageDraw.Draw(image).rectangle((20, 20, 59, 49), fill="blue")
            image.save(root / "input.png")
            result = segment(root / "input.png", root / "out", colors=2, min_area=.01)
            target = next(r for r in result["regions"] if r["bbox"] == [20, 20, 40, 30])
            mask = Image.open(root / "out" / target["mask"])
            self.assertEqual(mask.getpixel((30, 30)), 255)
            self.assertEqual(mask.getpixel((0, 0)), 0)
            self.assertEqual(target["area_pixels"], 1200)

    def test_invalid_rectangle_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            Image.new("RGB", (30, 30), "white").save(root / "input.png")
            with self.assertRaises(ValueError):
                segment(root / "input.png", root / "out", rect=(0, 0, 30, 30))
