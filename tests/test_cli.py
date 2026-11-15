import json
import tempfile
import unittest
from pathlib import Path
from PIL import Image
from rea_design.cli import inspect, compare


class EvidenceTests(unittest.TestCase):
    def test_inspect_preserves_dimensions_and_builds_context(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image = Image.new("RGBA", (120, 90), (255, 0, 0, 255))
            image.putpixel((0, 0), (0, 0, 0, 0))
            image.save(root / "source.png")
            result = inspect(root / "source.png", root / "out")
            self.assertEqual((result["width"], result["height"]), (120, 90))
            self.assertEqual(result["alpha_extrema"], [0, 255])
            self.assertEqual(len(list((root / "out/crops").glob("*.png"))), 4)
            self.assertAlmostEqual(sum(c["fraction"] for c in result["palette"]), 1, places=3)
            self.assertEqual(json.loads((root / "out/evidence.json").read_text())["sha256"], result["sha256"])

    def test_compare_identical_and_mismatched_dimensions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            Image.new("RGB", (10, 10), "red").save(root / "a.png")
            Image.new("RGB", (11, 10), "red").save(root / "b.png")
            self.assertEqual(compare(root / "a.png", root / "a.png", root / "out")["mean_absolute_rgb_error"], 0)
            with self.assertRaises(ValueError):
                compare(root / "a.png", root / "b.png", root / "out")


if __name__ == "__main__":
    unittest.main()
