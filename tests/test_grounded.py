import base64
import importlib.util
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from xml.etree import ElementTree as ET

from PIL import Image, ImageDraw

from rea_design.grounded import segment_text
from rea_design.scene import export_svg, make_scene


class GroundedTests(unittest.TestCase):
    def test_no_match_does_not_create_a_layer(self):
        with tempfile.TemporaryDirectory() as temp, patch("rea_design.grounded.infer_mask", return_value=None):
            root = Path(temp)
            source = root / "poster.png"
            Image.new("RGB", (40, 30), "white").save(source)
            evidence = segment_text(source, "unicorn", root / "detection")
            self.assertEqual(evidence["status"], "no-match")
            self.assertEqual(evidence["items"], [])
            self.assertFalse((root / "detection/mask.png").exists())

    @unittest.skipUnless(importlib.util.find_spec("cv2") and importlib.util.find_spec("numpy"), "install rea-design[vision]")
    def test_description_mask_is_preserved_in_an_editable_svg_layer(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "poster.png"
            Image.new("RGB", (40, 30), "blue").save(source)
            mask = Image.new("L", (40, 30), 0)
            ImageDraw.Draw(mask).ellipse((10, 5, 29, 17), fill=255)
            with patch("rea_design.grounded.infer_mask", return_value=(mask, [10, 5, 30, 18], .6, .9)) as model:
                segment_text(source, "recorte só os óculos de sol", root / "detection", models_dir=root)
            self.assertEqual(model.call_args.args[1:], ("sunglasses", root, .25))
            manifest = root / "detection/grounded.json"
            scene = make_scene(source, None, None, root / "scene", grounded_json=manifest)
            self.assertEqual(scene["render_order"], ["grounded-001"])
            svg = root / "layer.svg"
            export_svg(source, root / "scene/scene.json", svg)
            ns = {"s": "http://www.w3.org/2000/svg"}
            cutout = ET.parse(svg).find(".//s:g[@id='grounded-001']/s:image", ns)
            with Image.open(io.BytesIO(base64.b64decode(cutout.get("href").split(",", 1)[1]))) as pixels:
                self.assertEqual(pixels.getchannel("A").tobytes(), mask.crop(mask.getbbox()).tobytes())
            Image.new("L", (40, 30), 255).save(root / "detection/mask.png")
            with self.assertRaisesRegex(ValueError, "integrity check failed"):
                make_scene(source, None, None, root / "bad", grounded_json=manifest)
