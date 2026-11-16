import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from xml.etree import ElementTree as ET

from PIL import Image, ImageDraw
from rea_design.scene import make_scene, export_svg


@unittest.skipUnless(importlib.util.find_spec("cv2") and importlib.util.find_spec("numpy"), "install rea-design[vision]")
class SceneTests(unittest.TestCase):
    def test_intersection_is_unknown_and_svg_has_editable_text(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image = Image.new("RGB", (80, 80), "white")
            ImageDraw.Draw(image).rectangle((10, 10, 50, 50), fill="blue")
            source = root / "source.png"
            image.save(source)
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            mask = Image.new("L", image.size, 0)
            ImageDraw.Draw(mask).rectangle((10, 10, 50, 50), fill=255)
            mask.save(root / "mask.png")
            semantic = {"width": 80, "height": 80, "source_sha256": digest, "items": [
                {"id": "object-001", "class": "person", "bbox": [10, 10, 41, 41], "mask": "mask.png"}]}
            ocr = {"width": 80, "height": 80, "source_sha256": digest, "items": [
                {"id": "text-001", "text": "A & B", "bbox": [20, 20, 20, 10], "confidence": .9,
                 "polygon": [[20, 20], [40, 20], [40, 30], [20, 30]]}]}
            (root / "semantic.json").write_text(json.dumps(semantic))
            (root / "ocr.json").write_text(json.dumps(ocr))
            scene = make_scene(source, root / "semantic.json", root / "ocr.json", root / "scene")
            self.assertGreater(scene["overlaps"][0]["intersection_pixels"], 0)
            self.assertEqual(scene["overlaps"][0]["ordering"], "unknown")
            scene["layers"][1]["text"] = "Edited & verified"
            scene_path = root / "scene/scene.json"
            scene_path.write_text(json.dumps(scene), encoding="utf-8")
            export_svg(source, scene_path, root / "out.svg")
            svg = ET.parse(root / "out.svg").getroot()
            ns = {"s": "http://www.w3.org/2000/svg"}
            self.assertEqual(svg.find(".//s:text", ns).text, "Edited & verified")
            self.assertEqual(len(svg.findall(".//s:image", ns)), 2)
            scene["source_sha256"] = "wrong"
            scene_path.write_text(json.dumps(scene))
            with self.assertRaises(ValueError):
                export_svg(source, scene_path, root / "out.svg")

