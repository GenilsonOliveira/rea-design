import base64
import hashlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from xml.etree import ElementTree as ET

from PIL import Image, ImageDraw
from rea_design.scene import make_scene, export_svg

SVG = {"s": "http://www.w3.org/2000/svg"}


def embedded_image(node):
    return Image.open(io.BytesIO(base64.b64decode(node.attrib["href"].split(",", 1)[1])))


@unittest.skipUnless(importlib.util.find_spec("cv2") and importlib.util.find_spec("numpy"), "install rea-design[vision]")
class SceneTests(unittest.TestCase):
    def test_intersection_is_unknown_and_svg_has_editable_text(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image = Image.new("RGBA", (80, 80), (0, 0, 0, 0))
            draw = ImageDraw.Draw(image)
            draw.rectangle((10, 10, 50, 50), fill="blue")
            draw.line((25, 22, 25, 28), fill="white", width=2)
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
            scene_path.write_text(json.dumps(scene), encoding="utf-8-sig")
            original_mask = root / "mask.png"
            Image.new("L", image.size, 255).save(original_mask)
            result = export_svg(source, scene_path, root / "out.svg")
            self.assertEqual(result["warnings"], [])
            svg = ET.parse(root / "out.svg").getroot()
            self.assertEqual(svg.find(".//s:text", SVG).text, "Edited & verified")
            background, cutout = map(embedded_image, svg.findall(".//s:image", SVG))
            self.assertEqual(background.getchannel("A").getbbox(), None)
            self.assertEqual(cutout.getpixel((11, 11)), (0, 0, 255, 255))
            cleaned = cutout.getpixel((15, 15))
            self.assertEqual(cleaned[3], 255)
            self.assertLess(cleaned[0], 25)
            with self.assertRaisesRegex(ValueError, "overwrite an input"):
                export_svg(source, scene_path, source)
            snapshot = scene_path.parent / scene["layers"][0]["mask"]
            Image.new("L", image.size, 255).save(snapshot)
            with self.assertRaisesRegex(ValueError, "integrity check failed"):
                export_svg(source, scene_path, root / "out.svg")
            scene["source_sha256"] = "wrong"
            scene_path.write_text(json.dumps(scene))
            with self.assertRaises(ValueError):
                export_svg(source, scene_path, root / "out.svg")

    def test_full_canvas_mask_emits_transparent_background(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.png"
            Image.new("RGBA", (10, 10), (255, 0, 0, 128)).save(source)
            mask = root / "mask.png"
            Image.new("L", (10, 10), 255).save(mask)
            scene = {"width": 10, "height": 10, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                     "layers": [{"id": "object-001", "kind": "object", "mask": mask.name}], "render_order": ["object-001"]}
            scene_path = root / "scene.json"
            scene_path.write_text(json.dumps(scene), encoding="utf-8")
            output = root / "out.svg"
            result = export_svg(source, scene_path, output)
            background, cutout = map(embedded_image, ET.parse(output).findall(".//s:image", SVG))
            self.assertTrue(any("No uncovered background" in warning for warning in result["warnings"]))
            self.assertIsNone(background.getbbox())
            self.assertEqual(cutout.getpixel((0, 0)), (255, 0, 0, 128))

