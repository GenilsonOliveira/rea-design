import base64
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from xml.etree import ElementTree as ET

from PIL import Image, ImageDraw

from rea_design.grounded import segment_text, segment_texts
from rea_design.scene import export_svg, make_scene, sha


class GroundedTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("cv2") and importlib.util.find_spec("numpy"), "install rea-design[vision]")
    def test_batch_behind_front_order_and_repair_when_moving_glasses(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            size = (80, 60)
            reference, photo, donor = (root / name for name in ("reference.png", "photo.png", "donor.png"))
            Image.new("RGB", size, "blue").save(reference)
            portrait = Image.new("RGB", size, "white")
            ImageDraw.Draw(portrait).rectangle((10, 5, 61, 49), fill="red")
            portrait.save(donor)
            ImageDraw.Draw(portrait).rectangle((20, 14, 34, 18), fill="black")
            portrait.save(photo)
            masks = []
            for box in ((20, 14, 34, 18), (10, 5, 61, 49)):
                mask = Image.new("L", size, 0)
                ImageDraw.Draw(mask).rectangle(box, fill=255)
                x, y, right, bottom = mask.getbbox()
                masks.append((mask, [x, y, right, bottom], .8, .95))
            with patch("rea_design.grounded.infer_masks", return_value=[masks[0], None, masks[1]]) as model:
                detection = segment_texts(photo, ["óculos de sol", "unicorn", "pessoa"], root / "seg")
            self.assertEqual(model.call_args.args[1], ["sunglasses", "unicorn", "person"])
            self.assertEqual(detection["status"], "partial")
            self.assertEqual([i["id"] for i in detection["items"]], ["grounded-001", "grounded-003"])
            self.assertEqual(detection["requests"][1]["status"], "no-match")

            meta = {"width": 80, "height": 60, "source_sha256": sha(reference)}
            fixtures = {
                "ocr": [{"id": "metadata", "text": "TYPEFACE", "bbox": [55, 28, 23, 5], "confidence": .95,
                         "polygon": [[55, 28], [78, 28], [78, 33], [55, 33]]}],
                "geometry": [{"id": "line", "kind": "rect", "role": "grid-line", "orientation": "vertical",
                              "bbox": [50, 0, 1, 60], "fill": "#ffff00"}],
                "type": [{"id": "title", "kind": "text-block", "bbox": [10, 25, 40, 20], "source_ids": [], "lines": [
                    {"bbox": [10, 25, 40, 8], "text": "GERO", "fill": "#ffffff"},
                    {"bbox": [10, 37, 40, 8], "text": "BOLD", "fill": "#ffffff"}]}],
            }
            for name, items in fixtures.items():
                (root / f"{name}.json").write_text(json.dumps({**meta, "items": items}), encoding="utf-8")
            scene = make_scene(reference, None, root / "ocr.json", root / "scene", geometry_json=root / "geometry.json",
                               typography_json=root / "type.json", grounded_json=root / "seg/grounded.json", grounded_source=photo)
            self.assertEqual(scene["render_order"][:2], ["grounded-003", "grounded-001"])
            self.assertEqual(scene["overlaps"][0]["smaller_mask_fraction"], 1.0)
            path = root / "scene/scene.json"
            with self.assertRaisesRegex(ValueError, "same aligned photo"):
                export_svg(reference, path, root / "invalid.svg")
            base = root / "base.svg"
            edited = root / "edited.svg"
            result = export_svg(reference, path, base, clean_plate=photo, repair_plate=donor)
            export_svg(reference, path, edited, clean_plate=photo, repair_plate=donor,
                       moves=[("grounded-001", 8, -4)], title_lines=["NOVA", "FORMA"])
            self.assertEqual(result["repaired_overlaps"], [{"layer": "grounded-003", "pixels": 75}])
            ns = {"s": "http://www.w3.org/2000/svg"}
            original, moved = (ET.parse(file) for file in (base, edited))
            for category in ("grade", "ficha-tecnica"):
                selector = f".//s:g[@id='group-{category}']"
                self.assertEqual(ET.tostring(original.find(selector, ns)), ET.tostring(moved.find(selector, ns)))
            self.assertEqual(moved.find(".//s:g[@id='grounded-001']", ns).get("transform"), "translate(8 -4)")
            self.assertEqual([node.text for node in moved.findall(".//s:g[@id='title']/s:text", ns)], ["NOVA", "FORMA"])
            underlay = original.find(".//s:g[@id='grounded-003']/s:image", ns)
            raw = base64.b64decode(underlay.get("href").split(",", 1)[1])
            with Image.open(io.BytesIO(raw)) as repaired:
                self.assertEqual(repaired.getpixel((12, 10)), (255, 0, 0, 255))

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
