import base64
import hashlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from xml.etree import ElementTree as ET

from PIL import Image, ImageDraw
from rea_design.generative import prepare_repair
from rea_design.scene import make_scene, export_svg
from rea_design.title_edit import title_cleanup_mask, title_ink, title_overlap_evidence

SVG = {"s": "http://www.w3.org/2000/svg"}


def embedded_image(node):
    return Image.open(io.BytesIO(base64.b64decode(node.attrib["href"].split(",", 1)[1])))


@unittest.skipUnless(importlib.util.find_spec("cv2") and importlib.util.find_spec("numpy"), "install rea-design[vision]")
class SceneTests(unittest.TestCase):
    def test_local_title_replacement_cleans_old_letters_and_keeps_foreground_pixels(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            photo = Image.new("RGBA", (110, 104), "#e2d4bf")
            draw = ImageDraw.Draw(photo)
            for left, right in ((17, 32), (36, 62), (68, 88)):
                draw.rectangle((left, 22, right, 61), fill="#4b321e")
            foreground = Image.new("L", photo.size)
            ImageDraw.Draw(foreground).rectangle((42, 49, 59, 95), fill=255)
            person_bounds = foreground.getbbox()
            photo.paste("#142844", (42, 49, 60, 96))
            source, mask = root / "source.png", root / "person.png"
            photo.save(source)
            foreground.save(mask)
            title = {"id": "title-001", "kind": "text-block", "bbox": [17, 22, 72, 40], "source_ids": [],
                     "lines": [{"bbox": [17, 22, 72, 40], "text": "OLD", "fill": "#4b321e", "ink_fraction": .72}]}
            typography = root / "typography.json"
            typography.write_text(json.dumps({"width": photo.width, "height": photo.height,
                                              "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                                              "items": [title]}), encoding="utf-8")
            out = root / "scene"
            scene = make_scene(source, None, None, out, typography_json=typography, manual=[("person", mask)])
            self.assertEqual(scene["render_order"], ["title-001", "manual-001"])
            overlap = scene["overlaps"][0]
            self.assertEqual((overlap["ordering"], overlap["behind"], overlap["in_front"]),
                             ("suggested", "title-001", "manual-001"))
            target = root / "edited.svg"
            result = export_svg(source, out / "scene.json", target, title_only=True, title_lines=["NEW"])
            self.assertEqual(result["render_order"], ["background-title-cleaned", "title-001", "manual-001"])
            svg = ET.parse(target)
            self.assertEqual(svg.find(".//s:g[@id='title-001']/s:text", SVG).text, "NEW")
            base = embedded_image(svg.find("s:image[@id='background-title-cleaned']", SVG))
            cutout = embedded_image(svg.find("s:g[@id='manual-001']/s:image", SVG))
            np.testing.assert_array_equal(np.asarray(cutout), np.asarray(photo.crop(person_bounds)))
            np.testing.assert_array_equal(np.asarray(base)[np.asarray(foreground) > 0], 0)
            removal = title_cleanup_mask(photo.convert("RGB"), title, np.asarray(foreground), []) > 0
            protected = (np.asarray(foreground) > 0) | removal
            np.testing.assert_array_equal(np.asarray(base)[~protected], np.asarray(photo)[~protected])
            self.assertEqual(base.getpixel((24, 40)), photo.getpixel((5, 5)))
            self.assertEqual(result["cleaned_title_pixels"], int(removal.sum()))

            # Dark ink that continues visibly across the subject must not reverse the layer order.
            draw.rectangle((42, 49, 59, 61), fill="#4b321e")
            region, ink = title_ink(photo.convert("RGB"), title["lines"][0])
            self.assertIsNone(title_overlap_evidence(region, ink, np.asarray(foreground)))

    def test_generative_repair_is_bounded_and_rejects_a_stale_manifest(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.png"
            photo = Image.new("RGBA", (50, 50), "blue")
            ImageDraw.Draw(photo).rectangle((18, 22, 22, 28), fill="red")
            photo.save(source)
            mask_inputs = []
            for name, bounds in (("portrait", (5, 5, 40, 40)), ("glasses", (18, 22, 22, 28))):
                path = root / f"{name}.png"
                mask = Image.new("L", photo.size)
                ImageDraw.Draw(mask).rectangle(bounds, fill=255)
                mask.save(path)
                mask_inputs.append((name, path))
            make_scene(source, None, None, root / "scene", manual=mask_inputs)
            scene_path = root / "scene/scene.json"
            job = prepare_repair(source, scene_path, root / "repair", margin=4)
            self.assertEqual(job["layer"], "manual-001")
            self.assertEqual(job["masked_pixels"], 35)
            manifest, draft = Path(job["job"]), root / "draft.png"
            # The generator changes the entire crop, including pixels that must stay blue.
            Image.new("RGB", (100, 100), "green").save(draft)
            svg_path = root / "edit.svg"
            options = {"generative_repairs": [(manifest, draft)], "moves": [("manual-002", 0, -10)]}
            result = export_svg(source, scene_path, svg_path, **options)
            self.assertEqual(result["repaired_overlaps"], [{"layer": "manual-001", "pixels": 35}])
            self.assertEqual(result["generative_repairs"][0]["layer"], "manual-001")
            svg = ET.parse(svg_path)
            portrait = embedded_image(svg.find(".//s:g[@id='manual-001']/s:image", SVG))
            self.assertEqual(portrait.getpixel((15, 20)), (0, 128, 0, 255))
            box = (5, 5, 41, 41)
            support = np.asarray(Image.open(job["mask"]).crop(box)) > 0
            np.testing.assert_array_equal(np.asarray(portrait)[~support], np.asarray(photo.crop(box))[~support])
            self.assertEqual(svg.find(".//s:g[@id='manual-002']", SVG).get("transform"), "translate(0 -10)")
            stale = json.loads(manifest.read_text(encoding="utf-8"))
            stale["overlap_sha256"] = "wrong"
            manifest.write_text(json.dumps(stale), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not match the pixel source or overlap"):
                export_svg(source, scene_path, svg_path, **options)

    def test_automatic_repair_uses_only_the_overlap_behind_moved_object(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image = Image.new("RGBA", (50, 50), "blue")
            ImageDraw.Draw(image).rectangle((18, 22, 22, 28), fill="red")
            source = root / "source.png"
            image.save(source)
            for name, box in (("portrait", (5, 5, 40, 40)), ("glasses", (18, 22, 22, 28))):
                mask = Image.new("L", image.size)
                ImageDraw.Draw(mask).rectangle(box, fill=255)
                mask.save(root / f"{name}.png")
            out = root / "scene"
            make_scene(source, None, None, out,
                       manual=[("portrait", root / "portrait.png"), ("glasses", root / "glasses.png")])
            model = root / "lama.onnx"
            model.write_bytes(b"test fixture")

            def generate(pixels, hidden):
                np.testing.assert_array_equal(hidden > 0, np.asarray(image)[:, :, 0] > 0)
                repaired = pixels.copy()
                repaired[hidden > 0, :3] = [0, 128, 0]
                return repaired, hidden

            scene_path = out / "scene.json"
            with patch("rea_design.scene.LamaRepair") as lama:
                lama.return_value.repair.side_effect = generate
                svg_path = root / "edit.svg"
                result = export_svg(source, scene_path, svg_path, repair_model=model,
                                    moves=[("manual-002", 0, -10)])
                lama.assert_called_once_with(model)
                lama.return_value.repair.assert_called_once()
            self.assertEqual(result["repaired_overlaps"], [{"layer": "manual-001", "pixels": 35}])
            self.assertEqual(result["automatic_repair"]["method"], "lama-onnx")
            self.assertIsNone(result["repair_plate_sha256"])
            svg = ET.parse(svg_path)
            portrait = svg.find(".//s:g[@id='manual-001']/s:image", SVG)
            glasses = svg.find(".//s:g[@id='manual-002']", SVG)
            self.assertEqual(embedded_image(portrait).getpixel((15, 20)), (0, 128, 0, 255))
            self.assertEqual(embedded_image(portrait).getpixel((0, 0)), (0, 0, 255, 255))
            self.assertEqual(embedded_image(glasses.find("s:image", SVG)).getpixel((2, 3)), (255, 0, 0, 255))
            self.assertEqual(glasses.get("transform"), "translate(0 -10)")

    def test_grabcut_correction_and_manual_mask_with_clean_plate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.png"
            plate = root / "plate.png"
            Image.new("RGB", (30, 30), "red").save(source)
            Image.new("RGB", (30, 30), "blue").save(plate)
            original = Image.new("L", (30, 30), 0)
            ImageDraw.Draw(original).rectangle((0, 0, 19, 19), fill=255)
            original.save(root / "grab.png")
            segments = root / "segments.json"
            segments.write_text(json.dumps({"width": 30, "height": 30,
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "subject": {"mask": "grab.png"}}))
            correction = Image.new("RGBA", (30, 30), (255, 255, 255, 0))
            ImageDraw.Draw(correction).rectangle((3, 4, 7, 9), fill=(0, 0, 0, 180))
            path = root / "correction.png"
            correction.save(path)
            with self.assertRaisesRegex(ValueError, "size or hash"):
                make_scene(plate, None, None, root / "bad", grabcut=[segments])
            scene = make_scene(source, None, None, root / "scene", grabcut=[segments],
                               manual=[("detail", path)], replacements=[("grabcut-001", path)])
            self.assertEqual([(layer["id"], layer["bbox"]) for layer in scene["layers"]],
                             [("grabcut-001", [3, 4, 5, 6]), ("manual-001", [3, 4, 5, 6])])
            self.assertEqual(scene["overlaps"][0]["intersection_pixels"], 30)
            scene["layers"].append({"id": "tile", "kind": "rect", "bbox": [20, 20, 5, 7], "fill": "#ff0000"})
            scene["render_order"].append("tile")
            scene_path = root / "scene/scene.json"
            scene_path.write_text(json.dumps(scene))
            out = root / "out.svg"
            background = root / "background.png"
            Image.new("RGB", (30, 30), "white").save(background)
            export_svg(source, scene_path, out, clean_plate=plate, background_plate=background)
            svg = ET.parse(out)
            self.assertEqual(svg.find(".//s:rect", SVG).get("width"), "5")
            bg = svg.find(".//s:image[@id='background-supplied']", SVG)
            self.assertEqual(embedded_image(bg).getpixel((3, 4)), (255, 255, 255, 255))
            layer = svg.find(".//s:g[@id='grabcut-001']/s:image", SVG)
            self.assertEqual(embedded_image(layer).getpixel((0, 0)), (0, 0, 255, 180))

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

