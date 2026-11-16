import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from xml.etree import ElementTree as ET

from PIL import Image, ImageDraw

from rea_design.scene import export_svg, make_scene
from rea_design.typography import common_title_ink, measure_lines, read_title_rows, title_crop_boxes, title_items


@unittest.skipUnless(importlib.util.find_spec("cv2") and importlib.util.find_spec("numpy"), "install rea-design[vision]")
class TypographyTests(unittest.TestCase):
    def test_overlapping_ocr_boxes_keep_all_lines_in_reading_order(self):
        boxes = [[5, 0, 520, 294], [5, 208, 677, 311], [15, 431, 420, 262], [5, 631, 462, 268]]
        items = [{"text": text, "bbox": box} for text, box in zip(("GOOD", "THINGS", "TAKE", "TIME"), boxes)]
        selected = title_items(items)
        self.assertEqual([item["text"] for item in selected], ["GOOD", "THINGS", "TAKE", "TIME"])
        clipped = title_crop_boxes(selected)
        self.assertEqual([y + height for _, y, _, height in clipped[:-1]], [b[1] for b in clipped[1:]])
        self.assertEqual([item["bbox"] for item in selected], boxes)

    def test_reliable_full_image_read_survives_a_conflicting_crop_read(self):
        original = {"text": "PROJECT", "confidence": .9877}
        sliver = {"bbox": [0, 0, 220, 20], "text": "NOISE", "confidence": .99}
        word = {"bbox": [0, 20, 220, 80], "text": "PROIECT", "confidence": .999}
        rows = read_title_rows(original, [sliver, word])
        self.assertEqual([(line["text"], line["confidence"]) for line in rows], [("PROJECT", .9877)])
        agreeing = read_title_rows({"text": "HOWTO", "confidence": .9967},
                                  [{"text": "HOW TO", "confidence": .9695}])
        self.assertEqual(agreeing[0]["text"], "HOW TO")

    def test_shared_chromatic_ink_recovers_a_line_on_mixed_background(self):
        image = Image.new("RGB", (150, 80), "#e1d5cc")
        draw = ImageDraw.Draw(image)
        draw.rectangle((80, 0, 149, 79), fill="#161616")
        for left in (10, 45, 80, 115):
            draw.rectangle((left, 15, left + 22, 64), fill="#b3130b")
        red = [({"fill": "#b3130b", "bbox": [0, 0, 130, 50]}, None)]
        cream = [({"fill": "#e1d5cc", "bbox": [0, 0, 130, 50]}, None)]
        ink = common_title_ink([red, red, cream, cream])
        hinted = measure_lines(image, {"bbox": [0, 0, 150, 80]}, ink_hint=ink)
        self.assertEqual([line["bbox"] for line, _ in hinted], [[10, 15, 128, 50]])
        self.assertEqual(hinted[0][0]["fill"], "#b3130b")

    def test_groups_adjacent_aligned_ocr_lines_and_splits_combined_box(self):
        items = [
            {"id": "title-a", "bbox": [20, 12, 100, 28]},
            {"id": "title-b", "bbox": [20, 46, 104, 28]},
            {"id": "tech", "bbox": [130, 12, 28, 15]},
        ]
        self.assertEqual([item["id"] for item in title_items(items)], ["title-a", "title-b"])
        image = Image.new("RGB", (160, 96), "#0a1231")
        draw = ImageDraw.Draw(image)
        for y in (12, 46):
            for x in (20, 48, 76):
                draw.rectangle((x, y, x + 21, y + 26), fill="#f8e33a")
        rows = measure_lines(image, {"bbox": [12, 6, 110, 74]})
        self.assertEqual([line["bbox"] for line, _ in rows], [[20, 12, 78, 27], [20, 46, 78, 27]])
        self.assertEqual(rows[0][0]["fill"], "#f8e33a")

    def test_replacement_keeps_the_title_and_discards_its_ocr_fragment(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ref = root / "poster.png"
            Image.new("RGB", (100, 80), "black").save(ref)
            metadata = {"width": 100, "height": 80, "source_sha256": hashlib.sha256(ref.read_bytes()).hexdigest()}
            ocr_path = root / "ocr.json"
            ocr_path.write_text(json.dumps({**metadata, "items": [{"id": "text-001"}]}), encoding="utf-8")
            type_path = root / "type.json"
            type_path.write_text(json.dumps({**metadata, "items": [{"id": "title-001", "kind": "text-block",
                "bbox": [4, 8, 40, 31], "source_ids": ["text-001"], "font_weight": 900, "lines": [
                {"bbox": [4, 8, 40, 14], "text": "NEW", "fill": "#ffee55"},
                {"bbox": [4, 25, 39, 14], "text": "TITLE", "fill": "#ffee55"}]}]}), encoding="utf-8")
            scene = make_scene(ref, None, ocr_path, root / "scene", typography_json=type_path)
            self.assertEqual(scene["render_order"], ["title-001"])
            svg = root / "out.svg"
            export_svg(ref, root / "scene/scene.json", svg, clean_plate=ref)
            ns = {"s": "http://www.w3.org/2000/svg"}
            texts = ET.parse(svg).findall(".//s:g[@id='title-001']/s:text", ns)
            self.assertEqual([(t.text, t.attrib["textLength"]) for t in texts], [("NEW", "40"), ("TITLE", "39")])
