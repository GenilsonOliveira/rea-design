import base64
import importlib.util
import io
import tempfile
import unittest
from pathlib import Path
from xml.etree import ElementTree as ET

from PIL import Image, ImageDraw

from rea_design.geometry import detect_geometry
from rea_design.scene import export_svg, make_scene


@unittest.skipUnless(importlib.util.find_spec("cv2") and importlib.util.find_spec("numpy"), "install rea-design[vision]")
class GeometryTests(unittest.TestCase):
    def test_detects_off_center_grid_and_solid_cell_in_svg(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "design.png"
            canvas = Image.new("RGB", (312, 226), "#202a3a")
            draw = ImageDraw.Draw(canvas)
            draw.rectangle((118, 55, 220, 136), fill="#9c274c")
            for x in (36, 118, 221, 284):
                draw.line((x, 22, x, 209), fill="#e5e5e5")
            for y in (22, 55, 137, 209):
                draw.line((36, y, 284, y), fill="#e5e5e5")
            canvas.save(source)
            proposal = detect_geometry(source, root / "detected")
            lines = [item for item in proposal["items"] if item["role"] == "grid-line"]
            blocks = [item for item in proposal["items"] if item["role"] == "color-block"]
            self.assertEqual([(item["bbox"][0]) for item in lines if item["orientation"] == "vertical"], [36, 118, 221, 284])
            self.assertEqual([(item["bbox"][1]) for item in lines if item["orientation"] == "horizontal"], [22, 55, 137, 209])
            self.assertEqual([(block["bbox"], block["fill"]) for block in blocks], [([118, 55, 103, 82], "#9c274c")])
            evidence = root / "detected/geometry.json"
            with self.assertRaisesRegex(ValueError, "size or hash"):
                other = root / "other.png"
                Image.new("RGB", canvas.size, "black").save(other)
                make_scene(other, None, None, root / "bad", geometry_json=evidence)
            make_scene(source, None, None, root / "scene", geometry_json=evidence)
            out = root / "out.svg"
            export_svg(source, root / "scene/scene.json", out)
            ns = {"s": "http://www.w3.org/2000/svg"}
            svg = ET.parse(out)
            self.assertEqual(len(svg.findall(".//s:g[@data-role='grid-line']/s:rect", ns)), 8)
            self.assertEqual(svg.find(".//s:g[@data-role='color-block']/s:rect", ns).get("fill"), "#9c274c")
            background = svg.find(".//s:image[@id='background-inpainted']", ns)
            png = base64.b64decode(background.get("href").split(",", 1)[1])
            with Image.open(io.BytesIO(png)) as image:
                pixel = image.getpixel((36, 40))
                self.assertEqual(pixel[3], 255)
                self.assertLessEqual(max(abs(actual - expected) for actual, expected in zip(pixel, (32, 42, 58))), 2)
