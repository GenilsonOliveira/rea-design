"""One-command reconstruction using the existing evidence and SVG pipelines."""

from pathlib import Path

from .cli import load, protect_inputs
from .geometry import detect_geometry
from .grounded import segment_texts
from .scene import export_svg, make_scene
from .typography import detect_typography
from .vision import ocr


def assemble(source, prompts, output, *, models_dir="models", threshold=.25, clean_plate=None,
             repair_plate=None, background_plate=None, title_font=None):
    if not 1 <= len(prompts) <= 12 or not 0 < threshold <= 1:
        raise ValueError("Provide 1 to 12 descriptions and a threshold greater than zero and at most 1")
    output = Path(output)
    reference_size = load(source).size
    plates = (clean_plate, repair_plate, background_plate)
    for plate in filter(None, plates):
        if load(plate).size != reference_size:
            raise ValueError(f"Plate {plate} must be aligned to the reference canvas {reference_size}")
    outputs = ["reconstruction.svg", "scene/scene.json", "ocr/ocr.json", "ocr/ocr-overlay.png",
               "geometry/geometry.json", "geometry/geometry-overlay.png", "typography/typography.json",
               "typography/typography-overlay.png", "grounded/grounded.json", "grounded/grounded-overlay.png"]
    outputs.extend(("grounded/mask.png", "grounded/cutout.png") if len(prompts) == 1 else
                   (f"grounded/{kind}-{index:03d}.png" for kind in ("mask", "cutout") for index in range(1, len(prompts)+1)))
    protect_inputs((output / path for path in outputs), source, *(path for path in (*plates, title_font) if path))
    photo = clean_plate or source
    ocr_path = output / "ocr/ocr.json"
    ocr(source, ocr_path.parent)
    geometry_path = output / "geometry/geometry.json"
    detect_geometry(source, geometry_path.parent)
    typography_path = output / "typography/typography.json"
    detect_typography(source, ocr_path, typography_path.parent)
    grounded_path = output / "grounded/grounded.json"
    grounded = segment_texts(photo, prompts, grounded_path.parent, models_dir=models_dir, threshold=threshold)
    scene_path = output / "scene/scene.json"
    scene = make_scene(source, None, ocr_path, scene_path.parent, geometry_json=geometry_path,
                       typography_json=typography_path, grounded_json=grounded_path, grounded_source=clean_plate)
    rendered = export_svg(source, scene_path, output / "reconstruction.svg", clean_plate=clean_plate,
                          repair_plate=repair_plate, background_plate=background_plate, title_font=title_font)
    return {**rendered, "status": grounded["status"], "requests": grounded["requests"], "scene": str(scene_path),
            "render_order": scene["render_order"],
            "order_proposals": [item for item in scene["overlaps"] if item["ordering"] == "suggested"]}
