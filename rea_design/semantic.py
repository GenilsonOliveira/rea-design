"""FCN-ResNet50 semantic segmentation using an explicitly supplied ONNX model."""
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw
from .cli import load, flatten
from .vision import _deps

CLASSES = ("background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor")


def semantic(source, output, model, min_area=0.002):
    cv2, np = _deps()
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ValueError("ONNX Runtime missing; install rea-design[vision]") from exc
    if not 0 <= min_area <= 1:
        raise ValueError("min-area must be 0–1")
    image = flatten(load(source))
    w, h = image.size
    # Keep aspect ratio; cap large inputs for bounded memory use.
    scale = min(520 / min(w, h), 900 / max(w, h))
    nw, nh = max(32, round(w * scale)), max(32, round(h * scale))
    pixels = np.asarray(image.resize((nw, nh), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    pixels = (pixels - np.array([.485, .456, .406], dtype=np.float32)) / np.array([.229, .224, .225], dtype=np.float32)
    input_data = np.transpose(pixels, (2, 0, 1))[None]
    session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    outputs = session.run(None, {session.get_inputs()[0].name: input_data})
    logits = outputs[0]
    if logits.ndim != 4 or logits.shape[0] != 1 or logits.shape[1] != len(CLASSES):
        raise ValueError(f"Expected FCN 21-class NCHW logits; got {logits.shape}; use the documented model")
    labels = np.argmax(logits[0], axis=0).astype(np.uint8)
    labels = cv2.resize(labels, (w, h), interpolation=cv2.INTER_NEAREST)
    output = Path(output)
    (output / "masks").mkdir(parents=True, exist_ok=True)
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    items = []
    for class_id in range(1, len(CLASSES)):
        count, components, stats, _ = cv2.connectedComponentsWithStats((labels == class_id).astype(np.uint8), 8)
        for component in range(1, count):
            x, y, rw, rh, area = [int(v) for v in stats[component]]
            if area < max(1, w*h*min_area):
                continue
            ident = f"object-{len(items)+1:03d}"
            mask_path = f"masks/{ident}.png"
            Image.fromarray(((components == component) * 255).astype(np.uint8)).save(output / mask_path)
            items.append({"id": ident, "class": CLASSES[class_id], "bbox": [x, y, rw, rh], "area_pixels": area, "mask": mask_path})
            draw.rectangle((x, y, x+rw-1, y+rh-1), outline="#ff00b7", width=2)
            draw.text((x, y), f"{ident} {CLASSES[class_id]}", fill="#ff00b7")
    data = {"width": w, "height": h, "source_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(),
            "model": "FCN-ResNet50-12 (ONNX Model Zoo)", "model_sha256": hashlib.sha256(Path(model).read_bytes()).hexdigest(),
            "items": items, "note": "Pixelwise semantic classes; disconnected components are not guaranteed instances. Unknown classes become background."}
    (output / "semantic.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    overlay.save(output / "semantic-overlay.png")
    return data
