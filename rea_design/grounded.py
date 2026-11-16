"""Local text-to-mask inference with Grounding DINO and Segment Anything."""

import json
import math
import re
import unicodedata
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw

from .cli import flatten, load, protect_inputs
from .scene import sha


DETECTOR = "grounding-dino-tiny"
SEGMENTER = "sam-vit-base"
CHECKPOINTS = (
    ("IDEA-Research/grounding-dino-tiny", "a2bb814dd30d776dcf7e30523b00659f4f141c71", DETECTOR),
    ("facebook/sam-vit-base", "70c1a07f894ebb5b307fd9eaaee97b9dfc16068f", SEGMENTER),
)
QUERY_ALIASES = {
    "oculos": "eyeglasses",
    "oculos de sol": "sunglasses",
    "armacao dos oculos": "eyeglass frame",
    "pessoa": "person",
    "rosto": "face",
    "garrafa": "bottle",
    "chapeu": "hat",
}


def fetch_prompt_models(directory):
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ValueError("Model download dependencies missing; install rea-design[prompt]") from exc
    result = []
    for repo, revision, folder in CHECKPOINTS:
        path = snapshot_download(repo, revision=revision, local_dir=Path(directory) / folder,
                                 allow_patterns=["*.json", "vocab.txt", "model.safetensors"])
        result.append({"repo": repo, "revision": revision, "path": str(path)})
    return {"models": result}


def model_query(prompt):
    prompt = " ".join(prompt.split()).rstrip(".?! ")
    if not prompt or len(prompt) > 180:
        raise ValueError("Provide a short object description (1 to 180 characters)")
    key = "".join(c for c in unicodedata.normalize("NFKD", prompt.lower()) if not unicodedata.combining(c))
    key = re.sub(r"^(?:recorte|separe|isole|segmente|encontre|detecte)\s+(?:(?:so|apenas|somente)\s+)?", "", key)
    for article in ("os ", "as ", "o ", "a "):
        if key.startswith(article):
            key = key[len(article):]
            break
    return QUERY_ALIASES.get(key, prompt.lower())


def infer_mask(image, query, models_dir, threshold):
    try:
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, SamModel, SamProcessor
    except ImportError as exc:
        raise ValueError("Text segmentation dependencies missing; install rea-design[prompt]") from exc

    detector_path = models_dir / DETECTOR
    sam_path = models_dir / SEGMENTER
    for folder in (detector_path, sam_path):
        if not (folder / "model.safetensors").is_file():
            raise ValueError(f"Missing model.safetensors in {folder}; see the model setup in the README")

    processor = AutoProcessor.from_pretrained(detector_path, local_files_only=True, use_fast=False)
    detector = AutoModelForZeroShotObjectDetection.from_pretrained(
        detector_path, local_files_only=True, use_safetensors=True).eval()
    inputs = processor(images=image, text=f"{query}.", return_tensors="pt")
    with torch.inference_mode():
        predictions = detector(**inputs)
        result = processor.post_process_grounded_object_detection(
            predictions, inputs.input_ids, threshold=threshold, text_threshold=min(.25, threshold),
            target_sizes=[(image.height, image.width)])[0]
    if not len(result["scores"]):
        return None
    best = int(result["scores"].argmax())
    x0, y0, x1, y1 = result["boxes"][best].tolist()
    box = [max(0, x0), max(0, y0), min(image.width, x1), min(image.height, y1)]
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    score = float(result["scores"][best])
    del detector, predictions, inputs

    sam_processor = SamProcessor.from_pretrained(sam_path, local_files_only=True, use_fast=False)
    sam = SamModel.from_pretrained(sam_path, local_files_only=True, use_safetensors=True).eval()
    input_data = sam_processor(images=image, input_boxes=[[box]], return_tensors="pt")
    with torch.inference_mode():
        output = sam(**input_data, multimask_output=True)
        candidates = sam_processor.image_processor.post_process_masks(
            output.pred_masks, input_data["original_sizes"], input_data["reshaped_input_sizes"])[0][0]
    scores = output.iou_scores[0, 0]
    for index in scores.argsort(descending=True).tolist():
        candidate = candidates[index]
        count = int(candidate.count_nonzero())
        left, top = (math.floor(v) for v in box[:2])
        right, bottom = (math.ceil(v) for v in box[2:])
        if count and int(candidate[top:bottom, left:right].count_nonzero()) / count >= .65:
            mask = Image.fromarray((candidate.to(torch.uint8) * 255).numpy())
            return mask, box, score, float(scores[index])
    return None


def segment_text(source, prompt, output, *, models_dir="models", threshold=.25):
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be greater than zero and at most 1")
    query = model_query(prompt)
    image = load(source)
    source_hash = sha(source)
    output = Path(output)
    manifest = output / "grounded.json"
    overlay_path = output / "grounded-overlay.png"
    mask_path, cutout_path = output / "mask.png", output / "cutout.png"
    protect_inputs((manifest, overlay_path, mask_path, cutout_path), source)
    flat = flatten(image)
    result = infer_mask(flat, query, Path(models_dir), threshold)
    items = []
    output.mkdir(parents=True, exist_ok=True)
    overlay = flat.copy()
    if result is not None:
        model_mask, box, detection_score, mask_score = result
        if model_mask.mode != "L" or model_mask.size != image.size:
            raise ValueError("Mask model returned the wrong size or format")
        mask = ImageChops.darker(model_mask, image.getchannel("A"))
        bounds = mask.getbbox()
        if bounds is not None:
            mask.save(mask_path)
            cutout = image.copy()
            cutout.putalpha(mask)
            cutout.crop(bounds).save(cutout_path)
            x, y, right, bottom = bounds
            items.append({"id": "grounded-001", "label": prompt.strip(), "bbox": [x, y, right-x, bottom-y],
                          "box_xyxy": [round(value, 2) for value in box], "mask": mask_path.name,
                          "mask_sha256": sha(mask_path), "cutout": cutout_path.name,
                          "detection_score": round(detection_score, 4), "mask_score": round(mask_score, 4)})
            highlight = Image.blend(flat, Image.new("RGB", flat.size, "#20ed99"), .38)
            overlay.paste(highlight, mask=mask)
            ImageDraw.Draw(overlay).rectangle(box, outline="#48ffd4", width=2)
    overlay.save(overlay_path)
    data = {"width": image.width, "height": image.height, "source_sha256": source_hash,
            "prompt": prompt.strip(), "model_query": query, "status": "found" if items else "no-match",
            "detector": DETECTOR, "segmenter": SEGMENTER, "threshold": threshold, "items": items,
            "note": "Returns one visible match. Model scores and mask boundaries require visual review."}
    manifest.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return data
