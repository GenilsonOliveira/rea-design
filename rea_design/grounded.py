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


def infer_masks(image, queries, models_dir, threshold):
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
    detections = []
    for index, query in enumerate(queries):
        inputs = processor(images=image, text=f"{query}.", return_tensors="pt")
        with torch.inference_mode():
            result = processor.post_process_grounded_object_detection(
                detector(**inputs), inputs.input_ids, threshold=threshold, text_threshold=min(.25, threshold),
                target_sizes=[(image.height, image.width)])[0]
        if not len(result["scores"]):
            continue
        best = int(result["scores"].argmax())
        x0, y0, x1, y1 = result["boxes"][best].tolist()
        box = [max(0, x0), max(0, y0), min(image.width, x1), min(image.height, y1)]
        if box[2] > box[0] and box[3] > box[1]:
            detections.append((index, box, float(result["scores"][best])))
    del detector
    results = [None] * len(queries)
    if not detections:
        return results

    sam_processor = SamProcessor.from_pretrained(sam_path, local_files_only=True, use_fast=False)
    sam = SamModel.from_pretrained(sam_path, local_files_only=True, use_safetensors=True).eval()
    input_data = sam_processor(images=image, input_boxes=[[box for _, box, _ in detections]], return_tensors="pt")
    with torch.inference_mode():
        output = sam(**input_data, multimask_output=True)
        candidates = sam_processor.image_processor.post_process_masks(
            output.pred_masks, input_data["original_sizes"], input_data["reshaped_input_sizes"])[0]
    for slot, (index, box, score) in enumerate(detections):
        scores = output.iou_scores[0, slot]
        left, top = (math.floor(v) for v in box[:2])
        right, bottom = (math.ceil(v) for v in box[2:])
        for selected in scores.argsort(descending=True).tolist():
            candidate = candidates[slot, selected]
            count = int(candidate.count_nonzero())
            if count and int(candidate[top:bottom, left:right].count_nonzero()) / count >= .65:
                mask = Image.fromarray((candidate.to(torch.uint8) * 255).numpy())
                results[index] = (mask, box, score, float(scores[selected]))
                break
    return results


def infer_mask(image, query, models_dir, threshold):
    return infer_masks(image, [query], models_dir, threshold)[0]


def segment_text(source, prompt, output, *, models_dir="models", threshold=.25):
    return segment_texts(source, [prompt], output, models_dir=models_dir, threshold=threshold)


def segment_texts(source, prompts, output, *, models_dir="models", threshold=.25):
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be greater than zero and at most 1")
    if not 1 <= len(prompts) <= 12:
        raise ValueError("Provide between 1 and 12 object descriptions")
    queries = [model_query(prompt) for prompt in prompts]
    image = load(source)
    source_hash = sha(source)
    output = Path(output)
    manifest = output / "grounded.json"
    overlay_path = output / "grounded-overlay.png"
    names = [("mask.png", "cutout.png")] if len(prompts) == 1 else [
        (f"mask-{i:03d}.png", f"cutout-{i:03d}.png") for i in range(1, len(prompts) + 1)]
    protect_inputs((manifest, overlay_path, *(output / name for pair in names for name in pair)), source)
    flat = flatten(image)
    results = ([infer_mask(flat, queries[0], Path(models_dir), threshold)] if len(prompts) == 1
               else infer_masks(flat, queries, Path(models_dir), threshold))
    items = []
    requests = []
    output.mkdir(parents=True, exist_ok=True)
    overlay = flat.copy()
    colors = ("#20ed99", "#ffbf53", "#fb77f7", "#4ec4ff")
    for index, (prompt, query, paths, result) in enumerate(zip(prompts, queries, names, results, strict=True), 1):
        ident = f"grounded-{index:03d}"
        request = {"id": ident, "prompt": prompt.strip(), "model_query": query, "status": "no-match"}
        requests.append(request)
        if result is None:
            continue
        mask_path, cutout_path = (output / name for name in paths)
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
            request["status"] = "found"
            items.append({"id": ident, "label": prompt.strip(), "bbox": [x, y, right-x, bottom-y],
                          "box_xyxy": [round(value, 2) for value in box], "mask": mask_path.name,
                          "mask_sha256": sha(mask_path), "cutout": cutout_path.name,
                          "detection_score": round(detection_score, 4), "mask_score": round(mask_score, 4)})
            color = colors[(index-1) % len(colors)]
            highlight = Image.blend(overlay, Image.new("RGB", flat.size, color), .30)
            overlay.paste(highlight, mask=mask)
            draw = ImageDraw.Draw(overlay)
            draw.rectangle(box, outline=color, width=2)
            if len(prompts) > 1:
                draw.text((box[0]+3, box[1]+3), ident, fill=color, stroke_width=1, stroke_fill="black")
    overlay.save(overlay_path)
    data = {"width": image.width, "height": image.height, "source_sha256": source_hash,
            "requests": requests, "status": "found" if len(items) == len(prompts) else "partial" if items else "no-match",
            "detector": DETECTOR, "segmenter": SEGMENTER, "threshold": threshold, "items": items,
            "note": "One best visible match per description. Model scores and masks require visual review."}
    if len(prompts) == 1:
        data.update(prompt=prompts[0].strip(), model_query=queries[0])
    manifest.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return data
