"""Local LaMa inpainting for areas hidden by a foreground layer."""

from pathlib import Path

from .vision import _deps

MODEL_SIZE = 512


def validate_repair_options(repair_plate, repair_model):
    if repair_plate and repair_model:
        raise ValueError("Choose either --repair-plate or --repair-model")
    if repair_model and not Path(repair_model).is_file():
        raise ValueError(f"LaMa ONNX model not found: {repair_model}")


class LamaRepair:
    def __init__(self, model):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ValueError("ONNX Runtime missing; install rea-design[vision]") from exc
        try:
            options = ort.SessionOptions()
            options.intra_op_num_threads = 4
            self.session = ort.InferenceSession(str(model), options, providers=["CPUExecutionProvider"])
        except Exception as exc:
            raise ValueError(f"Could not open LaMa ONNX model: {exc}") from exc
        inputs = {item.name: item for item in self.session.get_inputs()}
        expected = {"image": (1, 3, MODEL_SIZE, MODEL_SIZE), "mask": (1, 1, MODEL_SIZE, MODEL_SIZE)}
        if inputs.keys() != expected.keys() or any(
            item.type != "tensor(float)" or len(item.shape) != 4
            or any(isinstance(dim, int) and dim != size for dim, size in zip(item.shape, expected[name]))
            for name, item in inputs.items()
        ):
            raise ValueError("Expected LaMa ONNX with float image and mask inputs at 512x512")

    def repair(self, pixels, hidden):
        cv2, np = _deps()
        height, width = hidden.shape
        x, y, w, h = cv2.boundingRect(hidden)
        radius = max(2, round(max(w, h) * .012))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        expanded = cv2.dilate(hidden, kernel)
        x, y, w, h = cv2.boundingRect(expanded)
        side = min(max(width, height), max(MODEL_SIZE, round(1.45 * max(w, h))))
        left = min(max(0, x + (w - side) // 2), max(0, width - side))
        top = min(max(0, y + (h - side) // 2), max(0, height - side))
        region = np.s_[top:top + side, left:left + side]
        cropped = pixels[region]
        selected = expanded[region] > 0
        # Transparent pixels supply no usable RGB context.
        unknown = (selected | (cropped[:, :, 3] == 0)).astype(np.float32)
        if unknown.all():
            raise ValueError("LaMa cannot recompose an overlap without visible context")
        crop_height, crop_width = cropped.shape[:2]
        model_width = max(1, round(crop_width * MODEL_SIZE / side))
        model_height = max(1, round(crop_height * MODEL_SIZE / side))
        size = (model_width, model_height)
        shrink = side > MODEL_SIZE
        rgb = cv2.resize(cropped[:, :, :3], size, interpolation=cv2.INTER_AREA if shrink else cv2.INTER_LINEAR)
        unknown = cv2.resize(unknown, size, interpolation=cv2.INTER_AREA if shrink else cv2.INTER_NEAREST)
        bottom_pad, right_pad = MODEL_SIZE - model_height, MODEL_SIZE - model_width
        rgb = cv2.copyMakeBorder(rgb, 0, bottom_pad, 0, right_pad, cv2.BORDER_REFLECT)
        unknown = cv2.copyMakeBorder(unknown, 0, bottom_pad, 0, right_pad, cv2.BORDER_REFLECT)
        inputs = {
            "image": rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255,
            "mask": (unknown > 0)[None, None].astype(np.float32),
        }
        try:
            prediction = self.session.run(None, inputs)[0]
        except Exception as exc:
            raise ValueError(f"LaMa inpainting failed: {exc}") from exc
        if prediction.shape != (1, 3, MODEL_SIZE, MODEL_SIZE) or not np.isfinite(prediction).all():
            raise ValueError("Unexpected LaMa output")
        generated = np.rint(prediction[0].transpose(1, 2, 0).clip(0, 255)).astype(np.uint8)
        generated = cv2.resize(generated[:model_height, :model_width], (crop_width, crop_height),
                               interpolation=cv2.INTER_LINEAR)
        result = pixels.copy()
        rgb = result[region][:, :, :3]
        rgb[selected] = generated[selected]
        return result, expanded
