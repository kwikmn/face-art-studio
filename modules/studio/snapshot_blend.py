"""Offline alignment and edge blending of a generated edit into its reference."""

from pathlib import Path
import hashlib
import json
import uuid
import cv2
import numpy as np
from PIL import Image, ImageOps, PngImagePlugin


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_photo(path):
    with Image.open(path) as photo:
        image = ImageOps.exif_transpose(photo).convert("RGB")
        image.thumbnail((1536, 1536), Image.Resampling.LANCZOS)
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def align_edit(original, edited, base_points, edit_points):
    base_points = np.asarray(base_points, dtype=np.float32)
    edit_points = np.asarray(edit_points, dtype=np.float32)
    if base_points.shape != (5, 2) or edit_points.shape != (5, 2):
        raise ValueError("Five facial landmarks are required for alignment.")
    if not np.isfinite(base_points).all() or not np.isfinite(edit_points).all():
        raise ValueError("Invalid facial landmarks.")
    transform, _ = cv2.estimateAffinePartial2D(
        edit_points, base_points, method=cv2.LMEDS
    )
    if (
        transform is None
        or not np.isfinite(transform).all()
        or np.linalg.det(transform[:, :2]) <= 0
    ):
        raise ValueError("Cannot align the edit with the reference face.")
    fitted = edit_points @ transform[:, :2].T + transform[:, 2]
    eye_distance = np.linalg.norm(base_points[1] - base_points[0])
    if (
        eye_distance < 5
        or np.linalg.norm(fitted - base_points, axis=1).max() > eye_distance * 0.3
    ):
        raise ValueError(
            "The edited pose changed too much. Try an edit that preserves the head position."
        )
    extent = (original.shape[1], original.shape[0])
    aligned = cv2.warpAffine(
        edited,
        transform,
        extent,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    support = cv2.warpAffine(
        np.full(edited.shape[:2], 255, np.uint8), transform, extent
    )
    distance = cv2.distanceTransform((support == 255).astype(np.uint8), cv2.DIST_L2, 3)
    support = np.clip(distance / 12, 0, 1)
    return aligned, support, transform


def prepare_pair(reference, edited):
    from modules.live_face import LiveFaceProcessor

    original = read_photo(reference)
    changed = read_photo(edited)
    loader = LiveFaceProcessor("", providers=["CPUExecutionProvider"])
    from modules.paths import MODELS_DIR
    detector = loader._load(Path(MODELS_DIR) / "buffalo_l/det_10g.onnx")
    detector.prepare(ctx_id=-1, input_size=(640, 640), det_thresh=0.6)
    observations = []
    for image in (original, changed):
        boxes, points = detector.detect(image, max_num=0)
        if len(boxes) != 1 or points is None:
            raise ValueError("Blending needs one clearly visible face in both photos.")
        observations.append((boxes[0, :4], points[0]))
    aligned, support, transform = align_edit(
        original, changed, observations[0][1], observations[1][1]
    )
    return dict(
        original=original,
        aligned=aligned,
        support=support,
        box=observations[0][0],
        transform=transform,
    )


def blend_pair(pair, coverage=1.0, feather=0.25, offset_x=0.0, offset_y=0.0):
    if not all(np.isfinite(v) for v in (coverage, feather, offset_x, offset_y)):
        raise ValueError("Invalid blend settings.")
    if (
        not 0.65 <= coverage <= 1.4
        or not 0.05 <= feather <= 0.5
        or max(abs(offset_x), abs(offset_y)) > 0.25
    ):
        raise ValueError("Blend settings outside supported range.")
    image = pair["original"]
    x1, y1, x2, y2 = pair["box"]
    width, height = x2 - x1, y2 - y1
    if width <= 0 or height <= 0:
        raise ValueError("Invalid face bounds.")
    y, x = np.mgrid[: image.shape[0], : image.shape[1]]
    cx = (x1 + x2) / 2 + offset_x * width
    cy = (y1 + y2) / 2 + offset_y * height
    radius = np.sqrt(
        ((x - cx) / (0.53 * width * coverage)) ** 2
        + ((y - cy) / (0.56 * height * coverage)) ** 2
    )
    opacity = np.clip((1 - radius) / feather, 0, 1)
    opacity = opacity * opacity * (3 - 2 * opacity) * pair["support"]
    weight = opacity[:, :, None].astype(np.float32)
    result = (
        np.rint(pair["aligned"] * weight + image * (1 - weight))
        .clip(0, 255)
        .astype(np.uint8)
    )
    return result, opacity


def save_blend(image, folder, metadata, reference, edited, settings, transform):
    with Image.open(edited) as source:
        try:
            data = json.loads(source.info.get("FaceArt Studio", "{}"))
        except ValueError:
            data = {}
    data.update(metadata)
    data.update(
        mode="blend",
        reference_sha256=file_hash(reference),
        edited_sha256=file_hash(edited),
        blend=settings,
        alignment=transform.tolist(),
        size=image.shape[1],
    )
    png = PngImagePlugin.PngInfo()
    png.add_text("FaceArt Studio", json.dumps(data, ensure_ascii=False))
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"face-blend-{uuid.uuid4().hex}.png"
    temporary = path.with_suffix(".tmp")
    try:
        Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).save(
            temporary, format="PNG", pnginfo=png
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return str(path)
