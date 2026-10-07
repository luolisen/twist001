#!/usr/bin/env python3
"""Reference full-view XGW10 wrist RGB preprocessing for square policy input.

The camera acquisition layer must supply a native 1280x720 uint8 frame and
declare its channel order. The camera optical calibration remains a separate
gate; this function fixes only the pixel geometry and color order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image, ImageDraw


NATIVE_WIDTH = 1280
NATIVE_HEIGHT = 720
POLICY_SIZE = 256
CONTENT_HEIGHT = 144
PAD_TOP = (POLICY_SIZE - CONTENT_HEIGHT) // 2  # 56
PAD_BOTTOM = POLICY_SIZE - CONTENT_HEIGHT - PAD_TOP  # 56
SIM_RENDER_SHAPES = {(720, 1280, 3), (360, 640, 3), (288, 512, 3)}
CROP_LEFT = (NATIVE_WIDTH - NATIVE_HEIGHT) // 2  # 280
CROP_TOP = 0
CROP_RIGHT = CROP_LEFT + NATIVE_HEIGHT  # 1000, exclusive
CROP_BOTTOM = NATIVE_HEIGHT


def preprocess_wrist_frame(
    frame: np.ndarray, *, color_order: Literal["RGB", "BGR"], policy_size: Literal[256, 512] = 256
) -> np.ndarray:
    """Keep the full 16:9 view, then letterbox it to square RGB uint8.

    `color_order` is required: OpenCV camera frames are commonly BGR, while
    decoded JPEG/PNG frames are commonly RGB. Simulated lower-resolution
    frames use `preprocess_simulated_wrist_frame`; both share `_letterbox`.
    """
    if frame.shape != (NATIVE_HEIGHT, NATIVE_WIDTH, 3) or frame.dtype != np.uint8:
        raise ValueError("wrist frame must be native (720,1280,3) uint8")
    return _letterbox(frame, color_order=color_order, policy_size=policy_size)


def preprocess_simulated_wrist_frame(
    frame: np.ndarray, *, color_order: Literal["RGB", "BGR"] = "RGB",
    policy_size: Literal[256, 512] = 256,
) -> np.ndarray:
    """Use identical letterboxing for a supported full-view 16:9 MuJoCo render."""
    if frame.shape not in SIM_RENDER_SHAPES or frame.dtype != np.uint8:
        raise ValueError("sim wrist frame must be 1280x720, 640x360, or 512x288 RGB uint8")
    return _letterbox(frame, color_order=color_order, policy_size=policy_size)


def _letterbox(frame: np.ndarray, *, color_order: Literal["RGB", "BGR"],
               policy_size: Literal[256, 512]) -> np.ndarray:
    if color_order not in ("RGB", "BGR"):
        raise ValueError("color_order must be RGB or BGR")
    if policy_size not in (256, 512):
        raise ValueError("policy_size must be 256 or 512")
    rgb = frame if color_order == "RGB" else frame[..., ::-1]
    content_height = policy_size * 9 // 16
    pad_top = (policy_size - content_height) // 2
    resized = Image.fromarray(np.ascontiguousarray(rgb), mode="RGB").resize(
        (policy_size, content_height), resample=Image.Resampling.LANCZOS
    )
    output = np.zeros((policy_size, policy_size, 3), dtype=np.uint8)
    output[pad_top:pad_top + content_height] = np.asarray(resized, dtype=np.uint8)
    return output


def center_crop_wrist_frame_experimental(
    frame: np.ndarray, *, color_order: Literal["RGB", "BGR"]
) -> np.ndarray:
    """Retained for optical experiments; it discards edge fingers and is not policy input."""
    if frame.shape != (NATIVE_HEIGHT, NATIVE_WIDTH, 3) or frame.dtype != np.uint8:
        raise ValueError("wrist frame must be native (720,1280,3) uint8")
    if color_order not in ("RGB", "BGR"):
        raise ValueError("color_order must be RGB or BGR")
    rgb = frame if color_order == "RGB" else frame[..., ::-1]
    square = np.ascontiguousarray(rgb[CROP_TOP:CROP_BOTTOM, CROP_LEFT:CROP_RIGHT])
    resized = Image.fromarray(square, mode="RGB").resize(
        (POLICY_SIZE, POLICY_SIZE), resample=Image.Resampling.LANCZOS
    )
    return np.ascontiguousarray(np.asarray(resized, dtype=np.uint8))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_image", type=Path, help="native 1280x720 camera still")
    parser.add_argument("output_image", type=Path, help="square policy RGB PNG")
    parser.add_argument("--content-preview", type=Path, help="optional full-view content PNG")
    parser.add_argument("--layout-preview", type=Path,
                        help="optional policy PNG with the content boundary marked")
    parser.add_argument("--report", type=Path, help="optional JSON pixel-contract report")
    parser.add_argument("--policy-size", type=int, choices=(256, 512), default=256,
                        help="candidate wrist policy resolution; default remains 256")
    args = parser.parse_args()

    with Image.open(args.input_image) as source:
        orientation = source.getexif().get(274, 1)
        if orientation != 1:
            raise ValueError(f"source EXIF orientation {orientation} needs an explicit capture transform")
        frame = np.asarray(source.convert("RGB"))
    output = preprocess_wrist_frame(frame, color_order="RGB", policy_size=args.policy_size)
    content_height = args.policy_size * 9 // 16
    pad_top = (args.policy_size - content_height) // 2
    pad_bottom = args.policy_size - content_height - pad_top
    args.output_image.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(output, mode="RGB").save(args.output_image)

    if args.content_preview:
        args.content_preview.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(output[pad_top:pad_top + content_height], mode="RGB").save(
            args.content_preview
        )
    if args.layout_preview:
        args.layout_preview.parent.mkdir(parents=True, exist_ok=True)
        overlay = Image.fromarray(output, mode="RGB")
        ImageDraw.Draw(overlay).rectangle(
            (0, pad_top, args.policy_size - 1, pad_top + content_height - 1),
            outline=(255, 0, 255), width=1,
        )
        overlay.save(args.layout_preview)
    report = {
        "source_sha256": hashlib.sha256(args.input_image.read_bytes()).hexdigest(),
        "source_shape_hwc": list(frame.shape),
        "source_color_order": "RGB (decoded image)",
        "native_view_retained": True,
        "resized_content_shape_hwc": [content_height, args.policy_size, 3],
        "padding_top_bottom_px": [pad_top, pad_bottom],
        "padding_left_right_px": [0, 0],
        "policy_shape_hwc": list(output.shape),
        "policy_dtype": str(output.dtype),
        "resize_filter": "Pillow LANCZOS",
        "aspect_ratio_stretched": False,
        "padding_added": True,
        "center_crop_used": False,
        "output_sha256": hashlib.sha256(args.output_image.read_bytes()).hexdigest(),
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
