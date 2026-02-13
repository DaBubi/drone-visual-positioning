"""Export SuperPoint and LightGlue models to ONNX format.

This runs on the laptop (requires torch + kornia) and produces
ONNX files that run on the RPi via onnxruntime.

Usage:
    python -m programmer.export_onnx --output-dir ./models/
    # Then copy models/ to RPi: scp -r models/ pi@drone:/opt/vps/models/
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def export_superpoint(output_path: Path, image_size: int = 640) -> None:
    """Export kornia SuperPoint to ONNX.

    Args:
        output_path: where to save the .onnx file
        image_size: expected input image size (square)
    """
    import torch
    from kornia.feature import SuperPoint as KorniaSuperPoint

    class SuperPointWrapper(torch.nn.Module):
        """Wraps kornia SuperPoint for clean ONNX export."""

        def __init__(self):
            super().__init__()
            self.sp = KorniaSuperPoint(num_features=1024)

        def forward(self, image: torch.Tensor):
            """
            Args:
                image: (1, 1, H, W) grayscale float32 [0, 1]
            Returns:
                keypoints: (1, N, 2) xy coordinates
                descriptors: (1, N, 256) feature descriptors
            """
            # kornia SuperPoint expects (B, 1, H, W)
            out = self.sp(image)
            return out.keypoints, out.descriptors

    model = SuperPointWrapper()
    model.eval()

    dummy = torch.randn(1, 1, image_size, image_size)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        input_names=["image"],
        output_names=["keypoints", "descriptors"],
        dynamic_axes={
            "image": {2: "height", 3: "width"},
            "keypoints": {1: "num_keypoints"},
            "descriptors": {1: "num_keypoints"},
        },
        opset_version=17,
    )
    logger.info("Exported SuperPoint to %s", output_path)


def export_lightglue(output_path: Path, descriptor_dim: int = 256) -> None:
    """Export kornia LightGlue to ONNX.

    Note: LightGlue ONNX export can be tricky due to dynamic shapes.
    This provides a wrapper that handles the interface.

    Args:
        output_path: where to save the .onnx file
        descriptor_dim: descriptor dimension (256 for SuperPoint)
    """
    import torch
    from kornia.feature import LightGlue as KorniaLightGlue

    class LightGlueWrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lg = KorniaLightGlue("superpoint")

        def forward(
            self,
            kpts0: torch.Tensor,
            kpts1: torch.Tensor,
            desc0: torch.Tensor,
            desc1: torch.Tensor,
        ):
            """
            Args:
                kpts0: (1, N, 2) keypoints from image 0
                kpts1: (1, M, 2) keypoints from image 1
                desc0: (1, N, D) descriptors from image 0
                desc1: (1, M, D) descriptors from image 1
            Returns:
                matches: (1, K, 2) matched index pairs
                scores: (1, K) match confidence scores
            """
            data = {
                "keypoints0": kpts0,
                "keypoints1": kpts1,
                "descriptors0": desc0,
                "descriptors1": desc1,
            }
            out = self.lg(data)
            return out["matches"], out["scores"]

    model = LightGlueWrapper()
    model.eval()

    N, M = 100, 120
    dummy_kpts0 = torch.randn(1, N, 2)
    dummy_kpts1 = torch.randn(1, M, 2)
    dummy_desc0 = torch.randn(1, N, descriptor_dim)
    dummy_desc1 = torch.randn(1, M, descriptor_dim)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        torch.onnx.export(
            model,
            (dummy_kpts0, dummy_kpts1, dummy_desc0, dummy_desc1),
            str(output_path),
            input_names=["kpts0", "kpts1", "desc0", "desc1"],
            output_names=["matches", "scores"],
            dynamic_axes={
                "kpts0": {1: "N"},
                "kpts1": {1: "M"},
                "desc0": {1: "N"},
                "desc1": {1: "M"},
                "matches": {1: "K"},
                "scores": {1: "K"},
            },
            opset_version=17,
        )
        logger.info("Exported LightGlue to %s", output_path)
    except Exception as e:
        logger.warning(
            "LightGlue ONNX export failed (expected — model has dynamic control flow): %s", e
        )
        logger.info(
            "Workaround: Use ORB fallback on RPi (--use-orb-fallback), "
            "or use the hloc/LightGlue standalone ONNX exporter."
        )
        raise


def main():
    parser = argparse.ArgumentParser(description="Export feature matching models to ONNX")
    parser.add_argument("--output-dir", type=Path, default=Path("./models"),
                        help="Output directory for ONNX files")
    parser.add_argument("--image-size", type=int, default=640,
                        help="Expected input image size")
    parser.add_argument("--superpoint-only", action="store_true",
                        help="Only export SuperPoint")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    export_superpoint(args.output_dir / "superpoint.onnx", args.image_size)

    if not args.superpoint_only:
        try:
            export_lightglue(args.output_dir / "lightglue.onnx")
        except Exception:
            logger.info("Continuing without LightGlue ONNX. Use ORB fallback on RPi.")


if __name__ == "__main__":
    main()
