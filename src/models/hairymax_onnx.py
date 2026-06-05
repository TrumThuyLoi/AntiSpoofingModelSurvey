"""Wrapper ONNX hairymax — dùng AntiSpoof từ third_party/Face-AntiSpoofing-hairymax."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

HAIRYMAX_DEFAULT_DIR = "third_party/Face-AntiSpoofing-hairymax"


def is_live_accepted(probs: Any, *, threshold: float) -> bool:
    """Quy ước hairymax binary: class 0 = live, chấp nhận nếu argmax==0 và p[0] > threshold."""
    p = np.asarray(probs).reshape(-1)
    if p.size == 0:
        return False
    label = int(np.argmax(p))
    score = float(p[0])
    return label == 0 and score > threshold


def probs_to_prediction(probs: Any, *, threshold: float) -> dict[str, Any]:
    p = np.asarray(probs).reshape(-1)
    if p.size < 2:
        raise ValueError("probs cần ít nhất 2 phần tử (live, spoof).")
    live_score = float(p[0])
    spoof_score = float(p[1])
    return {
        "label_pred": "live" if is_live_accepted(p, threshold=threshold) else "spoof",
        "live_score": live_score,
        "spoof_score": spoof_score,
        "raw_output": p.tolist(),
    }


def _onnx_providers(*, prefer_cpu: bool) -> list[str]:
    import onnxruntime as ort

    available = ort.get_available_providers()
    if prefer_cpu or "CUDAExecutionProvider" not in available:
        return ["CPUExecutionProvider"]
    return ["CUDAExecutionProvider", "CPUExecutionProvider"]


class HairymaxONNXWrapper:
    """Minimal interface-compatible wrapper: load / predict / predict_batch."""

    def __init__(
        self,
        weights_path: str | Path,
        *,
        hairymax_dir: str | Path | None = None,
        input_size: tuple[int, int] = (128, 128),
        threshold: float = 0.5,
        prefer_cpu: bool = True,
    ) -> None:
        self.repo_root = Path(__file__).resolve().parents[2]
        self.weights_path = self._resolve_weights_path(weights_path)
        self.hairymax_dir = self._resolve_hairymax_dir(hairymax_dir)
        self.input_size = (int(input_size[0]), int(input_size[1]))
        self.threshold = float(threshold)
        self.prefer_cpu = prefer_cpu
        self._device = "cpu" if prefer_cpu else "cuda"
        self._detector: Any = None
        self._loaded = False

    def _resolve_weights_path(self, value: str | Path) -> Path:
        p = Path(value)
        if not p.is_absolute():
            p = self.repo_root / p
        if not p.is_file():
            raise FileNotFoundError(f"Không tìm thấy weight file: {p}")
        return p

    def _resolve_hairymax_dir(self, value: str | Path | None) -> Path:
        p = Path(value or HAIRYMAX_DEFAULT_DIR)
        if not p.is_absolute():
            p = self.repo_root / p
        return p

    @staticmethod
    def _import_antispoof_class(hairymax_dir: Path) -> type:
        face_py = hairymax_dir / "src" / "FaceAntiSpoofing.py"
        if not face_py.is_file():
            raise FileNotFoundError(
                f"Thiếu {face_py}. Clone:\n"
                "  git clone --depth 1 https://github.com/hairymax/Face-AntiSpoofing.git "
                f"{hairymax_dir}",
            )
        if str(hairymax_dir) not in sys.path:
            sys.path.insert(0, str(hairymax_dir))
        from src.FaceAntiSpoofing import AntiSpoof  # noqa: E402

        return AntiSpoof

    def load(self) -> None:
        if self._loaded:
            return

        AntiSpoof = self._import_antispoof_class(self.hairymax_dir)
        prefer_cpu = self.prefer_cpu
        providers = _onnx_providers(prefer_cpu=prefer_cpu)
        self._device = "cpu" if providers == ["CPUExecutionProvider"] else "cuda"
        original_init = AntiSpoof._init_session_

        def _init_session_patched(_self: Any, onnx_model_path: str) -> tuple[Any, str | None]:
            import onnxruntime as ort

            if not __import__("os").path.isfile(onnx_model_path):
                return None, None
            session = ort.InferenceSession(onnx_model_path, providers=providers)
            return session, session.get_inputs()[0].name

        AntiSpoof._init_session_ = _init_session_patched
        try:
            self._detector = AntiSpoof(
                str(self.weights_path),
                model_img_size=int(self.input_size[0]),
            )
        finally:
            AntiSpoof._init_session_ = original_init

        if self._detector.ort_session is None:
            raise RuntimeError(f"Không load được ONNX session: {self.weights_path}")
        self._loaded = True

    @property
    def device(self) -> str:
        return self._device

    @staticmethod
    def _read_bgr(image_or_path: str | Path | np.ndarray) -> np.ndarray:
        if isinstance(image_or_path, (str, Path)):
            bgr = cv2.imread(str(image_or_path))
            if bgr is None:
                raise ValueError(f"cv2.imread thất bại: {image_or_path}")
            return bgr
        if isinstance(image_or_path, np.ndarray):
            if image_or_path.ndim != 3 or image_or_path.shape[2] != 3:
                raise ValueError("np.ndarray input phải có shape [H, W, 3].")
            return image_or_path.astype(np.uint8)
        raise TypeError(f"Kiểu input không hỗ trợ: {type(image_or_path)}")

    def predict(self, image_or_path: str | Path | np.ndarray) -> dict[str, Any]:
        if not self._loaded:
            self.load()
        assert self._detector is not None
        bgr = self._read_bgr(image_or_path)
        probs = self._detector([bgr])[0]
        return probs_to_prediction(probs, threshold=self.threshold)

    def predict_batch(
        self, images_or_paths: list[str | Path | np.ndarray]
    ) -> list[dict[str, Any]]:
        return [self.predict(item) for item in images_or_paths]
