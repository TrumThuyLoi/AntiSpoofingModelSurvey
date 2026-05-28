"""MiniFASNet model wrapper for pre-cropped inputs."""

from __future__ import annotations

import importlib.util
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from src.preprocessing.minifasnet import preprocess_bgr, preprocess_path


class MiniFASNetWrapper:
    """Wrapper thống nhất cho MiniFASNet inference."""

    def __init__(
        self,
        model_config_path: str | Path | None = None,
        *,
        prefer_cpu: bool = True,
    ) -> None:
        self.repo_root = Path(__file__).resolve().parents[2]
        self.model_config_path = (
            Path(model_config_path)
            if model_config_path is not None
            else self.repo_root / "configs" / "model.yaml"
        )
        self.prefer_cpu = prefer_cpu

        self.config = self._load_yaml(self.model_config_path)
        self.input_size = self._parse_input_size(self.config)
        self.threshold = float(self.config.get("threshold", 0.5))
        self.device = self._resolve_device(str(self.config.get("device", "cpu")))
        self.weights_path = self._resolve_weights_path(str(self.config["weights_dir"]))

        self._model_mapping: dict[str, Any] | None = None
        self._parse_model_name = None
        self._get_kernel = None
        self.model: torch.nn.Module | None = None
        self._loaded = False

    @staticmethod
    def _load_yaml(path: Path) -> dict[str, Any]:
        with path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError(f"{path}: YAML root phải là dict")
        return data

    @staticmethod
    def _parse_input_size(cfg: dict[str, Any]) -> tuple[int, int]:
        raw = cfg.get("input_size", [80, 80])
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            raise ValueError(f"input_size không hợp lệ: {raw!r}")
        return int(raw[0]), int(raw[1])

    def _resolve_device(self, device_cfg: str) -> torch.device:
        spec = device_cfg.strip().lower()
        if self.prefer_cpu:
            return torch.device("cpu")
        if spec in ("gpu", "cuda") and torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")

    def _resolve_weights_path(self, value: str) -> Path:
        p = Path(value)
        if not p.is_absolute():
            p = self.repo_root / p
        if not p.is_file():
            raise FileNotFoundError(f"Không tìm thấy weight file: {p}")
        return p

    def _load_module(self, file_path: Path, module_name: str):
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Không load được module: {file_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def _load_submodule_symbols(self) -> None:
        third_party = self.repo_root / "third_party" / "Silent-Face-Anti-Spoofing" / "src"
        utility_mod = self._load_module(third_party / "utility.py", "sfas_utility")
        model_mod = self._load_module(third_party / "model_lib" / "MiniFASNet.py", "sfas_minifasnet")

        self._parse_model_name = utility_mod.parse_model_name
        self._get_kernel = utility_mod.get_kernel
        self._model_mapping = {
            "MiniFASNetV1": model_mod.MiniFASNetV1,
            "MiniFASNetV2": model_mod.MiniFASNetV2,
            "MiniFASNetV1SE": model_mod.MiniFASNetV1SE,
            "MiniFASNetV2SE": model_mod.MiniFASNetV2SE,
        }

    def _build_model(self) -> torch.nn.Module:
        if self._model_mapping is None or self._parse_model_name is None or self._get_kernel is None:
            self._load_submodule_symbols()

        h_input, w_input, model_type, _ = self._parse_model_name(self.weights_path.name)
        kernel_size = self._get_kernel(h_input, w_input)
        model_cls = self._model_mapping[model_type]
        model = model_cls(conv6_kernel=kernel_size).to(self.device)

        state_dict = torch.load(self.weights_path, map_location=self.device)
        first_key = next(iter(state_dict.keys()))
        if first_key.startswith("module."):
            stripped = OrderedDict((k[7:], v) for k, v in state_dict.items())
            model.load_state_dict(stripped)
        else:
            model.load_state_dict(state_dict)
        model.eval()
        return model

    def load(self) -> None:
        if self._loaded:
            return
        self.model = self._build_model()
        self._loaded = True

    def preprocess(self, image_or_path: str | Path | np.ndarray) -> torch.Tensor:
        if isinstance(image_or_path, (str, Path)):
            return preprocess_path(
                image_or_path,
                input_size=self.input_size,
                add_batch_dim=True,
                device=self.device,
            )
        if isinstance(image_or_path, np.ndarray):
            tensor = preprocess_bgr(image_or_path, self.input_size).unsqueeze(0).to(self.device)
            return tensor
        raise TypeError(f"Kiểu input không hỗ trợ: {type(image_or_path)}")

    def _predict_tensor(self, batch_tensor: torch.Tensor) -> dict[str, Any]:
        if not self._loaded:
            self.load()
        assert self.model is not None
        with torch.no_grad():
            probs = F.softmax(self.model.forward(batch_tensor), dim=1).detach().cpu().numpy()

        live_score = float(probs[0, 1])
        spoof_score = float(probs[0, 0] + probs[0, 2])
        label_pred = "live" if live_score >= self.threshold else "spoof"
        return {
            "label_pred": label_pred,
            "live_score": live_score,
            "spoof_score": spoof_score,
            "raw_output": probs[0].tolist(),
        }

    def predict(self, image_or_path: str | Path | np.ndarray) -> dict[str, Any]:
        batch_tensor = self.preprocess(image_or_path)
        return self._predict_tensor(batch_tensor)

    def predict_batch(self, images_or_paths: list[str | Path | np.ndarray]) -> list[dict[str, Any]]:
        outputs: list[dict[str, Any]] = []
        for item in images_or_paths:
            outputs.append(self.predict(item))
        return outputs

