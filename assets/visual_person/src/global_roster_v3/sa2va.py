# -*- coding: utf-8 -*-
"""Self-contained Sa2VA single-frame referring segmentation.

Loads Sa2VA-Qwen3-VL-4B and segments one frame per call with a fixed
instruction (default: all main characters, plural).  No repo imports.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

DEFAULT_PROMPT = "<image>Please segment all the main characters."


def _mask_to_2d(value) -> Optional[np.ndarray]:
    """Normalize a raw mask tensor to a single 2D boolean array."""
    array = np.asarray(value)
    while array.ndim > 2 and 1 in array.shape:
        axis = int(np.argmin(array.shape))
        array = array.squeeze(axis=axis)
    if array.ndim != 2:
        return None
    return array.astype(bool)


class Sa2VASegmenter:
    def __init__(
        self,
        model_path: str,
        *,
        prompt: str = DEFAULT_PROMPT,
        dtype: str = "bfloat16",
        device: str = "cuda:0",
    ):
        self.model_path = model_path
        self.prompt = prompt
        self.dtype_name = dtype
        self.device = device
        self._model = None
        self._processor = None

    def load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoConfig, AutoProcessor
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }.get(self.dtype_name, torch.bfloat16)
        config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
        model_class = get_class_from_dynamic_module(
            config.auto_map["AutoModel"], self.model_path
        )
        if not hasattr(model_class, "all_tied_weights_keys"):
            model_class.all_tied_weights_keys = {}
        loaded = model_class.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=False,
            use_flash_attn=False,
            trust_remote_code=True,
        )
        if not hasattr(loaded, "all_tied_weights_keys"):
            loaded.all_tied_weights_keys = []
        self._model = loaded.eval().to(self.device)
        self._processor = AutoProcessor.from_pretrained(
            self.model_path, trust_remote_code=True, use_fast=False
        )
        if not getattr(self._processor, "chat_template", None):
            template_path = Path(__file__).with_name("qwen3_vl_chat_template.j2")
            self._processor.chat_template = template_path.read_text(encoding="utf-8")
            if getattr(self._processor, "tokenizer", None) is not None:
                self._processor.tokenizer.chat_template = self._processor.chat_template

    def segment_frame(self, pil_frame) -> np.ndarray:
        """Return a single (H, W) boolean mask for one frame (empty if none)."""
        self.load()
        result = self._model.predict_forward(
            video=[pil_frame],
            text=self.prompt,
            past_text="",
            mask_prompts=None,
            processor=self._processor,
        )
        masks = result.get("prediction_masks", [])
        if masks is None or len(masks) == 0:
            return np.zeros((0, 0), dtype=bool)
        mask = _mask_to_2d(masks)
        return mask if mask is not None else np.zeros((0, 0), dtype=bool)

    def segment_frames(self, pil_frames: List) -> np.ndarray:
        """Return (n, H, W) bool masks aligned to the input frames."""
        height = width = 0
        per_frame: List[np.ndarray] = []
        for frame in pil_frames:
            mask = self.segment_frame(frame)
            if mask.size:
                height, width = mask.shape
            per_frame.append(mask)
        out = np.zeros((len(pil_frames), height, width), dtype=bool)
        for index, mask in enumerate(per_frame):
            if mask.size and mask.shape == (height, width):
                out[index] = mask
        return out
