# -*- coding: utf-8 -*-
"""⑦ 窗内视觉一致性（DINOv2）与叙事连续性（小 LLM）。

视觉一致性使用 DINOv2 (vit_base_patch14_dinov2, timm) 提取代表帧 CLS
embedding，窗内两两余弦相似度均值作为场景连贯性得分。
若 torch/timm 不可用则降级为灰度直方图代理（proxy=True）。

叙事连续性：小 LLM 输入两个 clip 的结构化 caption + 连续性标记，
输出 是/否/不确定（轻量 scene boundary detection）。
"""

from __future__ import annotations

import importlib.util
import os
from typing import List, Optional, Tuple

import numpy as np

from ..common import ffmpeg as ff
from ..common.vlm import LLMClient, extract_json
from ..schema import ClipCaption, ClipRecord


class DINOv3Embedder:
    """视觉代表帧 embedding（CLIP ViT-L/14，本地权重）。"""

    DEFAULT_WEIGHTS = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "..", "local_models", "ViT-L-14.pt")
    EMBED_DIM = 768

    def __init__(self, device: str = "cuda", weights: str = ""):
        self._device = device
        self._weights = weights or os.path.normpath(self.DEFAULT_WEIGHTS)
        self._model = None
        self._preprocess = None

    def available(self) -> bool:
        return (importlib.util.find_spec("torch") is not None
                and os.path.isfile(self._weights))

    def _load(self):
        if self._model is not None:
            return
        import torch
        import clip
        model, preprocess = clip.load(self._weights, device=self._device)
        self._model = model.float().eval()
        self._preprocess = preprocess

    def embed(self, video_path: str, max_frames: int = 4) -> np.ndarray:
        """从视频中抽代表帧，返回 CLIP visual embedding（均值池化）。"""
        self._load()
        import torch
        from PIL import Image

        frames = ff.read_rgb_frames(video_path, fps=0.5, width=224, height=224,
                                    max_frames=max_frames)
        if len(frames) == 0:
            return np.zeros(self.EMBED_DIM, dtype=np.float32)

        tensors = torch.stack([self._preprocess(Image.fromarray(f)) for f in frames])
        tensors = tensors.to(self._device)

        with torch.no_grad(), torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            feats = self._model.encode_image(tensors)  # (N, 768)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.mean(dim=0).cpu().float().numpy()


def gray_hist_embedding(image_path: str, bins: int = 32) -> np.ndarray:
    """降级代理：灰度直方图向量（仅保证同场景光照相似的最粗判定）。"""
    frames = ff.read_gray_frames(image_path, fps=0.2, width=64, height=64,
                                 max_frames=4)
    if len(frames) == 0:
        return np.zeros(bins, dtype=np.float32)
    h = np.histogram(frames, bins=bins, range=(0, 256))[0].astype(np.float32)
    return h / max(h.sum(), 1e-12)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(a @ b / (na * nb))


def window_consistency(embeddings: List[np.ndarray]) -> float:
    """窗内两两相似度均值。"""
    if len(embeddings) < 2:
        return 1.0
    sims = [cosine(embeddings[i], embeddings[j])
            for i in range(len(embeddings)) for j in range(i + 1, len(embeddings))]
    return float(np.mean(sims))


# ---------------------------------------------------------------------------
# 叙事连续性判定
# ---------------------------------------------------------------------------

CONTINUITY_PROMPT = """判断镜头 B 是否是镜头 A 的剧情延续（同一场景/同一事件的继续）。
镜头 A：{a}
镜头 B：{b}
邻接连续性标记（同镜头再切割为"是"）：{link}
只输出 JSON：{{"continuity": "是" | "否" | "不确定", "reason": "一句话"}}"""


def caption_brief(cap: ClipCaption) -> str:
    parts = [f"场景:{cap.scene}", f"人物:{','.join(cap.characters)}",
             f"动作:{cap.action}", f"对话:{cap.dialogue}"]
    return "；".join(p for p in parts if not p.endswith(":"))


def build_continuity_prompt(a: ClipRecord, b: ClipRecord) -> str:
    link = "是" if b.prev.clip_id == a.clip_id and b.prev.continuous else "否"
    return CONTINUITY_PROMPT.format(a=caption_brief(a.caption),
                                    b=caption_brief(b.caption), link=link)


def parse_continuity(text: str) -> Tuple[bool, str]:
    """是→True；否→False；不确定/解析失败→按 False 处理并保留原文。"""
    try:
        data = extract_json(text)
        v = str(data.get("continuity", ""))
        return v == "是", str(data.get("reason", ""))
    except Exception:
        return False, f"parse_error: {text[:100]}"


def window_continuity(window: List[ClipRecord], client: LLMClient) -> Tuple[bool, List[str]]:
    """窗内相邻 clip 两两判定；出现"否"则整窗不连续。"""
    reasons: List[str] = []
    ok = True
    for a, b in zip(window, window[1:]):
        # 同 parent 的再切割子 clip 必然连续，省一次 LLM 调用
        if b.prev.clip_id == a.clip_id and b.prev.continuous:
            continue
        yes, reason = parse_continuity(client.chat(build_continuity_prompt(a, b)))
        reasons.append(f"{a.clip_id}->{b.clip_id}: {reason}")
        ok = ok and yes
    return ok, reasons
