#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate consecutive shots into global-caption clips from the source video.

Atomic migration of ``lsj/video_process/scripts/aggregate_global_clips.py``.
Reads a ``shots.jsonl`` produced by :mod:`mimo_shot_splitter`, optionally runs
SCRL scene boundary detection, then greedily packs consecutive shots into
60~120 second groups and cuts each group from the **original source video** so
the audio stream stays continuous across shot boundaries.

Pipeline::

    shots.jsonl
      -> scene boundaries: SCRL (learned BiLSTM) | semsim (training-free
         DINOv3 context self-similarity + audio contrast) | metadata | none
      -> choose_groups (min/target/max seconds)
      -> export from source video via ffmpeg
      -> global_clips.jsonl

Usage::

    python mimo_global_clip_aggregator.py \
        --input ./outputs/shot_outputs/<video> \
        --output-dir ./outputs/global_outputs \
        --scene-detector scrl --scrl-device cuda:0
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from mimo_caption_client import find_shot_manifest, load_jsonl, resolve_clip_path

# SCRL 代码默认用本目录 m02_shots/teleai_src/others/scrl 的 vendored 副本
# （不加载 teleAI-T2V 仓库代码）；模型权重仍在 teleAI-T2V/local_models/。
# 集群默认路径可通过环境变量覆盖（SCRL_ENCODER_PATH / SCRL_BOUNDARY_PATH），
# 否则新机器上 --scene-detector auto 会响亮回落。
DEFAULT_SCRL_SOURCE_DIR = Path(__file__).resolve().parent / "m02_shots" / "teleai_src"
DEFAULT_SCRL_ENCODER = Path(
    os.environ.get("SCRL_ENCODER_PATH")
    or "/gemini/platform/public/aigc/lss/Interns/wyn/teleAI-T2V/local_models/scrl/checkpoint_0099.pth.tar"
)
DEFAULT_SCRL_BOUNDARY = Path(
    os.environ.get("SCRL_BOUNDARY_PATH")
    or "/gemini/platform/public/aigc/lss/Interns/wyn/teleAI-T2V/local_models/scrl/model_best.pth.tar"
)
# semsim 检测器复用场景提取链的本地 DINOv3 权重（transformers 目录格式）。
DEFAULT_SEMSIM_MODEL = Path(
    os.environ.get("SEMSIM_DINO_MODEL")
    or Path(__file__).resolve().parent / "local_models" / "dinov3-vitl16"
)
LOGGER = logging.getLogger("mimo_global_clip_aggregator")


# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------
def find_manifests(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.name != "shots.jsonl":
            raise ValueError(f"Input file must be shots.jsonl: {input_path}")
        return [input_path]
    manifests = sorted(input_path.rglob("shots.jsonl"))
    if not manifests:
        raise FileNotFoundError(f"No shots.jsonl found under {input_path}")
    return manifests


def _resolve_source(record: dict, manifest_path: Path) -> Path | None:
    raw_path = record.get("source_video")
    if not raw_path:
        return None
    path = Path(raw_path)
    if path.is_file():
        return path
    for base in manifest_path.parents:
        candidate = (base / path).resolve()
        if candidate.is_file():
            return candidate
    return None


def load_records(manifest_path: Path) -> list[dict]:
    records = []
    for line_number, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {manifest_path}:{line_number}: {exc}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"Manifest record must be an object: {manifest_path}:{line_number}")
        if "shot_id" not in record or "time_range_sec" not in record:
            raise ValueError(f"Missing shot_id/time_range_sec: {manifest_path}:{line_number}")
        record = dict(record)
        clip_path = record.get("clip_path")
        path = resolve_clip_path(clip_path, manifest_path) if clip_path else None
        if path is not None and path.is_file():
            record["clip_path"] = str(path)
        else:
            # Tolerate split_shots --no-export: scene detection can sample
            # frames from the source video instead of the missing clip.
            source = _resolve_source(record, manifest_path)
            if source is None:
                raise ValueError(
                    f"Record has neither a readable clip_path nor a resolvable source_video "
                    f"(rerun split_shots without --no-export, or fix source_video): "
                    f"{manifest_path}:{line_number}"
                )
            record["clip_path"] = None
            record["source_video"] = str(source)
        records.append(record)
    records.sort(key=lambda item: int(item["shot_id"]))
    if not records:
        raise ValueError(f"Manifest is empty: {manifest_path}")
    for expected_id, record in enumerate(records, start=1):
        if int(record["shot_id"]) != expected_id:
            raise ValueError(f"Shot IDs must be consecutive from 1 in {manifest_path}")
    return records


# ---------------------------------------------------------------------------
# Group selection (greedy by min/target/max seconds)
# ---------------------------------------------------------------------------
def choose_groups(records: list[dict], min_sec: float, target_sec: float, max_sec: float) -> list[list[dict]]:
    groups: list[list[dict]] = []
    start = 0
    while start < len(records):
        candidates: list[tuple[float, int]] = []
        total = 0.0
        for end in range(start, len(records)):
            begin = float(records[end]["time_range_sec"][0])
            finish = float(records[end]["time_range_sec"][1])
            total = max(0.0, finish - float(records[start]["time_range_sec"][0]))
            if total >= min_sec and total <= max_sec:
                candidates.append((abs(total - target_sec), end + 1))
            if total > max_sec:
                break
        if candidates:
            _, end_index = min(candidates)
        else:
            end_index = start + 1
            while end_index < len(records):
                total = float(records[end_index - 1]["time_range_sec"][1]) - float(records[start]["time_range_sec"][0])
                if total >= min_sec or total >= max_sec:
                    break
                end_index += 1
        groups.append(records[start:end_index])
        start = end_index

    if len(groups) >= 2:
        last_duration = float(groups[-1][-1]["time_range_sec"][1]) - float(groups[-1][0]["time_range_sec"][0])
        previous_duration = float(groups[-2][-1]["time_range_sec"][1]) - float(groups[-2][0]["time_range_sec"][0])
        if last_duration < min_sec and previous_duration + last_duration <= max_sec:
            groups[-2].extend(groups[-1])
            groups.pop()
    return groups


def metadata_scene_groups(records: list[dict]) -> list[list[dict]]:
    """Use an upstream scene ID when available, conservatively staying contiguous."""
    key_name = next(
        (name for name in ("scene_group_id", "scene_id", "raw_scene_index")
         if any(record.get(name) is not None for record in records)),
        None,
    )
    if key_name is None:
        return [[record] for record in records]
    groups: list[list[dict]] = []
    current_key = object()
    for record in records:
        key = record.get(key_name)
        if groups and key != current_key:
            groups.append([])
        if not groups:
            groups.append([])
        groups[-1].append(record)
        current_key = key
    return groups


# ---------------------------------------------------------------------------
# SCRL scene detector (loads from vendored m02_shots/teleai_src/others/scrl)
# ---------------------------------------------------------------------------
def resolve_scene_device(requested: str) -> str:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for scene grouping detectors.") from exc
    if requested == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested detector device {requested!r}, but CUDA is unavailable.")
    return requested


def load_scrl_detector(args: argparse.Namespace):
    source_dir = args.scrl_source_dir
    others_dir = source_dir / "others"
    if not (others_dir / "scrl").is_dir():
        raise RuntimeError(f"SCRL package not found under {others_dir}")
    for path in (str(source_dir), str(others_dir)):
        if path not in sys.path:
            sys.path.insert(0, path)
    try:
        import torch
        from scrl.SceneSeg.BiLSTM_protocol import BiLSTM
        from scrl.extract_embeddings import get_encoder
    except ImportError as exc:
        raise RuntimeError("SCRL dependencies are unavailable; install the TransNet environment.") from exc
    device = resolve_scene_device(args.scrl_device)
    encoder = get_encoder("resnet50", str(args.scrl_encoder), input_channel=9)
    encoder = encoder.to(device).eval()
    boundary_model = BiLSTM()
    checkpoint = torch.load(args.scrl_boundary, map_location=device)
    boundary_model.load_state_dict(checkpoint["state_dict"], strict=True)
    boundary_model = boundary_model.to(device).eval()
    try:
        import cv2
        from torchvision import transforms
    except ImportError as exc:
        raise RuntimeError("OpenCV and torchvision are required for SCRL scene grouping.") from exc
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    LOGGER.info("Loaded SCRL scene detector on %s", device)
    return {"torch": torch, "cv2": cv2, "encoder": encoder, "boundary_model": boundary_model,
            "transform": transform, "device": device}


def progress_or_identity(iterable, args: argparse.Namespace, *, total: int, desc: str, unit: str):
    if not args.progress:
        return iterable
    try:
        from tqdm import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, total=total, desc=desc, unit=unit, leave=False)


def iter_shot_frames(record: dict, fractions: list[float], detector: dict):
    """Yield BGR frames at given positions (0..1) of a shot.

    Reads the exported shot clip when present; otherwise seeks the source
    video at the shot's absolute time range (split_shots --no-export mode).
    """
    cv2 = detector["cv2"]
    t0, t1 = (float(value) for value in record["time_range_sec"][:2])
    clip_path = record.get("clip_path")
    if clip_path:
        capture = cv2.VideoCapture(str(clip_path))
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            capture.release()
            raise RuntimeError(f"Unable to read frames from {clip_path}")
        try:
            for fraction in fractions:
                index = min(total - 1, max(0, int(round(fraction * (total - 1)))))
                capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = capture.read()
                if not ok or frame is None:
                    raise RuntimeError(f"Unable to sample frame {index} from {clip_path}")
                yield frame
        finally:
            capture.release()
        return
    source = Path(record["source_video"])
    capture = cv2.VideoCapture(str(source))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open source video {source}")
        for fraction in fractions:
            target_sec = t0 + fraction * max(0.0, t1 - t0)
            capture.set(cv2.CAP_PROP_POS_MSEC, target_sec * 1000.0)
            ok, frame = capture.read()
            if not ok or frame is None:
                # Seek near the end can land past the last decodable frame.
                capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, target_sec - 0.2) * 1000.0)
                ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError(f"Unable to sample {target_sec:.2f}s from {source}")
            yield frame
    finally:
        capture.release()


def sample_clip_tensor(record: dict, detector: dict):
    """SCRL 3-frame input, mirroring the original fixed indices (2 / mid / last-10)."""
    clip_path = record.get("clip_path")
    if clip_path:
        cv2 = detector["cv2"]
        capture = cv2.VideoCapture(str(clip_path))
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        if total > 0:
            spans = [(index / max(1, total - 1)) for index in
                     (min(2, total - 1), total // 2, max(0, total - 10))]
        else:
            spans = [0.0, 0.5, 1.0]
    else:
        spans = [0.05, 0.5, 0.95]
    frames = [detector["transform"](cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
              for frame in iter_shot_frames(record, spans, detector)]
    return detector["torch"].cat(frames, dim=0).unsqueeze(0)


def scrl_scene_groups(records: list[dict], args: argparse.Namespace) -> list[list[dict]]:
    detector = args._scrl_detector
    torch = detector["torch"]
    features = []
    with torch.inference_mode():
        sampled_records = progress_or_identity(
            records, args, total=len(records), desc="SCRL scene detection", unit="shot"
        )
        for index, record in enumerate(sampled_records, start=1):
            data = sample_clip_tensor(record, detector).to(detector["device"])
            features.append(detector["encoder"](data).detach().cpu())
            if index % 50 == 0:
                LOGGER.info("SCRL sampled %d/%d shots", index, len(records))
    sequence = torch.cat(features, dim=0)
    seq_len = args.scrl_seq_len
    stride = seq_len // 2
    left = seq_len // 4
    right = seq_len // 4
    remainder = sequence.shape[0] % stride
    if remainder:
        right += stride - remainder
    padded = torch.cat((sequence[0].repeat(left, 1), sequence, sequence[-1].repeat(right, 1)), dim=0)
    probabilities = []
    with torch.inference_mode():
        for start in range(0, padded.shape[0] - seq_len + 1, stride):
            window = padded[start:start + seq_len].unsqueeze(0).to(detector["device"])
            output = detector["boundary_model"](window, None).reshape(-1, 2)[:, 1]
            probabilities.append(output[stride // 2:stride + stride // 2].detach().cpu())
    scores = torch.cat(probabilities).numpy()[:len(records)]
    boundaries = set(int(index) for index, score in enumerate(scores) if score > args.scrl_threshold)
    groups: list[list[dict]] = []
    start = 0
    for boundary_index in sorted(boundaries):
        end = min(len(records), boundary_index + 1)
        if end > start:
            groups.append(records[start:end])
            start = end
    if start < len(records):
        groups.append(records[start:])
    LOGGER.info("SCRL detected %d semantic scene groups", len(groups))
    return groups


# ---------------------------------------------------------------------------
# semsim scene detector: training-free DINOv3 context self-similarity
# blended with an audio-energy contrast cue.
#
# Grounding (no LLM, no extra training):
# - context self-similarity for scene boundaries follows the BaSSL
#   (Mun et al., CVPR 2022) / ShotCoL (Chen et al., 2021) line: a boundary is
#   where the *context windows* on both sides of a shot stop matching, not
#   merely where two adjacent shots differ;
# - the audio-visual contrast cue follows CMS-style multi-modal similarity;
# - the min/max scene-duration prior follows genre-and-duration-signal scene
#   segmentation work (long-form video needs duration regularisation).
# ---------------------------------------------------------------------------
def load_semsim_embedder(args: argparse.Namespace) -> dict:
    if not args.semsim_model.is_dir():
        raise FileNotFoundError(
            f"semsim DINOv3 model directory not found: {args.semsim_model} "
            f"(set --semsim-model or SEMSIM_DINO_MODEL)")
    try:
        import cv2
        import torch
        from transformers import AutoImageProcessor, AutoModel
    except ImportError as exc:
        raise RuntimeError("torch/transformers/opencv are required for semsim detection") from exc
    device = resolve_scene_device(args.semsim_device)
    processor = AutoImageProcessor.from_pretrained(args.semsim_model, local_files_only=True)
    model = AutoModel.from_pretrained(args.semsim_model, local_files_only=True).eval().to(device)
    LOGGER.info("Loaded semsim DINOv3 embedder on %s", device)
    return {"torch": torch, "cv2": cv2, "model": model, "processor": processor, "device": device}


def semsim_shot_embeddings(records: list[dict], embedder: dict, args: argparse.Namespace):
    """One L2-normalised embedding per shot, from adaptive frame sampling.

    Unlike SCRL's fixed 3 frames, longer shots contribute more frames
    (one per ~2 s, capped), so mid-shot scenery changes are not invisible.
    """
    import numpy as np
    torch = embedder["torch"]
    cv2 = embedder["cv2"]
    embeddings = []
    sampled_records = progress_or_identity(
        records, args, total=len(records), desc="semsim embeddings", unit="shot"
    )
    for record in sampled_records:
        duration = float(record["time_range_sec"][1]) - float(record["time_range_sec"][0])
        count = int(min(args.semsim_max_frames_per_shot, max(3, round(duration / 2.0))))
        fractions = [(index + 0.5) / count for index in range(count)]
        frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                  for frame in iter_shot_frames(record, fractions, embedder)]
        vectors = []
        for start in range(0, len(frames), args.semsim_batch):
            batch = frames[start:start + args.semsim_batch]
            inputs = embedder["processor"](images=batch, return_tensors="pt").to(embedder["device"])
            with torch.inference_mode():
                output = embedder["model"](**inputs).last_hidden_state.mean(dim=1)
            vectors.append(output.detach().cpu())
        pooled = torch.cat(vectors, dim=0).mean(dim=0).numpy()
        norm = float(np.linalg.norm(pooled))
        embeddings.append(pooled / norm if norm > 0 else pooled)
    return np.stack(embeddings, axis=0)


def audio_energy_profile(source_path: Path, ffmpeg_bin: str):
    """Per-100ms RMS loudness profile of a video's audio, via one ffmpeg pass.

    Returns (times_sec, rms) or None when the file has no audio stream.
    """
    import numpy as np
    command = [
        ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-i", str(source_path),
        "-map", "0:a?", "-vn", "-ac", "1", "-ar", "8000", "-f", "s16le", "-",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    block_samples = 800  # 100 ms at 8 kHz
    times, values = [], []
    index = 0
    assert process.stdout is not None
    try:
        while True:
            raw = process.stdout.read(block_samples * 2)
            if not raw:
                break
            samples = np.frombuffer(raw[:len(raw) // 2 * 2], dtype=np.int16).astype(np.float32) / 32768.0
            times.append((index + 0.5) * block_samples / 8000.0)
            values.append(float(np.sqrt(np.mean(samples ** 2))) if samples.size else 0.0)
            index += 1
    finally:
        process.stdout.close()
        process.wait()
    if not values:
        return None
    return np.asarray(times), np.asarray(values)


def _shot_rms(records: list[dict], profile):
    """Mean loudness per shot from the audio profile; None if no audio."""
    if profile is None:
        return None
    import numpy as np
    times, values = profile
    result = []
    for record in records:
        t0, t1 = (float(v) for v in record["time_range_sec"][:2])
        mask = (times >= t0) & (times < t1)
        result.append(float(values[mask].mean()) if mask.any() else 0.0)
    return np.asarray(result)


def semsim_boundary_scores(records: list[dict], embeddings, shot_rms, args: argparse.Namespace):
    """Scene-boundary score for each shot gap (length len(records)-1)."""
    import numpy as np
    count = len(records)
    window = max(1, args.semsim_window)
    scores = np.zeros(max(0, count - 1), dtype=np.float64)
    if count < 2:
        return scores

    def context_mean(matrix: np.ndarray, lo: int, hi: int) -> np.ndarray:
        return matrix[max(0, lo):max(0, hi) + 1].mean(axis=0)

    for i in range(count - 1):
        left = context_mean(embeddings, i - window + 1, i)
        right = context_mean(embeddings, i + 1, i + window)
        left /= max(1e-8, float(np.linalg.norm(left)))
        right /= max(1e-8, float(np.linalg.norm(right)))
        context_dissim = 1.0 - float(np.dot(left, right))
        adjacent_dissim = 1.0 - float(np.dot(embeddings[i], embeddings[i + 1]))
        visual = 0.6 * context_dissim + 0.4 * adjacent_dissim
        audio = 0.0
        if shot_rms is not None and args.semsim_audio_weight > 0:
            left_r = float(shot_rms[max(0, i - window + 1):i + 1].mean())
            right_r = float(shot_rms[i + 1:min(count, i + 1 + window)].mean())
            spread = float(shot_rms.std()) + 1e-6
            audio = min(3.0, abs(left_r - right_r) / spread)
        weight = args.semsim_audio_weight if shot_rms is not None else 0.0
        scores[i] = (1.0 - weight) * visual + weight * audio
    return scores


def _scene_spans(records: list[dict], boundaries: list[int]) -> list[tuple[int, int]]:
    points = [0, *(b + 1 for b in sorted(boundaries)), len(records)]
    return [(points[k], points[k + 1]) for k in range(len(points) - 1)]


def semsim_scene_groups(records: list[dict], args: argparse.Namespace) -> list[list[dict]]:
    embedder = args._semsim_embedder
    embeddings = semsim_shot_embeddings(records, embedder, args)
    profile = None
    if args.semsim_audio_weight > 0:
        source = None
        for record in records:
            source = _resolve_source(record, Path(record.get("source_video", ".")))
            if source is not None:
                break
        if source is not None:
            cache = getattr(args, "_semsim_audio_cache", None)
            if cache is None:
                cache = args._semsim_audio_cache = {}
            if str(source) not in cache:
                cache[str(source)] = audio_energy_profile(source, args.ffmpeg)
            profile = cache[str(source)]
            if profile is None:
                LOGGER.warning("semsim: no audio stream in %s; audio cue disabled", source)
    shot_rms = _shot_rms(records, profile)
    scores = semsim_boundary_scores(records, embeddings, shot_rms, args)

    import numpy as np
    if args.semsim_threshold is not None:
        threshold = float(args.semsim_threshold)
    else:
        threshold = float(scores.mean() + args.semsim_sigma * scores.std()) if scores.size else 1.0

    def span_duration(start: int, end: int) -> float:
        return float(records[end - 1]["time_range_sec"][1]) - float(records[start]["time_range_sec"][0])

    # Duration-regularised greedy acceptance: take the strongest candidate
    # boundaries first, but never create a scene shorter than the minimum.
    boundaries: list[int] = []
    candidates = [int(i) for i in np.argsort(scores)[::-1] if scores[i] > threshold]
    for index in candidates:
        spans = _scene_spans(records, boundaries)
        host = next(((s, e) for s, e in spans if s <= index and index + 1 < e), None)
        if host is None:
            continue
        host_start, host_end = host
        if (span_duration(host_start, index + 1) >= args.semsim_min_scene_sec
                and span_duration(index + 1, host_end) >= args.semsim_min_scene_sec):
            boundaries.append(index)
    # Enforce the maximum scene length by splitting at the strongest internal
    # candidate (fallback: the temporal midpoint of the scene).
    changed = True
    while changed:
        changed = False
        for start, end in _scene_spans(records, boundaries):
            if span_duration(start, end) <= args.semsim_max_scene_sec:
                continue
            internal = sorted(i for i in range(start, end - 1))
            if internal:
                best = max(internal, key=lambda i: scores[i] if i < len(scores) else -1.0)
            else:
                best = (start + end) // 2 - 1
            boundaries.append(best)
            changed = True
            break
    groups = [records[start:end] for start, end in _scene_spans(records, boundaries)]
    LOGGER.info("semsim detected %d semantic scene groups (threshold=%.3f, audio=%s)",
                len(groups), threshold, "on" if shot_rms is not None else "off")
    return groups


def get_scene_groups(records: list[dict], args: argparse.Namespace) -> tuple[list[list[dict]], str]:
    mode = args.scene_detector
    if mode == "none":
        return [records], "none"
    if mode == "metadata":
        return metadata_scene_groups(records), "metadata"
    if mode == "auto":
        use_scrl = args.scrl_encoder.is_file() and args.scrl_boundary.is_file()
        if args.scrl_device == "auto":
            try:
                import torch
                use_scrl = use_scrl and torch.cuda.is_available()
            except ImportError:
                use_scrl = False
        if use_scrl:
            try:
                args._scrl_detector = load_scrl_detector(args)
                return scrl_scene_groups(records, args), "scrl"
            except Exception:
                LOGGER.warning("SCRL unavailable; trying semsim before falling back.", exc_info=True)
        else:
            LOGGER.warning(
                "SCRL weights not found (%s / %s) — auto will use semsim instead. "
                "Set SCRL_ENCODER_PATH / SCRL_BOUNDARY_PATH to keep the SCRL behaviour.",
                args.scrl_encoder, args.scrl_boundary)
        if args.semsim_model.is_dir():
            try:
                args._semsim_embedder = load_semsim_embedder(args)
                return semsim_scene_groups(records, args), "semsim"
            except Exception:
                LOGGER.warning("semsim unavailable; using metadata scene groups.", exc_info=True)
        else:
            LOGGER.warning(
                "semsim DINOv3 model not found (%s) — falling back to metadata scene "
                "groups; boundary semantics will be much weaker. Set --semsim-model "
                "or SEMSIM_DINO_MODEL.", args.semsim_model)
        return metadata_scene_groups(records), "metadata"
    if mode == "semsim":
        if not hasattr(args, "_semsim_embedder"):
            args._semsim_embedder = load_semsim_embedder(args)
        return semsim_scene_groups(records, args), "semsim"
    if not hasattr(args, "_scrl_detector"):
        args._scrl_detector = load_scrl_detector(args)
    return scrl_scene_groups(records, args), "scrl"


# ---------------------------------------------------------------------------
# Clip export
# ---------------------------------------------------------------------------
def write_concat_list(paths: list[Path], directory: Path) -> Path:
    handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", dir=directory, delete=False)
    list_path = Path(handle.name)
    with handle:
        for path in paths:
            escaped = str(path.resolve()).replace("'", "'\\''")
            handle.write(f"file '{escaped}'\n")
    return list_path


def concat_clips(paths: list[Path], output_path: Path, ffmpeg_bin: str, reencode: bool) -> None:
    list_path = write_concat_list(paths, output_path.parent)
    try:
        command = [
            ffmpeg_bin, "-hide_banner", "-loglevel", "error",
            "-f", "concat", "-safe", "0",
            "-i", str(list_path),
            "-map", "0:v:0", "-map", "0:a?",
        ]
        if reencode:
            command += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac"]
        else:
            command += ["-c", "copy"]
        command += ["-movflags", "+faststart", "-y", str(output_path)]
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError:
            if reencode:
                raise
            LOGGER.warning("Stream-copy concat failed; retrying with re-encoding: %s", output_path.name)
            concat_clips(paths, output_path, ffmpeg_bin, reencode=True)
    finally:
        list_path.unlink(missing_ok=True)


def resolve_source_video(group: list[dict], manifest_path: Path) -> Path | None:
    for record in group:
        raw_path = record.get("source_video")
        if not raw_path:
            continue
        path = Path(raw_path)
        if path.is_file():
            return path
        for base in manifest_path.parents:
            candidate = (base / path).resolve()
            if candidate.is_file():
                return candidate
    return None


def cut_source_video(
    source_path: Path, output_path: Path, start_sec: float, end_sec: float, ffmpeg_bin: str, crf: int
) -> None:
    """Cut one contiguous source range so video and audio stay continuous."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.001, end_sec - start_sec)
    command = [
        ffmpeg_bin, "-hide_banner", "-loglevel", "error",
        "-ss", f"{start_sec:.6f}", "-i", str(source_path), "-t", f"{duration:.6f}",
        "-map", "0:v:0", "-map", "0:a?",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
        "-pix_fmt", "yuv420p", "-c:a", "aac",
        "-movflags", "+faststart", "-y", str(output_path),
    ]
    subprocess.run(command, check=True)


def export_global_clip(
    group: list[dict], output_path: Path, start_sec: float, end_sec: float,
    manifest_path: Path, args: argparse.Namespace,
) -> tuple[Path | None, str]:
    """Export from the source video when possible; fall back to shot concat."""
    source_path = resolve_source_video(group, manifest_path)
    if source_path is not None:
        cut_source_video(source_path, output_path, start_sec, end_sec, args.ffmpeg, args.crf)
        return source_path, "source_slice"
    LOGGER.warning("No source_video in %s; falling back to concatenated shot clips for %s",
                   manifest_path.name, output_path.name)
    concat_clips([Path(item["clip_path"]) for item in group], output_path, args.ffmpeg, args.reencode)
    return None, "concat_fallback"


def process_manifest(manifest_path: Path, args: argparse.Namespace) -> dict:
    records = load_records(manifest_path)
    scene_groups, scene_detector = get_scene_groups(records, args)
    groups: list[tuple[list[dict], int]] = []
    for scene_id, scene_records in enumerate(scene_groups, start=1):
        for group in choose_groups(scene_records, args.min_sec, args.target_sec, args.max_sec):
            groups.append((group, scene_id))
    video_name = manifest_path.parent.name
    if args.flat:
        # 扁平模式：直接落到 output_dir；clip 命名改用源视频 stem
        #（此模式下 manifest 父目录名没有语义）
        output_dir = args.output_dir
        source_stem = Path(str(records[0].get("source_video") or "")).stem
        if source_stem:
            video_name = source_stem
    else:
        output_dir = args.output_dir / video_name
    clip_dir = output_dir / "global_clips"
    clip_dir.mkdir(parents=True, exist_ok=True)
    output_records = []
    source_slice_count = 0
    concat_fallback_count = 0
    aggregate_groups = progress_or_identity(
        groups, args, total=len(groups), desc=f"Aggregate {video_name}", unit="clip"
    )
    for global_id, (group, scene_id) in enumerate(aggregate_groups, start=1):
        start_sec = float(group[0]["time_range_sec"][0])
        end_sec = float(group[-1]["time_range_sec"][1])
        clip_path = clip_dir / f"{video_name}_global_{global_id:04d}.mp4"
        source_video, export_mode = export_global_clip(
            group, clip_path, start_sec, end_sec, manifest_path, args
        )
        if export_mode == "source_slice":
            source_slice_count += 1
        else:
            concat_fallback_count += 1
        output_records.append({
            "global_id": global_id,
            "source_video": str(source_video) if source_video is not None else group[0].get("source_video"),
            "clip_path": str(clip_path),
            "time_range_sec": [round(start_sec, 6), round(end_sec, 6)],
            "duration_sec": round(end_sec - start_sec, 6),
            "shot_ids": [int(item["shot_id"]) for item in group],
            "shot_count": len(group),
            "scene_id": scene_id,
            "scene_detector": scene_detector,
            "source_shots_manifest": str(manifest_path),
            "export_mode": export_mode,
        })
    manifest_output = output_dir / "global_clips.jsonl"
    with manifest_output.open("w", encoding="utf-8") as handle:
        for record in output_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        "source_shots_manifest": str(manifest_path),
        "global_clip_count": len(output_records),
        "source_shot_count": len(records),
        "min_sec": args.min_sec,
        "target_sec": args.target_sec,
        "max_sec": args.max_sec,
        "scene_detector": scene_detector,
        "scene_group_count": len(scene_groups),
        "source_slice_count": source_slice_count,
        "concat_fallback_count": concat_fallback_count,
        "manifest": str(manifest_output),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("%s: %d shots -> %d global clips", video_name, len(records), len(output_records))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate shots within semantic scenes into global-caption clips.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--flat", action="store_true",
                        help="单视频模式：产物直接写入 --output-dir，不再追加 <视频名>/ 子目录")
    parser.add_argument("--min-sec", type=float, default=60.0)
    parser.add_argument("--target-sec", type=float, default=90.0)
    parser.add_argument("--max-sec", type=float, default=120.0)
    parser.add_argument(
        "--scene-detector",
        choices=["auto", "scrl", "semsim", "metadata", "none"],
        default="auto",
        help="auto: SCRL → semsim → metadata (loud warnings on each fallback); "
             "semsim: training-free DINOv3 context self-similarity + audio contrast",
    )
    parser.add_argument("--scrl-source-dir", type=Path, default=DEFAULT_SCRL_SOURCE_DIR)
    parser.add_argument("--scrl-encoder", type=Path, default=DEFAULT_SCRL_ENCODER)
    parser.add_argument("--scrl-boundary", type=Path, default=DEFAULT_SCRL_BOUNDARY)
    parser.add_argument("--scrl-device", default="auto")
    parser.add_argument("--scrl-threshold", type=float, default=0.5)
    parser.add_argument("--scrl-seq-len", type=int, default=16)
    parser.add_argument("--semsim-model", type=Path, default=DEFAULT_SEMSIM_MODEL,
                        help="local DINOv3 directory (same format as the scene extractor)")
    parser.add_argument("--semsim-device", default="auto")
    parser.add_argument("--semsim-window", type=int, default=3,
                        help="context window size (shots) on each side of a boundary")
    parser.add_argument("--semsim-audio-weight", type=float, default=0.25,
                        help="blend weight of the audio loudness-contrast cue (0 disables)")
    parser.add_argument("--semsim-threshold", type=float, default=None,
                        help="fixed boundary threshold; default adapts as mean+sigma*std")
    parser.add_argument("--semsim-sigma", type=float, default=1.0)
    parser.add_argument("--semsim-min-scene-sec", type=float, default=24.0)
    parser.add_argument("--semsim-max-scene-sec", type=float, default=240.0)
    parser.add_argument("--semsim-max-frames-per-shot", type=int, default=8)
    parser.add_argument("--semsim-batch", type=int, default=16)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--reencode", action="store_true")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING"], default="INFO")
    args = parser.parse_args()
    if not (0 < args.min_sec <= args.target_sec <= args.max_sec):
        parser.error("Require 0 < --min-sec <= --target-sec <= --max-sec")
    if args.scrl_seq_len < 4 or args.scrl_seq_len % 4:
        parser.error("--scrl-seq-len must be a positive multiple of 4 and at least 4")
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifests = find_manifests(args.input)
    if args.flat and len(manifests) > 1:
        parser.error("--flat 只支持单个 shots manifest 输入")
    failed = 0
    for manifest in manifests:
        try:
            process_manifest(manifest, args)
        except Exception:
            failed += 1
            LOGGER.exception("Failed to aggregate %s", manifest)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())