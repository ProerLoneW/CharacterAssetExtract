#!/usr/bin/env python3
"""Stage 5 V2: Speech Window -> Local Speaker Tracklet -> Global SPK.

Diarization labels are local activity evidence, not global identities. Overlap
is retained in a multi-label timeline. Identity embeddings and voice assets
are extracted only from non-overlap speech.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import soundfile as sf

SCHEMA_VERSION = "speaker-registry-v2"
SAMPLE_RATE = 16000


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def unit(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, np.float32).reshape(-1)
    return value / max(float(np.linalg.norm(value)), 1e-9)


def extract_audio(source: Path, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
                    "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le",
                    str(destination)], check=True)
    info = sf.info(str(destination))
    return {"source": str(source), "audio_path": str(destination), "sample_rate": info.samplerate,
            "channels": info.channels, "duration": round(info.duration, 6), "audio_offset": 0.0}


def slice_audio(audio: np.ndarray, start: float, end: float, destination: Path) -> bool:
    left, right = max(0, int(round(start*SAMPLE_RATE))), min(len(audio), int(round(end*SAMPLE_RATE)))
    if right <= left:
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(destination), audio[left:right], SAMPLE_RATE, subtype="PCM_16")
    return True


def _rows(raw: Any) -> list[Any]:
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        for key in ("segments", "turns", "text", "diarization", "overlaps", "windows"):
            if isinstance(raw.get(key), list):
                return raw[key]
        # Accept diagnostic/sidecar wrappers such as {"pyannote": {"segments": ...}}.
        for value in raw.values():
            nested = _rows(value)
            if nested:
                return nested
    return []


def _first(row: dict[str, Any], keys: Sequence[str]) -> Any:
    return next((row[key] for key in keys if row.get(key) is not None), None)


def normalize_diarization(raw: Any, offset: float = 0.0) -> list[tuple[float, float, str]]:
    """Normalize tuple/dict formats and expand ``active_speakers`` lists."""
    output = []
    for item in _rows(raw):
        if isinstance(item, (list, tuple)) and len(item) >= 3:
            start, end, labels = item[:3]
        elif isinstance(item, dict):
            start = _first(item, ("start", "start_time", "begin"))
            end = _first(item, ("end", "end_time", "stop"))
            labels = _first(item, ("active_speakers", "speakers", "speaker", "label", "speaker_id"))
        else:
            continue
        if start is None or end is None or labels is None or float(end) <= float(start):
            continue
        labels = labels if isinstance(labels, (list, tuple, set)) else [labels]
        output.extend((float(start)+offset, float(end)+offset, str(label)) for label in labels)
    return sorted(set(output), key=lambda row: (row[0], row[1], row[2]))


def normalize_vad(raw: Any) -> list[tuple[float, float]]:
    output = []
    for item in _rows(raw):
        if isinstance(item, dict) and isinstance(item.get("value"), list):
            output.extend(normalize_vad(item["value"]))
            continue
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            start, end = item[:2]
        elif isinstance(item, dict):
            start = _first(item, ("start", "start_time", "begin"))
            end = _first(item, ("end", "end_time", "stop"))
        else:
            continue
        if start is not None and end is not None and float(end) > float(start):
            output.append((float(start), float(end)))
    return merge_intervals(output)


def _seconds_if_needed(rows: Sequence[tuple], duration: float) -> list[tuple]:
    """ModelScope VAD commonly returns milliseconds; diarizers often return seconds."""
    maximum = max((float(row[1]) for row in rows), default=0.0)
    scale = 0.001 if maximum > max(100.0, duration * 5.0) else 1.0
    return [tuple([float(row[0])*scale, float(row[1])*scale, *row[2:]]) for row in rows]


def merge_intervals(intervals: Iterable[tuple[float, float]], gap: float = 0.0) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1] + gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(round(a, 6), round(b, 6)) for a, b in merged]


def intersect_intervals(left: Sequence[tuple[float, float]], right: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    return merge_intervals((max(a, c), min(b, d)) for a, b in left for c, d in right if min(b, d) > max(a, c))


def subtract_intervals(base: Sequence[tuple[float, float]], blockers: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    result = []
    blockers = merge_intervals(blockers)
    for start, end in base:
        cursor = start
        for block_start, block_end in blockers:
            if block_end <= cursor or block_start >= end:
                continue
            if block_start > cursor:
                result.append((cursor, min(block_start, end)))
            cursor = max(cursor, block_end)
            if cursor >= end:
                break
        if cursor < end:
            result.append((cursor, end))
    return [(round(a, 6), round(b, 6)) for a, b in result if b > a]


class Diarizer:
    def __init__(self, model: Path):
        from modelscope.pipelines import pipeline
        self.pipeline = pipeline(task="speaker-diarization", model=str(model))
    def __call__(self, audio: Path, oracle_num: Optional[int]) -> list[tuple[float, float, str]]:
        raw = self.pipeline(str(audio), oracle_num=oracle_num) if oracle_num else self.pipeline(str(audio))
        return _seconds_if_needed(normalize_diarization(raw), sf.info(str(audio)).duration)


class SpeechDetector:
    def __init__(self, model: Path):
        from modelscope.pipelines import pipeline
        self.pipeline = pipeline(task="voice-activity-detection", model=str(model))
    def __call__(self, audio: Path) -> list[tuple[float, float]]:
        return _seconds_if_needed(normalize_vad(self.pipeline(str(audio))), sf.info(str(audio)).duration)


class SpeakerEmbedder:
    """CAM++ compatibility implementation of the identity-only encoder API."""
    def __init__(self, model_dir: Path):
        import torch
        from speakerlab.models.campplus.DTDNN import CAMPPlus
        from speakerlab.process.processor import FBank
        self.torch, self.fbank = torch, FBank(80, sample_rate=SAMPLE_RATE, mean_nor=True)
        self.model = CAMPPlus(feat_dim=80, embedding_size=192)
        state = torch.load(str(model_dir/"campplus_cn_en_common.pt"), map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
    def embed(self, waveform: np.ndarray) -> Optional[np.ndarray]:
        if len(waveform) < int(.5*SAMPLE_RATE):
            return None
        feature = self.fbank(self.torch.from_numpy(np.asarray(waveform, np.float32)).unsqueeze(0))
        with self.torch.inference_mode():
            vector = self.model(feature.unsqueeze(0))[0].cpu().numpy()
        return unit(vector) if np.isfinite(vector).all() else None


class Transcriber:
    def __init__(self, model: Path, device: str, compute_type: str):
        from faster_whisper import WhisperModel
        self.model = WhisperModel(str(model), device=device, compute_type=compute_type, local_files_only=True)
    def __call__(self, audio: Path, language: Optional[str]) -> list[dict[str, Any]]:
        segments, _ = self.model.transcribe(str(audio), language=language, vad_filter=True, word_timestamps=False)
        return [{"asr_id": f"ASR{i:05d}", "start": round(float(x.start), 3),
                 "end": round(float(x.end), 3), "text": x.text.strip(),
                 "confidence": round(float(math.exp(x.avg_logprob)), 6)} for i, x in enumerate(segments, 1)]


@dataclass
class SpeechWindow:
    window_id: str; start: float; end: float; vad_start: float; vad_end: float
    asr_text: str = ""; asr_confidence: float = 0.0


@dataclass
class SpeakerTurn:
    turn_id: str; local_tracklet_id: str; start: float; end: float; duration: float
    audio_path: str; quality_score: float; quality_tier: str; quality: dict[str, float]
    overlap: bool; embedding: Optional[list[float]] = field(default=None, repr=False)
    speaker_id: Optional[str] = None; text: str = ""; asr_confidence: float = 0.0


@dataclass
class LocalSpeakerTracklet:
    local_tracklet_id: str; window_id: str; local_id: str; source_label: str
    segments: list[tuple[float, float]]; non_overlap_segments: list[tuple[float, float]]
    overlap_segments: list[tuple[float, float]]; duration: float; clean_duration: float
    quality_score: float = 0.0; quality_tier: str = "rejected"
    gallery: list[np.ndarray] = field(default_factory=list, repr=False)
    gallery_weights: list[float] = field(default_factory=list, repr=False)
    turn_ids: list[str] = field(default_factory=list); speaker_id: Optional[str] = None


def _split_window(start: float, end: float, asr: Sequence[dict[str, Any]], maximum: float) -> list[tuple[float, float]]:
    result = []
    cursor = start
    boundaries = sorted(float(row["end"]) for row in asr if start < float(row["end"]) < end)
    while end-cursor > maximum:
        target = cursor+maximum
        choices = [x for x in boundaries if cursor+maximum*.45 <= x <= target]
        cut = max(choices) if choices else target
        result.append((cursor, cut)); cursor = cut
    if end > cursor:
        result.append((cursor, end))
    return result


def build_speech_windows(vad: Sequence[tuple[float, float]], asr: Sequence[dict[str, Any]], duration: float,
                         padding: float=.5, merge_gap: float=.4, max_duration: float=15.) -> list[SpeechWindow]:
    """ASR refines/splits high-recall VAD windows but never deletes one."""
    merged = merge_intervals(vad, merge_gap)
    padded = merge_intervals((max(0., a-padding), min(duration, b+padding)) for a, b in merged)
    windows = []
    for outer_start, outer_end in padded:
        for start, end in _split_window(outer_start, outer_end, asr, max_duration):
            speech = [(max(start, a), min(end, b)) for a, b in merged if min(end, b) > max(start, a)]
            rows = [x for x in asr if float(x["end"]) > start and float(x["start"]) < end]
            weights = [max(0., min(end, float(x["end"]))-max(start, float(x["start"]))) for x in rows]
            total = sum(weights)
            confidence = sum(w*float(x.get("confidence", 0.)) for w, x in zip(weights, rows))/total if total else 0.
            windows.append(SpeechWindow(f"SW{len(windows)+1:04d}", round(start, 6), round(end, 6),
                round(min((a for a, _ in speech), default=start), 6), round(max((b for _, b in speech), default=end), 6),
                " ".join(str(x.get("text", "")).strip() for x in rows if str(x.get("text", "")).strip()), round(confidence, 6)))
    return windows


def build_local_tracklets(windows: Sequence[SpeechWindow], diarization: Sequence[tuple[float,float,str]]) -> tuple[list[LocalSpeakerTracklet], list[dict[str, Any]]]:
    output, cannot = [], []
    for window in windows:
        by_label: dict[str, list[tuple[float,float]]] = {}
        for start, end, label in diarization:
            begin, finish = max(start, window.start), min(end, window.end)
            if finish > begin:
                by_label.setdefault(label, []).append((begin, finish))
        local = []
        for index, (label, values) in enumerate(sorted(by_label.items())):
            values = merge_intervals(values)
            other = merge_intervals(pair for other_label, rows in by_label.items() if other_label != label for pair in rows)
            clean, overlap = subtract_intervals(values, other), intersect_intervals(values, other)
            local_id = f"LSPK{index:02d}"
            local.append(LocalSpeakerTracklet(f"{window.window_id}_{local_id}", window.window_id, local_id,
                label, values, clean, overlap, round(sum(b-a for a,b in values), 6), round(sum(b-a for a,b in clean), 6)))
        for i, left in enumerate(local):
            for right in local[i+1:]:
                regions = intersect_intervals(left.segments, right.segments)
                if regions:
                    cannot.append({"left": left.local_tracklet_id, "right": right.local_tracklet_id,
                                   "reason": "simultaneous_speech", "regions": [list(x) for x in regions]})
        output.extend(local)
    return output, cannot


def turn_quality(waveform: np.ndarray, duration: float, overlap: bool, min_duration: float,
                 max_duration: float) -> tuple[float,str,dict[str,float]]:
    rms = float(np.sqrt(np.mean(np.asarray(waveform, np.float64)**2)+1e-12))
    noise = float(np.percentile(np.abs(waveform), 20)) if len(waveform) else 0.
    snr = 20*math.log10((rms+1e-8)/(noise+1e-8)); clipping = float(np.mean(np.abs(waveform)>=.99)) if len(waveform) else 0.
    duration_score = min(duration/3.,1.)*min(max_duration/max(duration,1e-6),1.)
    score = .30*duration_score+.30*np.clip((snr-5)/25,0,1)+.20*np.clip(rms/.08,0,1)+.15*(not overlap)+.05*(1-min(clipping*50,1))
    tier = "rejected" if duration < min_duration or overlap or rms < .003 else "high" if score >= .60 and duration <= max_duration else "medium" if score >= .40 else "rejected"
    return round(float(score),6), tier, {"snr_db":round(snr,3),"rms":round(rms,6),"clipping_ratio":round(clipping,6),
        "duration_score":round(duration_score,6),"overlap_probability":1. if overlap else 0.,"music_contamination":-1.}


def align_asr(turn: SpeakerTurn, asr: Sequence[dict[str,Any]]) -> None:
    rows=[(max(0.,min(turn.end,float(x["end"]))-max(turn.start,float(x["start"]))),x) for x in asr]
    rows=[x for x in rows if x[0]>0]
    if not rows:return
    rows.sort(key=lambda x:float(x[1]["start"]));total=sum(x[0] for x in rows)
    turn.text=" ".join(str(x[1].get("text","")).strip() for x in rows if str(x[1].get("text","")).strip())
    turn.asr_confidence=round(sum(w*float(x.get("confidence",0.)) for w,x in rows)/total,6)


def extract_tracklet_evidence(tracklets: Sequence[LocalSpeakerTracklet], audio: np.ndarray, output: Path,
                              embedder: SpeakerEmbedder, asr: Sequence[dict[str,Any]], min_duration: float,
                              max_duration: float) -> list[SpeakerTurn]:
    turns=[]
    for tracklet in tracklets:
        tiers=[]
        for overlap, intervals in ((False,tracklet.non_overlap_segments),(True,tracklet.overlap_segments)):
            for start,end in intervals:
                waveform=audio[int(start*SAMPLE_RATE):int(end*SAMPLE_RATE)];duration=end-start
                score,tier,quality=turn_quality(waveform,duration,overlap,min_duration,max_duration)
                if overlap:tier="rejected"
                turn_id=f"TURN{len(turns)+1:05d}";path=output/"turn_audio"/f"{turn_id}.wav"
                slice_audio(audio,start,end,path);embedding=embedder.embed(waveform) if not overlap and tier!="rejected" else None
                turn=SpeakerTurn(turn_id,tracklet.local_tracklet_id,round(start,6),round(end,6),round(duration,6),str(path),score,tier,quality,overlap,
                                 embedding.tolist() if embedding is not None else None)
                align_asr(turn,asr);turns.append(turn);tracklet.turn_ids.append(turn_id);tiers.append(tier)
                if embedding is not None:tracklet.gallery.append(embedding);tracklet.gallery_weights.append(max(score,.05))
        tracklet.quality_score=round(max((x.quality_score for x in turns if x.local_tracklet_id==tracklet.local_tracklet_id),default=0.),6)
        tracklet.quality_tier="high" if "high" in tiers else "medium" if tracklet.gallery else "rejected"
    return turns


def gallery_metrics(left: Sequence[np.ndarray], right: Sequence[np.ndarray], top_k: int=3,
                    left_weights: Optional[Sequence[float]]=None, right_weights: Optional[Sequence[float]]=None,
                    support_similarity: float=.55) -> dict[str,Any]:
    lw=list(left_weights) if left_weights is not None else [1.]*len(left);rw=list(right_weights) if right_weights is not None else [1.]*len(right)
    pairs=sorted(((float(np.dot(unit(a),unit(b))),math.sqrt(max(lw[i],0.)*max(rw[j],0.))) for i,a in enumerate(left) for j,b in enumerate(right)),reverse=True)
    selected=pairs[:max(1,top_k)];den=sum(w for _,w in selected)
    mean=sum(x*w for x,w in selected)/den if den else None
    return {"max_similarity":round(pairs[0][0],6) if pairs else None,"topk_mean":round(float(mean),6) if mean is not None else None,
            "support_count":sum(x>=support_similarity for x,_ in pairs),"pair_count":len(pairs)}


_NON_LEXICAL_ASR_TOKENS = {
    "ah", "aah", "da", "dum", "hm", "hmm", "la", "mm", "mmm", "na",
    "oh", "ooh", "uh", "um", "whoa", "woo", "yeah",
}


def has_lexical_speech(text: str, minimum_characters: int = 3) -> bool:
    """Reject ASR that only describes humming/singing syllables.

    This is deliberately language-agnostic: unicode alphabetic runs count as words,
    while the small deny-list only removes common non-lexical vocalisations.
    """
    tokens = re.findall(r"[^\W\d_]+", str(text).lower(), flags=re.UNICODE)
    lexical = [token for token in tokens if token not in _NON_LEXICAL_ASR_TOKENS]
    # A short CJK utterance carries more lexical information than the same number
    # of Latin letters, so do not make an English-centric length gate.
    effective_characters = sum(
        2 if ord(character) > 127 else 1 for token in lexical for character in token
    )
    return effective_characters >= minimum_characters


def reference_identity_evidence(
    candidate: np.ndarray,
    own_support: Sequence[np.ndarray],
    competitors: Sequence[np.ndarray],
    admitted: Sequence[np.ndarray],
    *,
    minimum_similarity: float,
    minimum_support_count: int,
    minimum_margin: float,
    maximum_competitor_similarity: float,
    minimum_pairwise_similarity: float,
) -> dict[str, Any]:
    """Return an auditable, conservative admission decision for one voice asset."""
    vector = unit(candidate)
    own_scores = sorted((float(np.dot(vector, unit(x))) for x in own_support), reverse=True)
    competitor_scores = sorted((float(np.dot(vector, unit(x))) for x in competitors), reverse=True)
    admitted_scores = sorted((float(np.dot(vector, unit(x))) for x in admitted))
    own_best = own_scores[0] if own_scores else None
    competitor_best = competitor_scores[0] if competitor_scores else None
    margin = own_best - competitor_best if own_best is not None and competitor_best is not None else None
    support_count = sum(score >= minimum_similarity for score in own_scores)
    reasons = []
    if support_count < minimum_support_count:
        reasons.append("insufficient_independent_identity_support")
    if competitor_best is not None and competitor_best > maximum_competitor_similarity:
        reasons.append("too_similar_to_competing_speaker")
    if margin is not None and margin < minimum_margin:
        reasons.append("insufficient_competitor_margin")
    if admitted_scores and admitted_scores[0] < minimum_pairwise_similarity:
        reasons.append("inconsistent_with_admitted_references")
    return {
        "status": "PASS" if not reasons else "REJECT",
        "reasons": reasons,
        "own_best_similarity": round(own_best, 6) if own_best is not None else None,
        "own_support_count": support_count,
        "own_evidence_count": len(own_scores),
        "competitor_best_similarity": round(competitor_best, 6) if competitor_best is not None else None,
        "competitor_margin": round(margin, 6) if margin is not None else None,
        "minimum_similarity_to_admitted_reference": (
            round(admitted_scores[0], 6) if admitted_scores else None
        ),
    }


def associate_global_speakers(tracklets: Sequence[LocalSpeakerTracklet], cannot_links: Sequence[dict[str,Any]],
                              threshold: float, possible_threshold: float, min_support: int,
                              competitor_margin: float, top_k: int) -> tuple[list[list[int]],list[dict[str,Any]],list[dict[str,Any]]]:
    forbidden={frozenset((x["left"],x["right"])) for x in cannot_links};groups=[];must=[];possible=[]
    def rank(index):
        values=[]
        for gi,group in enumerate(groups):
            if any(frozenset((tracklets[index].local_tracklet_id,tracklets[j].local_tracklet_id)) in forbidden for j in group):continue
            gallery=[v for j in group for v in tracklets[j].gallery];weights=[v for j in group for v in tracklets[j].gallery_weights]
            metrics=gallery_metrics(tracklets[index].gallery,gallery,top_k,tracklets[index].gallery_weights,weights)
            if metrics["topk_mean"] is not None:values.append((float(metrics["topk_mean"]),gi,metrics))
        return sorted(values,reverse=True,key=lambda x:x[0])
    seeds=sorted((i for i,x in enumerate(tracklets) if x.quality_tier=="high" and x.gallery),key=lambda i:(-tracklets[i].quality_score,tracklets[i].segments[0][0]))
    assigned=set()
    for index in seeds:
        ranked=rank(index);best=ranked[0] if ranked else None;second=ranked[1][0] if len(ranked)>1 else -1.
        if best and best[0]>=threshold and best[2]["support_count"]>=min_support and best[0]-second>=competitor_margin:
            groups[best[1]].append(index);assigned.add(index);must.append({"relation":"MUST_LINK","tracklet":tracklets[index].local_tracklet_id,"group_index":best[1],"competitor_margin":round(best[0]-second,6),**best[2]})
        else:
            if best and best[0]>=possible_threshold:possible.append({"tracklet":tracklets[index].local_tracklet_id,"group_index":best[1],"reason":"insufficient_score_support_or_margin",**best[2]})
            groups.append([index]);assigned.add(index)
    for index,x in enumerate(tracklets):
        if index in assigned or not x.gallery:continue
        ranked=rank(index);best=ranked[0] if ranked else None;second=ranked[1][0] if len(ranked)>1 else -1.
        if best and best[0]>=threshold and best[2]["support_count"]>=min_support and best[0]-second>=competitor_margin:
            groups[best[1]].append(index);assigned.add(index);must.append({"relation":"MUST_LINK","tracklet":x.local_tracklet_id,"group_index":best[1],"competitor_margin":round(best[0]-second,6),**best[2]})
        elif best and best[0]>=possible_threshold:possible.append({"tracklet":x.local_tracklet_id,"group_index":best[1],"reason":"medium_or_ambiguous_evidence",**best[2]})
    return groups,must,possible


def build_multilabel_timeline(tracklets: Sequence[LocalSpeakerTracklet], assignments: dict[str,str]) -> list[dict[str,Any]]:
    segments=[(a,b,assignments[x.local_tracklet_id]) for x in tracklets if x.local_tracklet_id in assignments for a,b in x.segments]
    boundaries=sorted({v for a,b,_ in segments for v in (a,b)});output=[]
    for start,end in zip(boundaries,boundaries[1:]):
        active=sorted({speaker for a,b,speaker in segments if a<end and b>start})
        if not active:continue
        if output and output[-1]["active_speakers"]==active and abs(output[-1]["end"]-start)<1e-6:output[-1]["end"]=round(end,6)
        else:output.append({"start":round(start,6),"end":round(end,6),"active_speakers":active})
    return output


def evaluate(turns: Sequence[SpeakerTurn], ground_truth: Optional[str]) -> Optional[dict[str,Any]]:
    if not ground_truth:return None
    truth=json.loads(Path(ground_truth).read_text(encoding="utf-8"));truth=truth.get("turn_to_speaker",truth)
    rows=[x for x in turns if x.turn_id in truth and x.speaker_id];tp=fp=fn=0
    for i,a in enumerate(rows):
        for b in rows[i+1:]:
            gt=truth[a.turn_id]==truth[b.turn_id];pred=a.speaker_id==b.speaker_id;tp+=gt and pred;fp+=not gt and pred;fn+=gt and not pred
    return {"labeled_turns":len(rows),"pairwise_precision":round(tp/max(tp+fp,1),6),"pairwise_recall":round(tp/max(tp+fn,1),6)}


def load_foreground_reference_audio(report_path: Path, expected_duration: float) -> tuple[np.ndarray, dict[str, Any]]:
    report_path = Path(report_path).expanduser().resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    files = report.get("files") or {}
    source = files.get(f"foreground_{SAMPLE_RATE}")
    if not source:
        raise ValueError(f"Foreground report has no foreground_{SAMPLE_RATE} stem: {report_path}")
    waveform, sample_rate = sf.read(str(Path(source).expanduser().resolve()), dtype="float32")
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"Foreground stem must be {SAMPLE_RATE} Hz, got {sample_rate}: {source}")
    duration = len(waveform) / float(sample_rate)
    if abs(duration - expected_duration) > .25:
        raise ValueError(
            f"Foreground stem duration mismatch: expected {expected_duration:.3f}s, got {duration:.3f}s"
        )
    return np.asarray(waveform, np.float32), report


def run(args: argparse.Namespace) -> dict[str,Any]:
    source=Path(args.input).expanduser().resolve();output=Path(args.output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):raise ValueError("--output must be empty; Stage5 V2 never overwrites an existing registry")
    output.mkdir(parents=True,exist_ok=True);audio_path=output/"audio.wav";timeline=extract_audio(source,audio_path)
    audio,_=sf.read(str(audio_path),dtype="float32")
    if args.asr_json:
        asr=_rows(json.loads(Path(args.asr_json).expanduser().read_text(encoding="utf-8")))
    else:
        asr=[] if args.disable_asr else Transcriber(Path(args.asr_model).expanduser().resolve(),args.device,args.compute_type)(audio_path,args.language)
    diar_raw=json.loads(Path(args.diarization_json).read_text(encoding="utf-8")) if args.diarization_json else None
    vad_raw=json.loads(Path(args.vad_json).read_text(encoding="utf-8")) if args.vad_json else None
    diar=_seconds_if_needed(normalize_diarization(diar_raw),timeline["duration"]) if diar_raw is not None else []
    if vad_raw is not None:vad=_seconds_if_needed(normalize_vad(vad_raw),timeline["duration"]);vad_source="vad_json"
    elif args.vad_model:vad=SpeechDetector(Path(args.vad_model).expanduser().resolve())(audio_path);vad_source="vad_model"
    else:
        if not diar:diar=Diarizer(Path(args.diarization_model).expanduser().resolve())(audio_path,args.oracle_speakers)
        vad=merge_intervals((a,b) for a,b,_ in diar);vad_source="diarization_speech_activity_fallback"
    windows=build_speech_windows(vad,asr,timeline["duration"],args.vad_padding,args.merge_gap_seconds,args.max_window_duration)
    if not diar:
        model=Diarizer(Path(args.diarization_model).expanduser().resolve())
        for window in windows:
            clip=output/"speech_windows"/f"{window.window_id}.wav";slice_audio(audio,window.start,window.end,clip)
            diar.extend((a+window.start,b+window.start,f"{window.window_id}:{label}") for a,b,label in model(clip,args.oracle_speakers))
    else:
        for window in windows:slice_audio(audio,window.start,window.end,output/"speech_windows"/f"{window.window_id}.wav")
    tracklets,cannot=build_local_tracklets(windows,diar);embedder=SpeakerEmbedder(Path(args.embedding_model).expanduser().resolve())
    turns=extract_tracklet_evidence(tracklets,audio,output,embedder,asr,args.min_turn_duration,args.max_turn_duration)
    groups,must,possible=associate_global_speakers(tracklets,cannot,args.cluster_threshold,args.possible_link_threshold,args.min_support,args.competitor_margin,args.gallery_top_k)
    reference_source_audio = audio
    foreground_report = None
    reference_source = "original_non_overlap"
    if args.foreground_report:
        reference_source_audio, foreground_report = load_foreground_reference_audio(
            Path(args.foreground_report), timeline["duration"]
        )
        reference_source = "enhanced_foreground_non_overlap"

    group_data=[]
    for members in groups:
        ids={tracklets[i].local_tracklet_id for i in members}
        member_turns=[x for x in turns if x.local_tracklet_id in ids]
        gallery=[v for i in members for v in tracklets[i].gallery]
        weights=[v for i in members for v in tracklets[i].gallery_weights]
        group_data.append({"members":members,"ids":ids,"turns":member_turns,"gallery":gallery,"weights":weights})

    speakers=[];assignments={}
    for number,data in enumerate(group_data,1):
        members=data["members"]
        speaker_id=f"SPK{number:03d}";root=output/"speaker_registry"/speaker_id;root.mkdir(parents=True,exist_ok=True)
        gallery=data["gallery"];weights=data["weights"]
        prototype=unit(np.average(np.asarray(gallery),axis=0,weights=np.asarray(weights)));np.save(root/"embedding_gallery.npy",np.asarray(gallery,np.float32));np.save(root/"prototype.npy",prototype.astype(np.float32))
        ids=data["ids"];member_turns=data["turns"]
        for i in members:tracklets[i].speaker_id=speaker_id;assignments[tracklets[i].local_tracklet_id]=speaker_id
        for turn in member_turns:turn.speaker_id=speaker_id
        ranked=sorted((x for x in member_turns if x.quality_tier in {"high","medium"} and not x.overlap and x.duration>=args.min_reference_duration),key=lambda x:-x.quality_score)
        competitors=[v for other_number,other in enumerate(group_data,1) if other_number!=number for v in other["gallery"]]
        refs=[];reference_candidates=[];admitted_vectors=[]
        for turn in ranked:
            reasons=[]
            if not args.disable_asr and not has_lexical_speech(turn.text,args.min_reference_asr_characters):
                reasons.append("no_lexical_speech")
            foreground_gate=None
            if foreground_report is not None:
                from audio_foreground_separator import foreground_gate_for_turn
                foreground_gate=foreground_gate_for_turn(
                    foreground_report,turn.start,turn.end,args.min_reference_foreground_coverage
                )
                if not foreground_gate["has_foreground_speech"]:
                    reasons.append("insufficient_foreground_speech")
                contamination=foreground_gate.get("music_contamination")
                if contamination is not None and contamination>args.max_reference_music_contamination:
                    reasons.append("excessive_music_or_background_contamination")
            elif args.require_foreground_references:
                reasons.append("foreground_report_required")

            end=min(turn.end,turn.start+args.max_reference_duration)
            waveform=reference_source_audio[int(round(turn.start*SAMPLE_RATE)):int(round(end*SAMPLE_RATE))]
            output_quality_score,output_quality_tier,output_quality=turn_quality(
                waveform,(len(waveform)/SAMPLE_RATE),False,args.min_reference_duration,args.max_reference_duration
            )
            if output_quality_tier=="rejected":
                reasons.append("reference_waveform_quality_rejected")
            candidate_embedding=embedder.embed(waveform)
            own_support=[np.asarray(other.embedding,np.float32) for other in member_turns
                         if other.turn_id!=turn.turn_id and other.embedding is not None]
            if candidate_embedding is None:
                identity={"status":"REJECT","reasons":["missing_reference_embedding"]}
            else:
                identity=reference_identity_evidence(
                    candidate_embedding,own_support,competitors,admitted_vectors,
                    minimum_similarity=args.min_reference_identity_similarity,
                    minimum_support_count=args.min_reference_identity_support,
                    minimum_margin=args.min_reference_competitor_margin,
                    maximum_competitor_similarity=args.max_reference_competitor_similarity,
                    minimum_pairwise_similarity=args.min_reference_pairwise_similarity,
                )
            reasons.extend(identity.get("reasons",[]))
            if not reasons and len(refs)>=args.voice_top_k:
                reasons.append("voice_top_k_limit")
            admitted=not reasons
            destination=None
            if admitted:
                destination=root/"references"/f"ref_{len(refs)+1:02d}.wav"
                slice_audio(reference_source_audio,turn.start,end,destination)
                refs.append(str(destination));admitted_vectors.append(candidate_embedding)
            reference_candidates.append({
                "turn_id":turn.turn_id,"status":"ADMITTED" if admitted else "REJECTED",
                "reasons":reasons,"start":turn.start,"end":round(end,6),"duration":round(end-turn.start,6),
                "text":turn.text,"source":reference_source,"audio_path":str(destination) if destination else None,
                "foreground_gate":foreground_gate,"identity_gate":identity,
                "output_quality_score":output_quality_score,"output_quality_tier":output_quality_tier,
                "output_quality":output_quality,
            })
        record={"speaker_id":speaker_id,"character_id":None,"local_tracklets":sorted(ids),"embedding_count":len(gallery),"embedding_gallery":str(root/"embedding_gallery.npy"),"prototype":str(root/"prototype.npy"),"turns":[x.turn_id for x in member_turns],"reference_audio":refs,
                # Compatibility alias consumed by the current Stage6 builder.
                "representative_audio":refs,"reference_status":"READY" if refs else "NO_SAFE_REFERENCE",
                "reference_source":reference_source,"reference_candidates":reference_candidates,
                "confidence":round(float(np.mean([tracklets[i].quality_score for i in members])),6)}
        write_json(root/"speaker.json",record);write_json(root/"turns.json",{"turns":[{k:v for k,v in asdict(x).items() if k!="embedding"} for x in member_turns]});speakers.append(record)
    # Unknown local speakers stay visible to overlap routing even when there is
    # not enough clean evidence to assign a global SPK identity.
    timeline_assignments={x.local_tracklet_id:assignments.get(x.local_tracklet_id,f"UNKNOWN::{x.local_tracklet_id}") for x in tracklets}
    global_timeline=build_multilabel_timeline(tracklets,timeline_assignments)
    overlaps=[{**row,"overlap_id":f"OVL{i:05d}"} for i,row in enumerate(
        (x for x in global_timeline if len(x["active_speakers"])>=2 and x["end"]-x["start"]>=args.min_overlap_duration),1)]
    public_turns=[{k:v for k,v in asdict(x).items() if k!="embedding"} for x in turns]
    public_tracklets=[{k:v for k,v in asdict(x).items() if k not in {"gallery","gallery_weights"}} for x in tracklets]
    unknown=[row for row in public_turns if row["speaker_id"] is None]
    result={"schema_version":SCHEMA_VERSION,"status":"ok","timeline":timeline,"speech_window_count":len(windows),"local_tracklet_count":len(tracklets),"speaker_count":len(speakers),"turn_count":len(turns),"unknown_turn_count":len(unknown),"rejected_asset_turn_count":sum(x.quality_tier=="rejected" for x in turns),"reference_audio_count":sum(len(x["reference_audio"]) for x in speakers),"safe_reference_speaker_count":sum(bool(x["reference_audio"]) for x in speakers),"quality_stats":{tier:sum(x.quality_tier==tier for x in turns) for tier in ("high","medium","rejected")},"speakers":speakers,"turns":public_turns,"local_tracklets":public_tracklets,"global_speaker_timeline":global_timeline,"overlap_regions":overlaps,"must_links":must,"possible_links":possible,"cannot_links":cannot,"unknown_turns":unknown,"asr_timeline":asr,"metadata":{"vad_source":vad_source,"identity_encoder":"CAM++","diarization_labels_are_local":True,"multi_label_timeline":True,"asr_can_delete_vad_windows":False,"reference_source":reference_source,"foreground_report":str(Path(args.foreground_report).expanduser().resolve()) if args.foreground_report else None},"evaluation":evaluate(turns,args.ground_truth)}
    files={"timeline.json":timeline,"speech_windows.json":{"windows":[asdict(x) for x in windows]},"local_speaker_tracklets.json":{"tracklets":public_tracklets},"speaker_turns.json":{"turns":public_turns},"asr_timeline.json":{"segments":asr},"global_speaker_timeline.json":{"segments":global_timeline},"overlap_regions.json":{"overlaps":overlaps},"must_links.json":{"links":must},"possible_links.json":{"links":possible},"cannot_links.json":{"links":cannot},"unknown_turns.json":{"turns":unknown},"registry.json":result}
    for name,value in files.items():write_json(output/name,value)
    return result


def parser() -> argparse.ArgumentParser:
    p=argparse.ArgumentParser(description="Stage 5 V2 overlap-aware global speaker registry")
    p.add_argument("--input",required=True);p.add_argument("--output",required=True);p.add_argument("--vad-model");p.add_argument("--vad-json")
    p.add_argument("--diarization-model",default="local_models/speaker/campplus_diarization");p.add_argument("--diarization-json")
    p.add_argument("--embedding-model",default="local_models/speaker/campplus_sv");p.add_argument("--asr-model",default="local_models/faster-whisper-large-v3");p.add_argument("--asr-json",help="Reuse a previously generated Whisper timeline")
    p.add_argument("--disable-asr",action="store_true");p.add_argument("--language");p.add_argument("--device",default="cuda");p.add_argument("--compute-type",default="float16");p.add_argument("--oracle-speakers",type=int)
    p.add_argument("--vad-padding",type=float,default=.5);p.add_argument("--merge-gap-seconds",type=float,default=.4);p.add_argument("--max-window-duration",type=float,default=15.);p.add_argument("--min-overlap-duration",type=float,default=.2)
    p.add_argument("--min-turn-duration",type=float,default=.8);p.add_argument("--max-turn-duration",type=float,default=15.);p.add_argument("--cluster-threshold",type=float,default=.60);p.add_argument("--possible-link-threshold",type=float,default=.50);p.add_argument("--competitor-margin",type=float,default=.05);p.add_argument("--min-support",type=int,default=1);p.add_argument("--gallery-top-k",type=int,default=3);p.add_argument("--voice-top-k",type=int,default=3);p.add_argument("--ground-truth");p.add_argument("--min-reference-duration",type=float,default=2.);p.add_argument("--max-reference-duration",type=float,default=10.)
    p.add_argument("--foreground-report",help="audio_foreground_separator foreground_report.json; references are cut from its 16 kHz foreground stem")
    p.add_argument("--require-foreground-references",action="store_true",help="Emit no formal reference when --foreground-report is absent")
    p.add_argument("--min-reference-foreground-coverage",type=float,default=.5)
    p.add_argument("--max-reference-music-contamination",type=float,default=.5)
    p.add_argument("--min-reference-asr-characters",type=int,default=3)
    p.add_argument("--min-reference-identity-similarity",type=float,default=.45)
    p.add_argument("--min-reference-identity-support",type=int,default=1)
    p.add_argument("--min-reference-competitor-margin",type=float,default=.15)
    p.add_argument("--max-reference-competitor-similarity",type=float,default=.55)
    p.add_argument("--min-reference-pairwise-similarity",type=float,default=.5)
    return p


def main()->int:
    args=parser().parse_args()
    if args.possible_link_threshold>args.cluster_threshold:raise SystemExit("--possible-link-threshold must not exceed --cluster-threshold")
    probabilities=(args.min_reference_foreground_coverage,args.max_reference_music_contamination)
    if any(value < 0. or value > 1. for value in probabilities):
        raise SystemExit("reference coverage/contamination thresholds must be within [0, 1]")
    similarities=(args.min_reference_identity_similarity,args.min_reference_competitor_margin,
                  args.max_reference_competitor_similarity,args.min_reference_pairwise_similarity)
    if any(value < -1. or value > 1. for value in similarities):
        raise SystemExit("reference similarity/margin thresholds must be within [-1, 1]")
    if args.min_reference_identity_support < 0 or args.min_reference_asr_characters < 0:
        raise SystemExit("reference support and ASR character thresholds must be non-negative")
    result=run(args);print(json.dumps({k:result[k] for k in ("status","speech_window_count","local_tracklet_count","speaker_count","turn_count","unknown_turn_count","reference_audio_count","safe_reference_speaker_count","quality_stats")},ensure_ascii=False));return 0


if __name__=="__main__":raise SystemExit(main())
