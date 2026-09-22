#!/usr/bin/env python3
"""Generate a bounded set of I2I review assets from a mainline workspace.

The standalone restoration CLIs accept explicitly selected images.  This
adapter performs that selection for the current ``run_full_pipeline.py``
layout and records every selection in ``i2i_test/manifest.json``.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable

try:
    from .restore_character_reference import restore_character_reference
    from .restore_key_object_reference import restore_key_object_reference
    from .restore_scene_reference import restore_scene_reference
    from .uniapi_gemini_image_client import (
        DEFAULT_GEMINI_BASE_URL,
        DEFAULT_GEMINI_IMAGE_MODEL,
        UniAPIGeminiImageClient,
    )
except ImportError:  # direct script execution
    from restore_character_reference import restore_character_reference
    from restore_key_object_reference import restore_key_object_reference
    from restore_scene_reference import restore_scene_reference
    from uniapi_gemini_image_client import (
        DEFAULT_GEMINI_BASE_URL,
        DEFAULT_GEMINI_IMAGE_MODEL,
        UniAPIGeminiImageClient,
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _load_env(path: Path) -> None:
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))
    # The original local file used UNIAPI_KEY while the public CLIs document
    # UNIAPI_API_KEY.  Accept it as a compatibility alias without printing it.
    if not os.getenv("UNIAPI_API_KEY") and os.getenv("UNIAPI_KEY"):
        os.environ["UNIAPI_API_KEY"] = os.environ["UNIAPI_KEY"]


def _safe_id(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", value).strip("_") or "asset"


def _take(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    return rows if limit == 0 else rows[:limit]


def _character_plans(workspace: Path, limit: int, max_images: int) -> list[dict[str, Any]]:
    plans: list[dict[str, Any]] = []
    for metadata_path in sorted((workspace / "characters").glob("*/*/role_character.json")):
        metadata = _read_json(metadata_path)
        visual = metadata_path.parent / "visual"
        images = (
            sorted((visual / "face").glob("*"))
            + sorted((visual / "body").glob("*"))
            + sorted((visual / "rgba").glob("*"))
        )
        images = [p for p in images if p.is_file()][:max_images]
        if not images:
            continue
        scope = str(metadata.get("semantic_scope_id") or metadata_path.parents[1].name)
        subject = str(metadata.get("subject_id") or metadata_path.parent.name)
        plans.append({
            "kind": "character",
            "task_id": f"{scope}_{subject}_reference",
            "images": images,
            "description": (
                f"semantic character {scope}:{subject}; "
                f"mean alignment confidence {metadata.get('mean_alignment_confidence', 'unknown')}"
            ),
            "prompt": "Preserve the same identity and wardrobe; reconstruct one clean full-body production reference.",
            "rank": (
                int(metadata.get("tracklet_count") or 0),
                float(metadata.get("mean_alignment_confidence") or 0.0),
            ),
        })
    plans.sort(key=lambda row: (row["rank"], row["task_id"]), reverse=True)
    return _take(plans, limit)


def _resolve_asset(root: Path, raw: object) -> Path | None:
    if not raw:
        return None
    path = Path(str(raw))
    if path.is_file():
        return path.resolve()
    candidate = (root / path).resolve()
    return candidate if candidate.is_file() else None


def _object_plans(workspace: Path, limit: int, max_images: int) -> list[dict[str, Any]]:
    manifest = workspace / "key_object" / "grounding" / "reference_sets.json"
    if not manifest.is_file():
        return []
    payload = _read_json(manifest)
    recorded_root = Path(str(payload.get("grounding_dir") or ""))
    root = recorded_root if recorded_root.is_dir() else manifest.parent
    plans: list[dict[str, Any]] = []
    for row in payload.get("candidates") or []:
        if not isinstance(row, dict):
            continue
        images: list[Path] = []
        fallback_crops: list[Path] = []
        for reference in row.get("references") or []:
            asset = reference.get("asset") if isinstance(reference, dict) else None
            if not isinstance(asset, dict):
                continue
            rgba = _resolve_asset(root, asset.get("rgba"))
            crop = _resolve_asset(root, asset.get("crop"))
            selected = rgba or crop
            if selected and selected not in images:
                images.append(selected)
            if crop and crop not in fallback_crops:
                fallback_crops.append(crop)
            if len(images) >= max_images:
                break
        if not images:
            continue
        task_id = str(row.get("candidate_id") or f"object_{len(plans) + 1:04d}")
        description = str(row.get("canonical_name") or "key object")
        attributes = row.get("visual_attributes")
        if attributes:
            description += "; " + json.dumps(attributes, ensure_ascii=False, sort_keys=True)
        plans.append({
            "kind": "object",
            "task_id": task_id,
            "images": images,
            # Tiny or heavily occluded objects occasionally produce an empty
            # candidates[] response when several references are combined.
            # Keep the largest original crop for one bounded retry.
            "fallback_images": sorted(
                fallback_crops, key=lambda path: path.stat().st_size, reverse=True
            )[:1],
            "description": description,
            "prompt": "Preserve the observed geometry, materials, colors and story-relevant state.",
            "rank": (len(images), task_id),
        })
    plans.sort(key=lambda row: row["rank"], reverse=True)
    return _take(plans, limit)


def _scene_plans(workspace: Path, limit: int, max_images: int) -> list[dict[str, Any]]:
    plans: list[dict[str, Any]] = []
    for manifest in sorted((workspace / "scene").glob("*/scene_index.json")):
        payload = _read_json(manifest)
        recorded_root = Path(str(payload.get("output_dir") or ""))
        root = recorded_root if recorded_root.is_dir() else manifest.parent
        for row in payload.get("scenes") or []:
            if not isinstance(row, dict):
                continue
            scene_name = str(row.get("scene_name") or f"scene_{len(plans) + 1:04d}")
            scene_root = root / scene_name
            images: list[Path] = []
            for selected in row.get("selected") or []:
                asset = selected.get("asset") if isinstance(selected, dict) else None
                image = _resolve_asset(scene_root, asset)
                if image and image not in images:
                    images.append(image)
                if len(images) >= max_images:
                    break
            if not images:
                continue
            plans.append({
                "kind": "scene",
                "task_id": scene_name,
                "images": images,
                "description": (
                    f"scene cluster {row.get('scene_cluster_id', 'unknown')}; "
                    f"importance {row.get('importance_score', 'unknown')}"
                ),
                "prompt": "Remove people and transient occluders while preserving the exact location layout and lighting.",
                "rank": (float(row.get("importance_score") or 0.0), scene_name),
            })
    plans.sort(key=lambda row: row["rank"], reverse=True)
    return _take(plans, limit)


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--kinds", nargs="+", choices=("character", "object", "scene"),
                        default=["character", "object", "scene"])
    parser.add_argument("--max-characters", type=int, default=3, help="0 means all")
    parser.add_argument("--max-objects", type=int, default=3, help="0 means all")
    parser.add_argument("--max-scenes", type=int, default=3, help="0 means all")
    parser.add_argument("--max-images", type=int, default=5)
    parser.add_argument("--model", default=os.getenv("UNIAPI_GEMINI_IMAGE_MODEL", DEFAULT_GEMINI_IMAGE_MODEL))
    parser.add_argument("--image-size", default="auto", choices=("auto", "1K", "2K", "4K"))
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--env-file", type=Path, default=Path(__file__).with_name(".env"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if min(args.max_characters, args.max_objects, args.max_scenes) < 0 or args.max_images < 1:
        parser.error("limits must be non-negative and --max-images must be positive")
    workspace = args.workspace.expanduser().resolve()
    output_root = (args.output_root or workspace / "i2i_test").expanduser().resolve()
    model_dir = output_root / _safe_id(args.model)

    plans: list[dict[str, Any]] = []
    if "character" in args.kinds:
        plans.extend(_character_plans(workspace, args.max_characters, args.max_images))
    if "object" in args.kinds:
        plans.extend(_object_plans(workspace, args.max_objects, args.max_images))
    if "scene" in args.kinds:
        plans.extend(_scene_plans(workspace, args.max_scenes, args.max_images))

    public_plans = [{
        "kind": row["kind"], "task_id": row["task_id"],
        "images": [str(p) for p in row["images"]],
        **({"fallback_images": [str(p) for p in row.get("fallback_images", [])]}
           if row.get("fallback_images") else {}),
        "description": row["description"],
        "output": str(model_dir / f"{_safe_id(row['kind'] + '_' + row['task_id'])}.png"),
    } for row in plans]
    if args.dry_run:
        print(json.dumps({"workspace": str(workspace), "model": args.model,
                          "tasks": public_plans}, ensure_ascii=False, indent=2))
        return 0

    _load_env(args.env_file.expanduser().resolve())
    api_key = os.getenv("UNIAPI_API_KEY")
    if not api_key:
        print("error: UNIAPI_API_KEY (or legacy UNIAPI_KEY in --env-file) is not set", file=sys.stderr)
        return 2
    client = UniAPIGeminiImageClient(
        api_key,
        base_url=os.getenv("UNIAPI_GEMINI_BASE_URL", DEFAULT_GEMINI_BASE_URL),
        timeout_seconds=args.timeout,
        max_retries=args.max_retries,
    )
    runners: dict[str, Callable[..., Any]] = {
        "character": restore_character_reference,
        "object": restore_key_object_reference,
        "scene": restore_scene_reference,
    }
    results: list[dict[str, Any]] = []
    failures = 0
    for row in plans:
        output = model_dir / f"{_safe_id(row['kind'] + '_' + row['task_id'])}.png"
        result_row = {
            "kind": row["kind"], "task_id": row["task_id"],
            "model": args.model, "inputs": [str(p) for p in row["images"]],
            "output": str(output), "description": row["description"],
        }
        if output.is_file() and not args.overwrite:
            result_row["status"] = "skipped_existing"
            results.append(result_row)
            continue
        try:
            try:
                result = runners[row["kind"]](
                    client,
                    image_paths=row["images"],
                    prompt=row["prompt"],
                    asset_description=row["description"],
                    output_path=output,
                    overwrite=args.overwrite,
                    model=args.model,
                    image_size=args.image_size,
                    temperature=args.temperature,
                )
            except Exception as primary_exc:
                fallback_images = row.get("fallback_images") or []
                if (row["kind"] != "object" or not fallback_images
                        or "no candidates" not in str(primary_exc).lower()):
                    raise
                print(
                    f"[i2i] retry object {row['task_id']} with single crop after empty candidates",
                    flush=True,
                )
                result = runners[row["kind"]](
                    client,
                    image_paths=fallback_images,
                    prompt=row["prompt"],
                    asset_description=row["description"],
                    output_path=output,
                    overwrite=args.overwrite,
                    model=args.model,
                    image_size=args.image_size,
                    temperature=args.temperature,
                )
                result_row.update({
                    "fallback_used": "single_largest_crop",
                    "fallback_inputs": [str(path) for path in fallback_images],
                    "primary_error": f"{type(primary_exc).__name__}: {primary_exc}"[:1000],
                })
            result_row.update({
                "status": "succeeded",
                "sha256": result.saved_images[0].sha256,
                "usage": result.response.get("usageMetadata"),
            })
            output.with_suffix(output.suffix + ".json").write_text(
                json.dumps(result_row, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"[i2i] {row['kind']} {row['task_id']} -> {output}", flush=True)
        except Exception as exc:  # task boundary; continue independent assets
            failures += 1
            result_row.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"[:2000]})
            print(f"[i2i] failed {row['kind']} {row['task_id']}: {result_row['error']}",
                  file=sys.stderr, flush=True)
        results.append(result_row)

    manifest = {
        "schema_version": "pipeline-i2i-review-v1",
        "workspace": str(workspace),
        "model": args.model,
        "task_count": len(results),
        "succeeded_count": sum(row["status"] in {"succeeded", "skipped_existing"} for row in results),
        "failed_count": failures,
        "tasks": results,
    }
    _write_manifest(output_root / "manifest.json", manifest)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
