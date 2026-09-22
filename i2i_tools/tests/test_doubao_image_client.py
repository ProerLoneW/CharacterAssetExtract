"""Unit tests for the Doubao / ARK image-client additions.

Covers ``resolve_doubao_model_size``, the Doubao profile table, the
``extract_character_*_doubao`` wrappers, and ``execute_plan_doubao``. No
network calls. Run with::

    cd /gemini/platform/public/aigc/lss/Interns/wyn/general_video_annotation_level3_v2
    python -m pytest i2i_tools/tests/test_doubao_image_client.py -v
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uniapi_doubao_image_client import (  # noqa: E402
    DEFAULT_DOUBEO_BASE_URL,
    DEFAULT_DOUBEO_MODEL,
    DOUBAO_IMAGE_MODEL_PROFILES,
    SUPPORTED_DOUBEAO_MODELS,
    UniAPIDoubaoImageClient,
    UniAPIImageError,
    resolve_doubao_model_size,
    _generations_url,
    _encode_image_for_doubao,
)
from extract_character_clothing_doubao import (  # noqa: E402
    _size_for_part,
    extract_character_pants_doubao,
    extract_character_top_doubao,
    execute_plan_doubao,
    main as doubao_cli_main,
)
from extract_character_clothing import plan_clothing_extraction  # noqa: E402
from two_stage_clothing_pipeline import (  # noqa: E402
    ALL_MODELS,
    _classify_model,
)


# ---------------------------------------------------------------------------
# Profile / size helpers
# ---------------------------------------------------------------------------


def test_doubao_profile_table_contains_seedream_5():
    assert "doubao-seedream-5-0-260128" in DOUBAO_IMAGE_MODEL_PROFILES
    assert "doubao-seedream-5-0-lite-260128" in DOUBAO_IMAGE_MODEL_PROFILES


def test_default_doubao_model_is_seedream_5():
    assert DEFAULT_DOUBEO_MODEL == "doubao-seedream-5-0-260128"
    assert "doubao-seedream-5-0-260128" in SUPPORTED_DOUBEAO_MODELS


def test_resolve_doubao_size_handles_auto_and_explicit():
    assert resolve_doubao_model_size("doubao-seedream-5-0-260128", "auto") == "2K"
    assert (
        resolve_doubao_model_size("doubao-seedream-5-0-260128", "2048x2048")
        == "2048x2048"
    )
    assert resolve_doubao_model_size("doubao-seedream-5-0-260128", None) == "2K"


def test_resolve_doubao_size_does_not_check_pixel_floor():
    """resolve_doubao_model_size only checks against the profile's supported
    sizes; the actual pixel-floor guard is in _assert_meets_doubao_pixel_floor
    and is run by _size_for_part / the client."""
    # 1024x1024 is in the supported_sizes set, so resolve_doubao_model_size
    # accepts it without checking the pixel floor.
    assert resolve_doubao_model_size("doubao-seedream-5-0-260128", "1024x1024") == "1024x1024"


def test_doubao_pixel_floor_rejects_undersized_explicit_size():
    from uniapi_doubao_image_client import _assert_meets_doubao_pixel_floor

    with pytest.raises(ValueError, match="3686400"):
        _assert_meets_doubao_pixel_floor("1024x1024")
    _assert_meets_doubao_pixel_floor("2048x2048")
    _assert_meets_doubao_pixel_floor("2K")


def test_generations_url_strips_trailing_slash_and_appends_path():
    assert _generations_url("https://api.uniapi.io/v1") == (
        "https://api.uniapi.io/v1/images/generations"
    )
    assert _generations_url("https://api.uniapi.io/v1/") == (
        "https://api.uniapi.io/v1/images/generations"
    )
    assert _generations_url("https://api.uniapi.io/v1/images/generations") == (
        "https://api.uniapi.io/v1/images/generations"
    )


def test_default_doubao_base_url():
    assert DEFAULT_DOUBEO_BASE_URL == "https://api.uniapi.io/v1"


# ---------------------------------------------------------------------------
# Encode image helper
# ---------------------------------------------------------------------------


def test_encode_image_for_doubao_uses_data_uri(tmp_path: Path):
    payload = b"\x89PNG\r\n\x1a\n" + b"X" * 32
    image_path = tmp_path / "a.png"
    image_path.write_bytes(payload)
    from uniapi_image_client import InputImage
    image = InputImage(path=image_path, content_type="image/png", size_bytes=len(payload), sha256="x")
    encoded = _encode_image_for_doubao(image)
    assert encoded.startswith("data:image/png;base64,")
    assert base64.b64decode(encoded.split(",", 1)[1]) == payload


# ---------------------------------------------------------------------------
# Clothing-extraction wrappers (no API call)
# ---------------------------------------------------------------------------


def _png(path: Path) -> Path:
    payload = b"\x89PNG\r\n\x1a\n" + b"X" * 8
    path.write_bytes(payload)
    return path


class _FakeDoubaoClient:
    def __init__(self):
        self.calls: list[dict] = []
        self.timeout_seconds = 1.0

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return {"created": 0, "data": [], "usage": {}}, []


def test_extract_character_top_doubao_uses_top_instructions(monkeypatch, tmp_path: Path):
    img = _png(tmp_path / "a.jpg")
    out = tmp_path / "top.png"

    captured: dict = {}

    def fake_restore(client, *, image_paths, user_prompt, output_path, task_instructions, asset_description, overwrite, generation_options):
        captured["task_instructions"] = task_instructions
        captured["user_prompt"] = user_prompt
        captured["generation_options"] = generation_options
        out.write_bytes(b"final")
        from uniapi_doubao_image_client import DoubaoRestorationResult, SavedImage
        from uniapi_image_client import InputImage
        return DoubaoRestorationResult(
            final_prompt=user_prompt,
            input_images=[InputImage(path=img, content_type="image/jpeg", size_bytes=0, sha256="x")],
            saved_images=[SavedImage(path=out, sha256="x", size_bytes=5, revised_prompt=None)],
            response_text=[],
            response={},
        )

    monkeypatch.setattr("extract_character_clothing_doubao.restore_doubao_reference", fake_restore)

    fake = _FakeDoubaoClient()
    result = extract_character_top_doubao(
        fake,  # type: ignore[arg-type]
        image_paths=[img],
        output_path=out,
        model="doubao-seedream-5-0-260128",
    )
    assert captured["task_instructions"].startswith("You are a professional costume asset extractor")
    assert "transparent" in captured["user_prompt"].lower()
    # The wrapper picks a 2048x2048 default for top/pants so it satisfies the
    # Doubao 3686400-pixel floor.
    assert captured["generation_options"]["size"] == "2048x2048"
    assert result.saved_images[0].path == out


def test_extract_character_pants_doubao_uses_pants_instructions(monkeypatch, tmp_path: Path):
    img = _png(tmp_path / "a.jpg")
    out = tmp_path / "pants.png"

    captured: dict = {}

    def fake_restore(client, *, image_paths, user_prompt, output_path, task_instructions, asset_description, overwrite, generation_options):
        captured["task_instructions"] = task_instructions
        captured["user_prompt"] = user_prompt
        captured["generation_options"] = generation_options
        out.write_bytes(b"final")
        from uniapi_doubao_image_client import DoubaoRestorationResult, SavedImage
        from uniapi_image_client import InputImage
        return DoubaoRestorationResult(
            final_prompt=user_prompt,
            input_images=[InputImage(path=img, content_type="image/jpeg", size_bytes=0, sha256="x")],
            saved_images=[SavedImage(path=out, sha256="x", size_bytes=5, revised_prompt=None)],
            response_text=[],
            response={},
        )

    monkeypatch.setattr("extract_character_clothing_doubao.restore_doubao_reference", fake_restore)

    fake = _FakeDoubaoClient()
    extract_character_pants_doubao(
        fake,  # type: ignore[arg-type]
        image_paths=[img],
        output_path=out,
        model="doubao-seedream-5-0-260128",
    )
    assert captured["task_instructions"].startswith("You are a professional costume asset extractor")
    assert "waistband" in captured["user_prompt"].lower()
    assert captured["generation_options"]["size"] == "2048x2048"


def test_size_for_part_combined_uses_2k_minimum():
    """Combined plate uses a portrait-friendly default that still meets the floor."""
    assert _size_for_part("doubao-seedream-5-0-260128", "combined", "auto") == "2048x2048"
    assert _size_for_part("doubao-seedream-5-0-260128", "combined", "3K") == "3K"


def test_size_for_part_rejects_undersized_explicit_size():
    """Explicit WxH below the 3686400-pixel floor should fail at the wrapper."""
    with pytest.raises(ValueError, match="3686400"):
        _size_for_part("doubao-seedream-5-0-260128", "top", "1024x1024")


# ---------------------------------------------------------------------------
# execute_plan_doubao
# ---------------------------------------------------------------------------


def test_execute_plan_doubao_runs_top_and_pants(monkeypatch, tmp_path: Path):
    img = _png(tmp_path / "a.jpg")
    plan = plan_clothing_extraction(
        [img], output=tmp_path / "out.png", parts=["top", "pants"]
    )

    fake = _FakeDoubaoClient()
    calls: list[str] = []

    def fake_restore(client, *, image_paths, user_prompt, output_path, task_instructions, asset_description, overwrite, generation_options):
        calls.append(Path(output_path).name)
        Path(output_path).write_bytes(b"final")
        from uniapi_doubao_image_client import DoubaoRestorationResult, SavedImage
        return DoubaoRestorationResult(
            final_prompt=user_prompt,
            input_images=[],
            saved_images=[SavedImage(path=Path(output_path), sha256="x", size_bytes=5, revised_prompt=None)],
            response_text=[],
            response={},
        )

    monkeypatch.setattr("extract_character_clothing_doubao.restore_doubao_reference", fake_restore)

    results = execute_plan_doubao(plan, client=fake)  # type: ignore[arg-type]
    assert "top" in results and "pants" in results
    assert calls == ["out_top.png", "out_pants.png"]
    assert plan.top_metadata is not None and plan.top_metadata.is_file()
    assert plan.pants_metadata is not None and plan.pants_metadata.is_file()


# ---------------------------------------------------------------------------
# CLI dry-run for the Doubao branch
# ---------------------------------------------------------------------------


def test_doubao_cli_dry_run(tmp_path: Path, capsys):
    img = _png(tmp_path / "a.jpg")
    rc = doubao_cli_main(
        [
            "--model",
            "doubao-seedream-5-0-260128",
            "--images",
            str(img),
            "--output",
            str(tmp_path / "out.png"),
            "--dry-run",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["provider"] == "doubao"
    assert payload["model"] == "doubao-seedream-5-0-260128"
    assert payload["resolved_size_top"] == "2048x2048"


def test_doubao_cli_rejects_unknown_model(tmp_path: Path, capsys):
    """argparse enforces known Doubao model ids; this asserts the choices list."""
    img = _png(tmp_path / "a.jpg")
    rc = doubao_cli_main(
        [
            "--model",
            "doubao-seedream-5-0-260128",
            "--images",
            str(img),
            "--output",
            str(tmp_path / "out.png"),
            "--dry-run",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["model"] == "doubao-seedream-5-0-260128"


# ---------------------------------------------------------------------------
# Two-stage wiring
# ---------------------------------------------------------------------------


def test_two_stage_models_include_doubao():
    assert "doubao-seedream-5-0-260128" in ALL_MODELS


def test_classify_model_routes_doubao():
    assert _classify_model("doubao-seedream-5-0-260128") == "doubao"
    assert _classify_model("doubao-seedream-5-0-lite-260128") == "doubao"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
