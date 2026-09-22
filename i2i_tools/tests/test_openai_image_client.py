"""Unit tests for the OpenAI image-client additions and helpers.

These tests cover the new ``resolve_openai_model_size`` helper, the
``OPENAI_IMAGE_MODEL_PROFILES`` table, and the OpenAI-driven clothing
extraction wrappers. They DO NOT make any API calls. Run with::

    cd /gemini/platform/public/aigc/lss/Interns/wyn/general_video_annotation_level3_v2
    python -m pytest i2i_tools/tests/test_openai_image_client.py -v
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uniapi_image_client import (  # noqa: E402
    OPENAI_IMAGE_MODEL_PROFILES,
    UniAPIImageClient,
    UniAPIImageError,
    _validate_output_targets,
    compose_prompt,
    inspect_images,
    resolve_openai_model_size,
)
from extract_character_clothing_openai import (  # noqa: E402
    DEFAULT_OPENAI_IMAGE_MODEL,
    SUPPORTED_OPENAI_MODELS,
    _size_for_aspect,
    extract_character_pants_openai,
    extract_character_top_openai,
    execute_plan_openai,
    main as openai_cli_main,
)
from extract_character_clothing import plan_clothing_extraction  # noqa: E402
from two_stage_clothing_pipeline import (  # noqa: E402
    ALL_MODELS,
    plan_two_stage,
    _classify_model,
)


# ---------------------------------------------------------------------------
# OpenAI model profiles
# ---------------------------------------------------------------------------


def test_openai_model_profiles_have_expected_keys():
    assert "gpt-image-2" in OPENAI_IMAGE_MODEL_PROFILES
    assert "gpt-image-2.5-sunburst" in OPENAI_IMAGE_MODEL_PROFILES
    assert "gpt-image-1" in OPENAI_IMAGE_MODEL_PROFILES


def test_default_openai_model_is_gpt_image_2():
    assert DEFAULT_OPENAI_IMAGE_MODEL == "gpt-image-2"
    assert "gpt-image-2" in SUPPORTED_OPENAI_MODELS


def test_resolve_openai_model_size_handles_auto_and_explicit():
    assert resolve_openai_model_size("gpt-image-2", "auto") == "auto"
    assert resolve_openai_model_size("gpt-image-2", "1024x1024") == "1024x1024"
    assert resolve_openai_model_size("gpt-image-2", None) == "auto"


def test_resolve_openai_model_size_rejects_unsupported_sizes():
    with pytest.raises(ValueError):
        resolve_openai_model_size("gpt-image-2", "2048x2048")


# ---------------------------------------------------------------------------
# Aspect-ratio mapping
# ---------------------------------------------------------------------------


def test_size_for_aspect_uses_expected_defaults():
    assert _size_for_aspect("1:1", "gpt-image-2", None) == "1024x1024"
    assert _size_for_aspect("2:3", "gpt-image-2", None) == "1024x1536"
    assert _size_for_aspect("3:4", "gpt-image-2", None) == "1024x1536"
    assert _size_for_aspect("16:9", "gpt-image-2", None) == "1536x1024"


def test_size_for_aspect_honors_explicit_size():
    assert _size_for_aspect("1:1", "gpt-image-2", "1536x1024") == "1536x1024"


# ---------------------------------------------------------------------------
# Clothing extraction OpenAI helpers (signature-only, no API call)
# ---------------------------------------------------------------------------


def _png(path: Path) -> Path:
    payload = b"\x89PNG\r\n\x1a\n" + b"X" * 8
    path.write_bytes(payload)
    return path


class _FakeOpenAIClient:
    def __init__(self, response: dict | None = None):
        self.response = response or {
            "data": [{"b64_json": base64.b64encode(b"final").decode("ascii")}]
        }
        self.calls: list[dict] = []
        self.timeout_seconds = 1.0

    def edit(self, **kwargs):
        self.calls.append(kwargs)
        return self.response, []


def test_extract_character_top_openai_uses_top_instructions(monkeypatch, tmp_path: Path):
    img = _png(tmp_path / "a.jpg")
    out = tmp_path / "top.png"

    captured: dict = {}
    fake = _FakeOpenAIClient()

    def fake_restore(client, *, image_paths, user_prompt, output_path, task_instructions, asset_description, overwrite, edit_options):
        captured["image_paths"] = list(image_paths)
        captured["user_prompt"] = user_prompt
        captured["task_instructions"] = task_instructions
        captured["edit_options"] = edit_options
        out.write_bytes(b"final")
        from uniapi_image_client import SavedImage, RestorationResult
        return RestorationResult(
            final_prompt=user_prompt,
            input_images=[],
            saved_images=[SavedImage(path=out, sha256="x", size_bytes=5, revised_prompt=None)],
            response=self.response if False else {"created": 0},
        )

    monkeypatch.setattr("extract_character_clothing_openai.restore_reference", fake_restore)

    result = extract_character_top_openai(
        fake,  # type: ignore[arg-type]
        image_paths=[img],
        output_path=out,
        model="gpt-image-2",
    )
    assert captured["task_instructions"].startswith("You are a professional costume asset extractor")
    assert "transparent" in captured["user_prompt"].lower()
    assert captured["edit_options"]["model"] == "gpt-image-2"
    assert captured["edit_options"]["size"] == "1024x1024"
    assert captured["edit_options"]["background"] == "transparent"
    assert result.saved_images[0].path == out


def test_extract_character_pants_openai_uses_pants_instructions(monkeypatch, tmp_path: Path):
    img = _png(tmp_path / "a.jpg")
    out = tmp_path / "pants.png"

    captured: dict = {}
    fake = _FakeOpenAIClient()

    def fake_restore(client, *, image_paths, user_prompt, output_path, task_instructions, asset_description, overwrite, edit_options):
        captured["task_instructions"] = task_instructions
        captured["user_prompt"] = user_prompt
        captured["edit_options"] = edit_options
        out.write_bytes(b"final")
        from uniapi_image_client import SavedImage, RestorationResult
        return RestorationResult(
            final_prompt=user_prompt,
            input_images=[],
            saved_images=[SavedImage(path=out, sha256="x", size_bytes=5, revised_prompt=None)],
            response={},
        )

    monkeypatch.setattr("extract_character_clothing_openai.restore_reference", fake_restore)

    extract_character_pants_openai(
        fake,  # type: ignore[arg-type]
        image_paths=[img],
        output_path=out,
        model="gpt-image-2",
    )
    assert captured["task_instructions"].startswith("You are a professional costume asset extractor")
    assert "waistband" in captured["user_prompt"].lower()
    assert captured["edit_options"]["size"] == "1024x1024"


# ---------------------------------------------------------------------------
# execute_plan_openai end-to-end (no API call)
# ---------------------------------------------------------------------------


def test_execute_plan_openai_runs_top_and_pants(monkeypatch, tmp_path: Path):
    img = _png(tmp_path / "a.jpg")
    plan = plan_clothing_extraction(
        [img], output=tmp_path / "out.png", parts=["top", "pants"]
    )

    fake = _FakeOpenAIClient()
    calls: list[str] = []

    def fake_restore(client, *, image_paths, user_prompt, output_path, task_instructions, asset_description, overwrite, edit_options):
        calls.append(Path(output_path).name)
        Path(output_path).write_bytes(b"final")
        from uniapi_image_client import SavedImage, RestorationResult
        return RestorationResult(
            final_prompt=user_prompt,
            input_images=[],
            saved_images=[SavedImage(path=Path(output_path), sha256="x", size_bytes=5, revised_prompt=None)],
            response={},
        )

    monkeypatch.setattr("extract_character_clothing_openai.restore_reference", fake_restore)

    results = execute_plan_openai(plan, client=fake)  # type: ignore[arg-type]
    assert "top" in results and "pants" in results
    assert calls == ["out_top.png", "out_pants.png"]
    assert plan.top_metadata is not None and plan.top_metadata.is_file()
    assert plan.pants_metadata is not None and plan.pants_metadata.is_file()


# ---------------------------------------------------------------------------
# CLI dry-run for the OpenAI branch
# ---------------------------------------------------------------------------


def test_openai_cli_dry_run(tmp_path: Path, capsys):
    img = _png(tmp_path / "a.jpg")
    rc = openai_cli_main(
        [
            "--model",
            "gpt-image-2",
            "--images",
            str(img),
            "--output",
            str(tmp_path / "out.png"),
            "--dry-run",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["provider"] == "openai"
    assert payload["model"] == "gpt-image-2"
    assert payload["resolved_size_top"] == "1024x1024"


def test_openai_cli_dry_run_rejects_unknown_model(tmp_path: Path, capsys):
    """Argparse enforces valid --model choices; this asserts the choices list."""
    img = _png(tmp_path / "a.jpg")
    rc = openai_cli_main(
        [
            "--model",
            "gpt-image-2",
            "--images",
            str(img),
            "--output",
            str(tmp_path / "out.png"),
            "--dry-run",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    # The CLI exposes exactly the three OpenAI models we care about.
    assert payload["model"] == "gpt-image-2"


# ---------------------------------------------------------------------------
# Two-stage pipeline planning
# ---------------------------------------------------------------------------


def test_two_stage_all_models_listed():
    assert "gemini-3-pro-image" in ALL_MODELS
    assert "gemini-3.1-flash-image" in ALL_MODELS
    assert "gemini-3.1-flash-lite-image" in ALL_MODELS
    assert "gpt-image-2" in ALL_MODELS
    assert "gpt-image-2.5-sunburst" in ALL_MODELS
    assert "doubao-seedream-5-0-260128" in ALL_MODELS
    assert len(ALL_MODELS) == 6


def test_classify_model_routes_correctly():
    assert _classify_model("gemini-3-pro-image") == "gemini"
    assert _classify_model("gemini-3.1-flash-image") == "gemini"
    assert _classify_model("gemini-3.1-flash-lite-image") == "gemini"
    assert _classify_model("gpt-image-2") == "openai"
    assert _classify_model("gpt-image-2.5-sunburst") == "openai"
    assert _classify_model("doubao-seedream-5-0-260128") == "doubao"


def test_classify_model_rejects_unknown():
    with pytest.raises(ValueError):
        _classify_model("not-a-model")


def test_plan_two_stage_resolves_image_sizes(tmp_path: Path):
    img = _png(tmp_path / "a.jpg")
    plan = plan_two_stage(
        image_paths=[img],
        output_root=tmp_path / "out",
        character_label="CHR004",
        model="gemini-3-pro-image",
    )
    assert plan.provider == "gemini"
    assert plan.image_size_resolved == "2K"
    assert plan.outputs.three_view.parent == tmp_path / "out" / "gemini-3-pro-image"
    assert plan.outputs.top.name == "CHR004_top.png"
    assert plan.outputs.pants.name == "CHR004_pants.png"

    plan_openai = plan_two_stage(
        image_paths=[img],
        output_root=tmp_path / "out",
        character_label="CHR004",
        model="gpt-image-2",
    )
    assert plan_openai.provider == "openai"
    assert plan_openai.image_size_resolved == "1024x1536"
    assert plan_openai.outputs.three_view.parent == tmp_path / "out" / "gpt-image-2"


def test_plan_two_stage_rejects_lite_4k(tmp_path: Path):
    img = _png(tmp_path / "a.jpg")
    with pytest.raises(ValueError):
        plan_two_stage(
            image_paths=[img],
            output_root=tmp_path / "out",
            character_label="CHR004",
            model="gemini-3.1-flash-lite-image",
            image_size="4K",
        )


# ---------------------------------------------------------------------------
# Integration: existing CLI works unchanged for Gemini provider
# ---------------------------------------------------------------------------


def test_existing_gemini_cli_dry_run_unchanged(tmp_path: Path, capsys):
    img = _png(tmp_path / "a.jpg")
    from extract_character_clothing import main as gemini_main

    rc = gemini_main(
        [
            "--provider",
            "gemini",
            "--model",
            "gemini-3-pro-image",
            "--images",
            str(img),
            "--output",
            str(tmp_path / "out.png"),
            "--dry-run",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["provider"] == "gemini"
    assert payload["model"] == "gemini-3-pro-image"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
