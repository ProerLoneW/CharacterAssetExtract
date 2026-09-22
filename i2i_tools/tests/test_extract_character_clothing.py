"""Unit tests for extract_character_clothing.

These tests cover the planning, prompt composition, and CLI plumbing without
making any API calls. Run with::

    cd /gemini/platform/public/aigc/lss/Interns/wyn/general_video_annotation_level3_v2
    python -m pytest i2i_tools/tests/test_extract_character_clothing.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from extract_character_clothing import (  # noqa: E402
    COMBINED_INSTRUCTIONS,
    ClothingExtractionPlan,
    PANTS_INSTRUCTIONS,
    TOP_INSTRUCTIONS,
    _resolve_base_paths,
    bake_alpha,
    bake_saved_image,
    discover_character_image_groups,
    estimate_background_color,
    extract_character_pants,
    extract_character_top,
    main,
    plan_clothing_extraction,
)


def _write_jpeg(path: Path) -> None:
    """Write a tiny but valid JPEG file (1x1 white pixel)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
        b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n"
        b"\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d"
        b"\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342"
        b"\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00"
        b"\xff\xc4\x00\x14\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
        b"\x00\x00\x00\x00\x00\x00\x00\x00"
        b"\xff\xda\x00\x08\x01\x01\x00\x00?\x00\x37\xff\xd9"
    )


# ---------------------------------------------------------------------------
# Prompt instructions
# ---------------------------------------------------------------------------


def test_top_instructions_require_transparent_background():
    assert "transparent" in TOP_INSTRUCTIONS.lower()
    assert "alpha" in TOP_INSTRUCTIONS.lower()
    # Hard prohibition against body parts bleeding into the output.
    assert "face" in TOP_INSTRUCTIONS.lower()
    assert "mannequin" in TOP_INSTRUCTIONS.lower()


def test_pants_instructions_require_transparent_background():
    assert "transparent" in PANTS_INSTRUCTIONS.lower()
    assert "alpha" in PANTS_INSTRUCTIONS.lower()
    # The pants plate must orient the waistband up, hem down.
    assert "waistband" in PANTS_INSTRUCTIONS.lower()
    assert "hem" in PANTS_INSTRUCTIONS.lower()


def test_combined_instructions_stack_two_halves():
    assert "transparent" in COMBINED_INSTRUCTIONS.lower()
    assert "top half" in COMBINED_INSTRUCTIONS.lower()
    assert "bottom half" in COMBINED_INSTRUCTIONS.lower()


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def test_resolve_base_paths_with_png_suffix(tmp_path: Path):
    top, pants, combined = _resolve_base_paths(
        tmp_path / "out.png",
        request_top=True,
        request_pants=True,
        request_combined=True,
    )
    assert top == (tmp_path / "out_top.png").resolve()
    assert pants == (tmp_path / "out_pants.png").resolve()
    assert combined == (tmp_path / "out_combined.png").resolve()


def test_resolve_base_paths_with_directory(tmp_path: Path):
    top, pants, combined = _resolve_base_paths(
        tmp_path / "out",
        request_top=True,
        request_pants=True,
        request_combined=False,
    )
    assert top == (tmp_path / "out" / "top.png").resolve()
    assert pants == (tmp_path / "out" / "pants.png").resolve()
    assert combined is None


def test_resolve_base_paths_only_top(tmp_path: Path):
    top, pants, combined = _resolve_base_paths(
        tmp_path / "out.png",
        request_top=True,
        request_pants=False,
        request_combined=False,
    )
    assert top == (tmp_path / "out_top.png").resolve()
    assert pants is None
    assert combined is None


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------


def test_plan_default_parts(tmp_path: Path):
    img1 = tmp_path / "a.jpg"
    img2 = tmp_path / "b.jpg"
    _write_jpeg(img1)
    _write_jpeg(img2)
    plan = plan_clothing_extraction(
        [img1, img2], output=tmp_path / "character.png", character_label="CHR001"
    )
    assert plan.character_label == "CHR001"
    assert {p.resolve() for p in plan.image_paths} == {img1.resolve(), img2.resolve()}
    assert plan.requested_parts() == {"top", "pants"}
    assert plan.top_output == (tmp_path / "character_top.png").resolve()
    assert plan.pants_output == (tmp_path / "character_pants.png").resolve()
    assert plan.combined_output is None
    assert plan.top_metadata == (tmp_path / "character_top.png.json").resolve()
    assert plan.pants_metadata == (tmp_path / "character_pants.png.json").resolve()


def test_plan_combined_only(tmp_path: Path):
    img = tmp_path / "a.jpg"
    _write_jpeg(img)
    plan = plan_clothing_extraction(
        [img], output=tmp_path / "out.png", parts=["combined"]
    )
    assert plan.requested_parts() == {"combined"}
    assert plan.top_output is None
    assert plan.pants_output is None
    assert plan.combined_output == (tmp_path / "out_combined.png").resolve()


def test_plan_rejects_unknown_parts(tmp_path: Path):
    img = tmp_path / "a.jpg"
    _write_jpeg(img)
    with pytest.raises(ValueError):
        plan_clothing_extraction([img], output=tmp_path / "out", parts=["shoes"])


def test_plan_rejects_empty_parts(tmp_path: Path):
    img = tmp_path / "a.jpg"
    _write_jpeg(img)
    with pytest.raises(ValueError):
        plan_clothing_extraction([img], output=tmp_path / "out", parts=[])


def test_plan_rejects_missing_input(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        plan_clothing_extraction(
            [tmp_path / "does_not_exist.jpg"], output=tmp_path / "out"
        )


# ---------------------------------------------------------------------------
# Programmatic extractors (signature only — no API call)
# ---------------------------------------------------------------------------


def test_extract_character_top_signature_uses_top_instructions(monkeypatch, tmp_path: Path):
    img = tmp_path / "a.jpg"
    _write_jpeg(img)
    out = tmp_path / "top.png"

    captured: dict = {}

    def fake_restore(client, *, image_paths, user_prompt, output_path, task_instructions, asset_description, overwrite, generation_options):
        captured["image_paths"] = list(image_paths)
        captured["user_prompt"] = user_prompt
        captured["task_instructions"] = task_instructions
        captured["output_path"] = output_path
        captured["generation_options"] = generation_options
        out.write_bytes(b"\x89PNG\r\n\x1a\nfake")
        from uniapi_gemini_image_client import GeminiRestorationResult, SavedImage
        from uniapi_image_client import InputImage
        return GeminiRestorationResult(
            final_prompt=user_prompt,
            input_images=[InputImage(path=img, content_type="image/jpeg", size_bytes=0, sha256="x")],
            saved_images=[SavedImage(path=out, sha256="x", size_bytes=0, revised_prompt=None)],
            response_text=[],
            response={},
        )

    from extract_character_clothing import restore_gemini_reference
    monkeypatch.setattr("extract_character_clothing.restore_gemini_reference", fake_restore)

    class _DummyClient:
        pass

    result = extract_character_top(
        _DummyClient(),  # type: ignore[arg-type]
        image_paths=[img],
        output_path=out,
    )
    assert captured["task_instructions"] == TOP_INSTRUCTIONS
    assert "transparent" in captured["user_prompt"].lower()
    assert captured["generation_options"]["aspect_ratio"] == "1:1"
    assert result.saved_images[0].path == out


def test_extract_character_pants_signature_uses_pants_instructions(monkeypatch, tmp_path: Path):
    img = tmp_path / "a.jpg"
    _write_jpeg(img)
    out = tmp_path / "pants.png"

    captured: dict = {}

    def fake_restore(client, *, image_paths, user_prompt, output_path, task_instructions, asset_description, overwrite, generation_options):
        captured["task_instructions"] = task_instructions
        captured["generation_options"] = generation_options
        out.write_bytes(b"\x89PNG\r\n\x1a\nfake")
        from uniapi_gemini_image_client import GeminiRestorationResult, SavedImage
        from uniapi_image_client import InputImage
        return GeminiRestorationResult(
            final_prompt=user_prompt,
            input_images=[InputImage(path=img, content_type="image/jpeg", size_bytes=0, sha256="x")],
            saved_images=[SavedImage(path=out, sha256="x", size_bytes=0, revised_prompt=None)],
            response_text=[],
            response={},
        )

    monkeypatch.setattr("extract_character_clothing.restore_gemini_reference", fake_restore)

    class _DummyClient:
        pass

    extract_character_pants(
        _DummyClient(),  # type: ignore[arg-type]
        image_paths=[img],
        output_path=out,
    )
    assert captured["task_instructions"] == PANTS_INSTRUCTIONS
    assert captured["generation_options"]["aspect_ratio"] == "1:1"


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def test_discover_character_image_groups_groups_by_subdir(tmp_path: Path):
    (tmp_path / "CHR001").mkdir()
    (tmp_path / "CHR002").mkdir()
    _write_jpeg(tmp_path / "CHR001" / "a.jpg")
    _write_jpeg(tmp_path / "CHR001" / "b.png")
    _write_jpeg(tmp_path / "CHR002" / "c.jpg")

    groups = discover_character_image_groups(tmp_path)
    assert len(groups) == 2
    keys = sorted(g[0].parent.name for g in groups)
    assert keys == ["CHR001", "CHR002"]
    chr001 = next(g for g in groups if g[0].parent.name == "CHR001")
    assert len(chr001) == 2
    # Sorted alphabetically by path.
    assert chr001[0].name == "a.jpg"
    assert chr001[1].name == "b.png"


def test_discover_character_image_groups_handles_root_images(tmp_path: Path):
    _write_jpeg(tmp_path / "loose_a.jpg")
    _write_jpeg(tmp_path / "loose_b.jpg")
    (tmp_path / "CHR001").mkdir()
    _write_jpeg(tmp_path / "CHR001" / "c.jpg")
    groups = discover_character_image_groups(tmp_path)
    assert len(groups) == 2
    flat = [p.name for g in groups for p in g]
    assert "loose_a.jpg" in flat
    assert "c.jpg" in flat


def test_discover_character_image_groups_missing_root(tmp_path: Path):
    assert discover_character_image_groups(tmp_path / "does_not_exist") == []


# ---------------------------------------------------------------------------
# CLI dry-run
# ---------------------------------------------------------------------------


def test_cli_dry_run_prints_plan(monkeypatch, tmp_path: Path, capsys):
    img = tmp_path / "a.jpg"
    _write_jpeg(img)
    rc = main(
        [
            "--images",
            str(img),
            "--output",
            str(tmp_path / "out.png"),
            "--dry-run",
        ]
    )
    assert rc == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["task"] == "character_clothing_extraction"
    assert payload["parts_requested"] == ["pants", "top"]
    assert payload["outputs"]["top"].endswith("out_top.png")
    assert payload["outputs"]["pants"].endswith("out_pants.png")


def test_cli_dry_run_combined_only(monkeypatch, tmp_path: Path, capsys):
    img = tmp_path / "a.jpg"
    _write_jpeg(img)
    rc = main(
        [
            "--images",
            str(img),
            "--output",
            str(tmp_path / "out.png"),
            "--parts",
            "combined",
            "--dry-run",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["parts_requested"] == ["combined"]
    assert payload["outputs"]["top"] is None
    assert payload["outputs"]["pants"] is None
    assert payload["outputs"]["combined"].endswith("out_combined.png")


def test_cli_dry_run_rejects_bad_parts(tmp_path: Path, capsys):
    img = tmp_path / "a.jpg"
    _write_jpeg(img)
    rc = main(
        [
            "--images",
            str(img),
            "--output",
            str(tmp_path / "out.png"),
            "--parts",
            "shoes",
            "--dry-run",
        ]
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert "shoes" in err


def test_cli_dry_run_rejects_missing_input(tmp_path: Path, capsys):
    rc = main(
        [
            "--images",
            str(tmp_path / "missing.jpg"),
            "--output",
            str(tmp_path / "out.png"),
            "--dry-run",
        ]
    )
    assert rc == 2


def test_cli_dry_run_with_aspect_override(tmp_path: Path, capsys):
    img = tmp_path / "a.jpg"
    _write_jpeg(img)
    rc = main(
        [
            "--images",
            str(img),
            "--output",
            str(tmp_path / "out.png"),
            "--aspect-ratio",
            "4:5",
            "--dry-run",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["common_generation_options"]["aspect_ratio"] == "4:5"


# ---------------------------------------------------------------------------
# Alpha baking
# ---------------------------------------------------------------------------


def _make_card_image(path: Path, *, bg=(230, 230, 230), fg=(60, 100, 200), size=64):
    """Write a tiny PNG: a solid-color card with a colored square in the middle."""
    from PIL import Image

    img = Image.new("RGB", (size, size), bg)
    inner_w = size // 3
    for y in range(size // 3, size // 3 + inner_w):
        for x in range(size // 3, size // 3 + inner_w):
            img.putpixel((x, y), fg)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="PNG")
    return path


def test_estimate_background_color_finds_card_color(tmp_path: Path):
    src = _make_card_image(tmp_path / "card.png", bg=(200, 210, 220))
    from PIL import Image
    with Image.open(src) as img:
        bg = estimate_background_color(img)
    assert bg is not None
    # Median of border should be exactly the card color.
    assert bg == (200, 210, 220)


def test_bake_alpha_makes_card_transparent(tmp_path: Path):
    src = _make_card_image(tmp_path / "card.png", bg=(240, 240, 240), fg=(40, 80, 200))
    from PIL import Image
    with Image.open(src) as raw:
        out = bake_alpha(raw, threshold=20, softness=10)
    assert out.mode == "RGBA"
    # Top-left corner pixel must now be transparent.
    assert out.getpixel((0, 0))[3] == 0
    # Center colored square must remain opaque.
    cx = out.size[0] // 2
    cy = out.size[1] // 2
    assert out.getpixel((cx, cy))[3] == 255


def test_bake_saved_image_writes_rgba_png(tmp_path: Path):
    src = _make_card_image(tmp_path / "card.png", bg=(220, 220, 220), fg=(10, 100, 50))
    info = bake_saved_image(src, threshold=20, softness=10)
    assert info["baked"] is True
    assert info["final_mode"] == "RGBA"
    from PIL import Image
    with Image.open(src) as reopened:
        assert reopened.mode == "RGBA"


def test_cli_dry_run_reports_bake_alpha_settings(tmp_path: Path, capsys):
    img = tmp_path / "a.jpg"
    _write_jpeg(img)
    rc = main(
        [
            "--images",
            str(img),
            "--output",
            str(tmp_path / "out.png"),
            "--bake-alpha",
            "--bake-alpha-threshold",
            "30",
            "--dry-run",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["bake_alpha"] is True
    assert payload["bake_alpha_threshold"] == 30


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
