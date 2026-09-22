"""Unit tests for restore_character_three_view_and_costume.

These tests build a synthetic character visual bank on disk and exercise the
discovery + planning logic without making any API calls. Run with::

    cd /gemini/platform/public/aigc/lss/Interns/wyn/general_video_annotation_level3_v2
    python -m pytest i2i_tools/tests/test_restore_character_three_view_and_costume.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from restore_character_three_view_and_costume import (  # noqa: E402
    CharacterAsset,
    CharacterVisualBank,
    discover_character_banks,
    load_character_visual_bank,
    plan_character_restorations,
    select_costume_inputs,
    select_three_view_inputs,
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


@pytest.fixture()
def fake_character_root(tmp_path: Path) -> Path:
    """Create one character bank with two appearance states."""

    root = tmp_path / "characters"
    scope = root / "global_TEST" / "S1" / "visual"
    scope.mkdir(parents=True)
    (scope / "representative").mkdir()
    _write_jpeg(scope / "representative" / "primary_face.jpg")
    _write_jpeg(scope / "representative" / "primary_appearance.jpg")

    body_app1 = scope / "body" / "front" / "partial" / "CHR001_APP001"
    body_app1.mkdir(parents=True)
    for index in range(3):
        _write_jpeg(body_app1 / f"CHR001_BODY_{index:04d}.jpg")
    body_app2 = scope / "body" / "unknown" / "upper_body" / "CHR001_APP002"
    body_app2.mkdir(parents=True)
    for index in range(2):
        _write_jpeg(body_app2 / f"CHR001_BODY_{index:04d}.jpg")

    face_dir = scope / "face" / "other"
    face_dir.mkdir(parents=True)
    _write_jpeg(face_dir / "CHR001_FACE_0001.jpg")

    character_payload = {
        "character_id": "CHR001",
        "semantic_scope_id": "global_TEST",
        "subject_id": "S1",
        "face_gallery": {
            "other": [
                {
                    "asset_id": "CHR001_FACE_0001",
                    "kind": "face",
                    "view": "other",
                    "crop_type": None,
                    "expression": "other",
                    "appearance_state_id": "CHR001_APP001",
                    "quality_score": 0.6,
                    "quality_components": {"sharpness": 0.5},
                    "registry_path": "face/other/CHR001_FACE_0001.jpg",
                    "perceptual_hash": "abc",
                }
            ]
        },
        "body_gallery": {
            "front": {
                "partial": [
                    {
                        "asset_id": f"CHR001_BODY_{i:04d}",
                        "kind": "body",
                        "view": "front",
                        "crop_type": "partial",
                        "appearance_state_id": "CHR001_APP001",
                        "quality_score": 0.7 - i * 0.05,
                        "quality_components": {"sharpness": 0.5},
                        "registry_path": f"body/front/partial/CHR001_APP001/CHR001_BODY_{i:04d}.jpg",
                        "perceptual_hash": f"hash{i}",
                    }
                    for i in range(3)
                ]
            },
            "unknown": {
                "upper_body": [
                    {
                        "asset_id": f"CHR001_BODY_APP2_{i:04d}",
                        "kind": "body",
                        "view": "unknown",
                        "crop_type": "upper_body",
                        "appearance_state_id": "CHR001_APP002",
                        "quality_score": 0.55,
                        "quality_components": {"sharpness": 0.5},
                        "registry_path": f"body/unknown/upper_body/CHR001_APP002/CHR001_BODY_{i:04d}.jpg",
                        "perceptual_hash": f"hash_app2_{i}",
                    }
                    for i in range(2)
                ]
            },
        },
    }
    (scope / "character.json").write_text(
        json.dumps(character_payload, ensure_ascii=False, indent=2)
    )

    appearance_payload = {
        "appearance_states": [
            {
                "state_id": "CHR001_APP001",
                "tracklet_ids": ["SHOT_TEST_T000"],
                "shots": ["SHOT_TEST"],
                "body_assets": [
                    f"body/front/partial/CHR001_APP001/CHR001_BODY_{i:04d}.jpg"
                    for i in range(3)
                ],
                "face_assets": ["face/other/CHR001_FACE_0001.jpg"],
            },
            {
                "state_id": "CHR001_APP002",
                "tracklet_ids": ["SHOT_TEST_T001"],
                "shots": ["SHOT_TEST2"],
                "body_assets": [
                    f"body/unknown/upper_body/CHR001_APP002/CHR001_BODY_{i:04d}.jpg"
                    for i in range(2)
                ],
                "face_assets": [],
            },
        ]
    }
    (scope / "appearance_states.json").write_text(
        json.dumps(appearance_payload, ensure_ascii=False, indent=2)
    )
    return root


def test_load_character_visual_bank_groups_by_state(fake_character_root: Path):
    visual = fake_character_root / "global_TEST" / "S1" / "visual"
    bank = load_character_visual_bank(visual)
    assert bank is not None
    assert bank.character_id == "CHR001"
    assert bank.appearance_state_ids == ["CHR001_APP001", "CHR001_APP002"]
    assert bank.asset_count("CHR001_APP001") == 4  # 3 bodies + 1 face
    assert bank.asset_count("CHR001_APP002") == 2
    assert bank.representative_face is not None
    assert bank.representative_face.exists()


def test_select_three_view_inputs_uses_representative_face(
    fake_character_root: Path,
):
    visual = fake_character_root / "global_TEST" / "S1" / "visual"
    bank = load_character_visual_bank(visual)
    inputs = select_three_view_inputs(bank, "CHR001_APP001", max_images=5)
    # representative face must be first.
    assert inputs[0] == bank.representative_face.resolve()
    # All inputs must exist on disk.
    for path in inputs:
        assert path.is_file()
    # We never include duplicate paths.
    assert len({p.resolve() for p in inputs}) == len(inputs)


def test_select_costume_inputs_excludes_faces(fake_character_root: Path):
    visual = fake_character_root / "global_TEST" / "S1" / "visual"
    bank = load_character_visual_bank(visual)
    inputs = select_costume_inputs(bank, "CHR001_APP001", max_images=4)
    # No face paths should be present.
    face_path = bank.representative_face.resolve()
    assert face_path not in [p.resolve() for p in inputs]
    assert len(inputs) == 3  # only body crops


def test_plan_character_restorations_writes_expected_paths(
    fake_character_root: Path,
    tmp_path: Path,
):
    visual = fake_character_root / "global_TEST" / "S1" / "visual"
    bank = load_character_visual_bank(visual)
    plans = plan_character_restorations(bank, output_root=tmp_path / "refs")
    assert {p.state_id for p in plans} == {"CHR001_APP001", "CHR001_APP002"}
    for plan in plans:
        assert plan.three_view_output.exists() is False  # not written yet
        assert plan.three_view_inputs
        assert plan.costume_inputs
        # Output paths follow the documented naming.
        assert plan.three_view_output.name == f"{plan.state_id}_three_view.png"
        assert plan.costume_output.name == f"{plan.state_id}_costume.png"


def test_discover_character_banks_filters(fake_character_root: Path):
    banks = discover_character_banks(fake_character_root)
    assert len(banks) == 1
    banks = discover_character_banks(fake_character_root, character_filter=["CHR999"])
    assert banks == []


def test_character_asset_dataclass():
    asset = CharacterAsset(
        path=Path("/tmp/test.jpg"),
        kind="body",
        view="front",
        crop_type="partial",
        appearance_state_id="CHR001_APP001",
        asset_id="CHR001_BODY_0001",
        quality_score=0.7,
    )
    assert asset.kind == "body"
    assert asset.view == "front"


def test_character_visual_bank_dataclass():
    bank = CharacterVisualBank(
        character_id="CHR001",
        semantic_scope_id="global_TEST",
        subject_id="S1",
        visual_root=Path("/tmp"),
        representative_face=None,
        representative_appearance=None,
    )
    assert bank.character_id == "CHR001"
    assert bank.asset_count("anything") == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
