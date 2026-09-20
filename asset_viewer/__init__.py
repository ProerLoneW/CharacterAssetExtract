"""Asset viewer web dashboard for the character asset extraction pipeline.

This package hosts a small read-only web UI that surfaces the outputs of
both pipeline chains the project supports:

* **New chain** (``extract_full_pipeline/global_align_pipeline/``):
  ``global_tracklet_aligner`` produces ``global_alignment/``,
  ``materialization_summary.json`` and ``characters/<scope>/<S#>/``.
* **Legacy chain** (``extract_full_pipeline/assets/visual_person/``):
  ``shot_tracklet_builder`` → ``visual_character_registry_builder`` →
  ``character_asset_bank`` → ``character_role_aligner`` produces
  ``assets/visual_person_v2/asset_bank/`` and ``characters/``.

A single :class:`asset_viewer.server.ViewerServer` discovers both layouts
on disk and renders a unified 3-pane HTML page (workspace picker,
character / tracklet grid, detail drawer) over
``ThreadingHTTPServer``.  No third-party dependencies are required.

Start with::

    python -m extract_full_pipeline.asset_viewer.server \
        --host 0.0.0.0 --port 8910 \
        --workspace-root /gemini/.../workspace
"""

from __future__ import annotations

__all__: list[str] = []