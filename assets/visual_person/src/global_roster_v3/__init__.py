# -*- coding: utf-8 -*-
"""global_roster: standalone per-global-unit character roster extraction.

Self-contained package.  Zero imports from video_annotation / m16 / m14 /
m15 or any other repo module.  Pipeline: shot probe -> frame sampling ->
Sa2VA protagonist segmentation -> connected-component instances -> face
detection/embedding + quality filter -> in-shot IoU tracks -> two-pass
density clustering -> protagonist selection (frequency + frontal-face
exception) -> roster with face avatars, full-body crops and centroids.
"""
