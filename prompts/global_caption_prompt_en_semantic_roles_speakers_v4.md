# MiMo Compact Global Caption + Authoritative ASR Speaker Attribution V4

Watch the complete supplied global clip once. In the same response, create the
global person registry, a compact factual global caption, and assign each
supplied ASR ID to one registered person or `unknown`. Output exactly one JSON
object without Markdown or reasoning.

## Source-of-truth rules

- The supplied `Authoritative ASR Timeline` owns speech text and timing. Use it
  as evidence, but DO NOT copy text or timestamps into your output. Return only
  the compact attribution for each supplied `asr_id`; local code performs the
  immutable join afterward.
- Return every supplied `asr_id` exactly once. Never invent, merge, split, or
  omit an ASR ID.
- `S1`, `S2`, ... identify people only. Define each ID exactly once in
  `subject_registry` and reuse it throughout this response.
- Keep one S-ID across cuts, pose changes, close-ups, and temporary occlusion.
  Use observable visual cues rather than names, celebrity knowledge,
  relationships, occupations, emotions, or inferred story roles.
- `speaker_id` must be one registered S-ID or `unknown`. Never generate an
  `SPK_*` voice-cluster label.

## Speaker-attribution rules

- Inspect multiple frames over the complete ASR interval. Strong direct
  evidence is synchronized visible mouth movement, exclusive audio/visual
  co-occurrence, or repeated voice/face co-occurrence.
- A reaction shot, screen presence, dialogue meaning, expected turn-taking,
  age/gender, or a generic voice impression is not direct identity evidence.
- Use `unknown` for an off-screen, occluded, ambiguous, group, or overlapping
  speaker unless repeated evidence uniquely binds that voice to an S-ID.
- `attribution_evidence_type` must be one of `visible_lip_sync`,
  `exclusive_audio_visual_cooccurrence`, `repeated_voice_face_cooccurrence`,
  `context_only`, `voice_only`, `none`, or `unknown`.
- Use `high` only with a direct evidence type. Keep evidence under 20 words.
- `overlap_status=none` means clearly one voice, `possible` means uncertain,
  `overlap` means multiple simultaneous voices, and `unknown` means the audio
  is insufficient.
- `speech_clarity` and `background_interference` are preliminary observations;
  local signal and purity checks make the final asset decision.

## Compactness rules

- Keep each identity description under 35 words and at most 8 visual cues.
- Return 2-6 concise global event rows when events exist.
- Keep `subject_event_timeline`, `semantic_role_observations`, and
  `visible_person_per_shot` empty; downstream shot/segment captioning owns that
  detail. Do not duplicate ASR speech into any event timeline.

## Required JSON schema

```json
{
  "video_duration_sec": 0.0,
  "annotation_confidence": "high|medium|low|unknown",
  "global_caption": {
    "global_description": "compact chronological factual description",
    "global_scene_description": "compact static scene description",
    "overall_soundscape": "audible non-dialogue sound or N/A",
    "non_diegetic_music": "background score or N/A",
    "key_topics": []
  },
  "subject_registry": [
    {
      "subject_id": "S1",
      "subject_type": "person",
      "identity_description": "observable re-identification description",
      "visual_cues": [],
      "voice_cues": [],
      "first_seen_sec": 0.0,
      "last_seen_sec": 1.0,
      "confidence": "high|medium|low|unknown",
      "importance_tier": "primary|supporting|incidental|background|unknown",
      "importance_score": 0.0,
      "role_label": "short observable functional label",
      "narrative_functions": []
    }
  ],
  "semantic_role_summary": {
    "primary_subject_ids": [],
    "supporting_subject_ids": [],
    "incidental_subject_ids": [],
    "background_subject_ids": [],
    "unknown_subject_ids": []
  },
  "semantic_role_observations": [],
  "subject_event_timeline": [],
  "global_event_timeline": [
    {
      "time_range_sec": [0.0, 1.0],
      "event_phase": "introduction|continuation|escalation|turning_point|resolution|transition",
      "subject_ids": ["S1"],
      "event": "directly observed event",
      "state_before": "observable state",
      "state_after": "observable state",
      "speaker_ids": ["S1"]
    }
  ],
  "visible_person_per_shot": [],
  "visible_count_total": 1,
  "asr_speaker_attributions": [
    {
      "asr_id": "ASR00001",
      "speaker_id": "S1|unknown",
      "attribution_confidence": "high|medium|low|unknown",
      "attribution_evidence_type": "visible_lip_sync|exclusive_audio_visual_cooccurrence|repeated_voice_face_cooccurrence|context_only|voice_only|none|unknown",
      "attribution_evidence": "short timestamp-specific basis",
      "is_off_screen": false,
      "overlap_status": "none|possible|overlap|unknown",
      "speech_clarity": "clean|usable|poor|unknown",
      "background_interference": "none|mild|strong|unknown"
    }
  ],
  "non_person_entities": [],
  "quality_issues": []
}
```

Before returning, verify that every attribution references a registered S-ID
or `unknown`, every supplied ASR ID occurs exactly once, no ASR text/time is
copied into the response, and the JSON object is complete.
