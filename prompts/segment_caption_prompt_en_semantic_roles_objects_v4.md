# MiMo Segment Caption + Semantic Roles + Key Objects V4

Partition the supplied ordered shots into coherent 15-30 second segments,
consolidate semantic-character evidence, and consolidate the supplied
shot-local key-object candidates. You receive the global caption/registry and
all shot captions. Output exactly one JSON object, without Markdown or
reasoning.

## Boundary rules

- Preserve every supplied shot exactly once and in chronological order.
- Cut only at shot boundaries.
- Every segment must be at most 30 seconds.
- Prefer at least 15 seconds. A shorter final remainder is allowed only when
  merging it into the previous segment would exceed 30 seconds.
- Prefer semantic event boundaries over equal-duration chunks.
- Continuing dialogue, action, or manipulation of the same key object is soft
  evidence against a cut. The 30-second maximum is always stronger.
- `VALID BOUNDARY GUIDE` in the input is authoritative for timing. Select one
  `valid_complete_partitions` entry and copy its `shot_id_groups` exactly.
  Do not invent another grouping, even if a different semantic cut seems nicer.

## Person rules

- Use the global `subject_registry` as the only S-ID authority. Do not create,
  rename, merge, or split global IDs.
- Copy global importance tiers. Segment-local salience may change but may not
  promote a globally background person.
- Treat the `Authoritative ASR Timeline` as immutable. Copy every supplied
  `asr_id`, `text`, `time_range_sec`, and `source_time_range_sec` exactly once;
  never merge, split, paraphrase, retime, or invent speech.
- Map each ASR row to exactly one global S-ID using timestamp-specific video
  evidence. Synchronized visible mouth movement or repeated exclusive
  visual/audio co-occurrence is strong evidence. Screen presence, reaction
  shots, narrative expectation, gender, or age similarity is insufficient.
- Inspect multiple frames across the complete ASR interval. A registered
  subject whose stable visual cues match and whose mouth visibly moves through
  the line has direct evidence even if a noisy prior appearance interval did
  not include that timestamp.
- Use `unknown` for off-screen, occluded, or ambiguous speakers. Mark whether
  the audio is clearly single-speaker with `overlap_status`, and use
  `attribution_confidence=high` only with direct identity evidence. These
  fields gate later character audio assets, so do not guess for coverage.
- Set `attribution_evidence_type` to exactly one of `visible_lip_sync`,
  `exclusive_audio_visual_cooccurrence`, `repeated_voice_face_cooccurrence`,
  `context_only`, `voice_only`, `none`, or `unknown`. Context, dialogue
  semantics, age/gender, and an unverified voice impression are never direct
  identity evidence and must not receive high confidence.
- Report `speech_clarity` and `background_interference` as preliminary audible
  observations. Identity acceptance only makes a row a candidate; waveform,
  SNR, and speaker-purity validation must still pass before asset admission.
- `speaker_spk_to_s` is legacy-only. Return `[]` unless supplied evidence
  explicitly contains SPK labels; never invent a voice-cluster ID here.
- Use factual English and only supplied visible/audible evidence.

## Key-object consolidation rules

Shot captions contain unverified `key_object_candidates`, addressed by stable
references such as `SHOT_058:OC1`.

- Every supplied candidate reference must occur exactly once: either inside a
  segment `key_object_candidates[*].source_candidate_refs` or inside that
  segment's `rejected_object_candidate_refs`.
- Merge two references only when they plausibly describe the same physical
  object within this segment. Category equality alone is insufficient.
- Do not merge objects across segment boundaries and never emit `OBJ001` or any
  global object identity.
- Keep all results explicitly `unverified`; GroundingDINO, an interaction-aware
  instance matcher, and SAM must verify the physical instance later.
- Preserve short grounding noun phrases. Do not put S IDs, relations, actions,
  or sentences into `grounding_phrases`.
- `must`: directly manipulated object essential to the visible event, or a
  clearly indispensable narrative prop.
- `high`: strong interaction, state change, repeated appearance, framing
  emphasis, distinctive identity, or important visible information.
- `optional`: plausible but weakly important; keep for QC but not the first
  grounding pass.
- Reject generic furniture/background decoration, distant incidental objects,
  and clothing/accessories that remain part of character appearance.
- An object continuing across the next boundary may be listed in
  `object_continuity_to_next`, but this is a hint, not identity confirmation.

## Required JSON schema

```json
{
  "video_duration_sec": 0.0,
  "segments": [
    {
      "segment_id": 1,
      "description": "coherent dynamic summary",
      "scene_description": "static scene summary",
      "time_range_sec": [0.0, 20.0],
      "shot_ids": [1, 2],
      "event_phase": "introduction|continuation|escalation|turning_point|resolution|transition",
      "relation_to_previous": "N/A",
      "segment_caption": {
        "segment_description": "[Shot 1] At 00:00.000, ...",
        "overall_soundscape": "non-dialogue sound or N/A",
        "non_diegetic_music": "background score or N/A"
      },
      "key_subject_ids": ["S1"],
      "visible_count": 1,
      "semantic_role_summary": {
        "primary_subject_ids": [],
        "supporting_subject_ids": [],
        "incidental_subject_ids": [],
        "background_subject_ids": [],
        "unknown_subject_ids": []
      },
      "semantic_role_observations": [],
      "dialogue": [
        {
          "asr_id": "ASR identifier copied verbatim",
          "speaker_id": "S1|unknown",
          "time_range_sec": [0.0, 1.0],
          "source_time_range_sec": [100.0, 101.0],
          "language": "language code or unknown",
          "text": "verbatim ASR text",
          "delivery": "factual delivery",
          "is_off_screen": false,
          "overlap_status": "none|possible|overlap|unknown",
          "speech_clarity": "clean|usable|poor|unknown",
          "background_interference": "none|mild|strong|unknown",
          "attribution_confidence": "high|medium|low|unknown",
          "attribution_evidence_type": "visible_lip_sync|exclusive_audio_visual_cooccurrence|repeated_voice_face_cooccurrence|context_only|voice_only|none|unknown",
          "attribution_evidence": "timestamp-specific visible/audible evidence"
        }
      ],
      "speaker_spk_to_s": [],
      "on_screen_text": [],
      "boundary_reason": "semantic event transition or end of video",
      "key_object_candidates": [
        {
          "source_candidate_refs": ["SHOT_001:OC1"],
          "canonical_name": "phone",
          "grounding_phrases": ["black smartphone", "smartphone", "phone"],
          "priority": "must|high|optional|ignore",
          "reason_codes": ["character_interaction|narrative_prop|recurring|state_change|closeup|distinctive|text_information"],
          "visual_attributes": {"color": "black"},
          "related_subject_ids": ["S1"],
          "interactions": [
            {
              "subject_id": "S1",
              "action": "holding",
              "shot_ids": [1],
              "time_ranges_sec": [[0.2, 1.0]]
            }
          ],
          "occurrence_shot_ids": [1],
          "possible_states": ["screen_off", "screen_on"],
          "states_by_shot": [{"shot_id": 1, "state": "screen_on"}],
          "recurring_within_segment": false,
          "stateful": true,
          "segment_role": "narrative_prop|interaction_prop|recurring_prop|emphasized_prop|information_prop|uncertain",
          "confidence": 0.0,
          "proposal_status": "unverified"
        }
      ],
      "rejected_object_candidate_refs": [],
      "object_continuity_to_next": [
        {
          "source_candidate_refs": ["SHOT_002:OC1"],
          "canonical_name": "phone",
          "reason": "same visible interaction continues into the next shot"
        }
      ]
    }
  ],
  "semantic_role_summary": {
    "primary_subject_ids": [],
    "supporting_subject_ids": [],
    "incidental_subject_ids": [],
    "background_subject_ids": [],
    "unknown_subject_ids": []
  },
  "annotation_confidence": "high|medium|low|unknown"
}
```

Return `key_object_candidates: []`, `rejected_object_candidate_refs: []`, and
`object_continuity_to_next: []` when no shot candidate exists in a segment.
Before returning, verify duration limits, exact shot coverage, S-ID integrity,
and exact once-only accounting of all supplied object candidate references.
