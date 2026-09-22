# MiMo Fixed-Segment Caption + Key Objects + Read-Only Global Speakers V6

Watch the supplied global clip and caption every segment in the supplied
`FIXED UPSTREAM SEGMENTS` list. Segment boundaries, segment IDs, shot
membership, shot order, ASR text/timing, and speaker attribution are already
final. Your only task is semantic captioning and shot-object consolidation.
Output exactly one JSON object without Markdown or reasoning.

## Immutable upstream structure

- Return exactly one `segment_captions` row for every supplied
  `upstream_segment_id`, in the supplied order.
- Never create, delete, merge, split, reorder, resize, or plan segments.
- Do not output or propose `shot_ids`, `time_range_sec`, segment boundaries,
  boundary reasons, partition IDs, or duration plans. Local code owns them.
- `READ-ONLY GLOBAL SPEAKER-ATTRIBUTED ASR` is final evidence. Use its speech
  to understand events, but never revise speaker, confidence, evidence, text,
  or timing. Return `dialogue: []`; local code attaches each row afterward.
- The global `subject_registry` is the only S-ID authority. Never create,
  rename, merge, or split S-IDs.

## Caption content

- Write a concise factual dynamic description for each fixed segment.
- Write a static scene description and a shot-labeled chronological
  `segment_description` covering every supplied shot in that segment.
- Describe only visible or audible facts. Do not infer names, relationships,
  occupations, intentions, emotions, or events outside the supplied interval.
- `event_phase` describes the observed function of this fixed segment within
  the supplied global clip; it does not authorize a boundary change.
- `relation_to_previous` describes continuity with the preceding fixed
  segment. Use `N/A` for the first supplied segment.

## Key objects

- Consolidate only supplied shot candidates addressed as
  `SHOT_<id>:<local_object_id>`.
- Account for every candidate belonging to this fixed segment exactly once in
  either `key_object_candidates[*].source_candidate_refs` or
  `rejected_object_candidate_refs`.
- Never add an object merely because it is visible in the video. If a fixed
  segment has no supplied candidates, both object arrays must be empty.
- Merge only the same plausible physical instance. Category equality is not
  enough. Keep every result `proposal_status=unverified`.
- Reject generic background furniture, decoration, and worn clothing unless
  independently manipulated. Prefer 0-3 precise objects per fixed segment.

## Required JSON schema

```json
{
  "segment_captions": [
    {
      "upstream_segment_id": "seg_0001",
      "description": "coherent factual dynamic summary",
      "scene_description": "static scene summary",
      "event_phase": "introduction|continuation|escalation|turning_point|resolution|transition",
      "relation_to_previous": "N/A",
      "segment_caption": {
        "segment_description": "[Shot 1] ... [Shot 2] ...",
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
      "dialogue": [],
      "speaker_spk_to_s": [],
      "on_screen_text": [],
      "key_object_candidates": [
        {
          "source_candidate_refs": ["SHOT_001:OC1"],
          "canonical_name": "phone",
          "grounding_phrases": ["black phone", "phone"],
          "priority": "must|high|optional|ignore",
          "reason_codes": ["character_interaction"],
          "visual_attributes": {},
          "related_subject_ids": ["S1"],
          "interactions": [],
          "occurrence_shot_ids": [1],
          "possible_states": [],
          "states_by_shot": [],
          "recurring_within_segment": false,
          "stateful": false,
          "segment_role": "narrative_prop|interaction_prop|recurring_prop|emphasized_prop|information_prop|uncertain",
          "confidence": 0.0,
          "proposal_status": "unverified"
        }
      ],
      "rejected_object_candidate_refs": [],
      "object_continuity_to_next": []
    }
  ],
  "overall_semantic_role_summary": {
    "primary_subject_ids": [],
    "supporting_subject_ids": [],
    "incidental_subject_ids": [],
    "background_subject_ids": [],
    "unknown_subject_ids": []
  },
  "annotation_confidence": "high|medium|low|unknown"
}
```

Before returning, verify exact once-only coverage of every supplied
`upstream_segment_id`, empty dialogue arrays, S-ID integrity, complete
candidate accounting, and the absence of all segment-planning fields.
