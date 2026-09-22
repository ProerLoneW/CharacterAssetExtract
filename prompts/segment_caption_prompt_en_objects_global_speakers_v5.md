# MiMo Segment Caption + Key Objects + Read-Only Global Speakers V5

Partition the supplied ordered shots into coherent 15-30 second segments and
produce important segment-level semantic and object consolidation. Global
speaker-attributed ASR is supplied as read-only evidence. Output exactly one
JSON object without Markdown or reasoning.

## Boundaries

- Preserve every supplied shot exactly once and in chronological order.
- Cut only at shot boundaries and copy one partition from the authoritative
  `VALID BOUNDARY GUIDE` exactly.
- Every segment is at most 30 seconds. Prefer at least 15 seconds; a shorter
  final remainder is allowed only when merging would exceed 30 seconds.
- Prefer semantic event boundaries among the valid partitions.

## People and speech

- The global `subject_registry` is the only S-ID authority. Never create,
  rename, merge, or split S-IDs.
- `READ-ONLY GLOBAL SPEAKER-ATTRIBUTED ASR` already contains the final global
  speaker judgment. Use it to understand dialogue and events, but never revise
  `speaker_id`, confidence, evidence, text, or timing.
- Return `dialogue: []` in every model-generated segment. Local code attaches
  each authoritative ASR row to exactly one segment after validation. Do not
  repeat ASR text anywhere except a concise segment description when needed.
- Describe only visible/audible facts. Do not infer names, relationships,
  occupations, intentions, or emotions.

## Key objects

- Consolidate supplied shot-local candidates addressed as
  `SHOT_<id>:<local_object_id>`.
- Account for every supplied candidate exactly once in either
  `key_object_candidates[*].source_candidate_refs` or
  `rejected_object_candidate_refs`.
- Merge only the same plausible physical instance. Category equality is not
  enough. Keep every result `proposal_status=unverified`.
- Never add an object observed directly in the video unless it has a supplied
  `SHOT_<id>:<local_object_id>` reference. If the supplied inventory is empty,
  both object arrays must be empty.
- Reject generic background furniture, decoration, and worn clothing unless
  independently manipulated. Prefer 0-3 precise objects per segment.

## Required JSON schema

```json
{
  "video_duration_sec": 0.0,
  "segments": [
    {
      "segment_id": 1,
      "description": "coherent factual dynamic summary",
      "scene_description": "static scene summary",
      "time_range_sec": [0.0, 20.0],
      "shot_ids": [1, 2],
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
      "boundary_reason": "semantic event transition or end of video",
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

Before returning, verify exact shot coverage, a boundary-guide-compliant
partition, S-ID integrity, complete object-candidate accounting, and empty
model-generated dialogue arrays.
