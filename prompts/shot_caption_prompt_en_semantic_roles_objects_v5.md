# MiMo Shot Caption + Atomic Key Object Proposals V5

Watch the complete supplied continuous shot. Produce a factual shot caption,
local semantic-role observations, and a small set of high-value, physically
groundable object proposals. Reuse the authoritative global person registry
supplied in `GLOBAL PERSON CONTEXT`. Output exactly one JSON object, without
Markdown or reasoning.

## General rules

- Describe only visible/audible evidence in this shot; never infer identity,
  emotion, intention, relationship, occupation, or events outside the clip.
- Reuse global `S1`, `S2`, ... IDs. Never create a local S ID. Use `unknown` or
  `new_subject_candidate` when no registered identity can be supported.
- Copy each registered person's global `importance_tier`; do not promote a
  background person merely because this shot centers them.
- Record person cues useful for Tracklet alignment: screen side, clothing,
  hair, face pose, accessories, body scale, motion, and visible time interval.
- Times are seconds relative to this shot unless absolute ASR times are
  explicitly supplied. Preserve supplied ASR text and timestamps verbatim.
- All generated prose is factual English; retain speech in its original
  language. Use `unknown` when evidence is insufficient.

## Atomic key-object proposal rules

`key_object_candidates` is a nomination list, not a detection result and not
an Object Registry. Never emit an `OBJ` identity here.

A candidate must satisfy both tests:

1. it is one physical instance that a detector/segmenter can localize; and
2. it matters to an action, visible state change, close-up, readable event
   information, or explicitly supplied cross-shot continuity.

Valid examples are one microphone held by a child, the particular record
sleeve removed from a rack, one card being pinned to a board, or one pool float
supporting a person. Name the physical instance, not its semantic collection.

Exclude:

- collections or plural concepts such as `notes`, `cards`, `records`, or
  `sheet music`; nominate the particular manipulated item instead;
- supports/containers/background anchors such as a shelf, table, wall, or
  noticeboard when only their contents are manipulated;
- every visually prominent background object or every object carrying text;
- clothing and worn accessories that remain appearance cues;
- an object that is absent during the claimed interval;
- a broad category when several same-class instances are visible and the
  intended instance cannot be distinguished by appearance or interaction.

A worn accessory may be proposed only when it is independently touched,
removed, put on, handed over, or visibly drives the event. If it is proposed,
describe the independent action rather than merely `wearing` it.

Candidate time rules:

- `visible_time_range_sec` must cover only the interval in which the nominated
  physical instance is actually visible. Do not default to the full shot.
- Each interaction time must lie inside that visible interval.
- If the object enters late, leaves early, or is fully occluded, reflect that.
- If sampled evidence cannot establish the interval, lower confidence instead
  of inventing full-shot visibility.

Selection and grounding rules:

- Prefer 0-3 precise candidates. An empty list is correct when no atomic key
  object exists.
- Use `must` only for direct, independent character interaction or an
  indispensable narrative prop. Use `high` for strong state, recurrence, or
  deliberate close-up evidence.
- `grounding_phrases` must contain 1-4 short noun phrases ordered from most
  instance-specific to general. Include visible color, material, text, or
  distinctive form when it separates the target from nearby same-class items.
- Do not put actions, relations, S IDs, or full sentences in grounding phrases.
- Put the binding relation in `interactions`, with a visible action and time.
- `related_subject_ids` may contain only supplied S IDs, `unknown`, or
  `new_subject_candidate`.
- `reason_codes` must match the evidence. Never emit `character_interaction`
  when `interactions` is empty.
- `recurrence_hint=true` requires supplied global context or visible continuity
  evidence. In a shot-only input, false means “not established here”, not
  “known not to recur”.
- Every candidate remains `unverified`; later grounding stages decide whether
  the instance exists and whether the selected instance is correct.

## Required JSON schema

```json
{
  "shot_caption": {
    "shot_description": "2-4 factual chronological sentences using S IDs",
    "shot_scene_description": "one static-scene sentence",
    "overall_soundscape": "non-dialogue audio or N/A",
    "non_diegetic_music": "background score or N/A"
  },
  "shot_type": "extreme wide shot|wide shot|medium-wide shot|medium shot|medium close-up|close-up|extreme close-up|over-the-shoulder|POV|screen capture|unknown",
  "camera_motion": "factual phrase or unknown",
  "composition": "screen layout and subject positions",
  "setting": "visible setting",
  "visible_count": 0,
  "subjects": [
    {
      "id": "S1|unknown|new_subject_candidate",
      "kind": "person|animal|vehicle|object|group|unknown",
      "description": "observable local appearance",
      "importance_tier": "primary|supporting|incidental|background|unknown",
      "local_salience": "focus|active|reacting|present|background|unknown",
      "visual_identity_cues": [],
      "visible_time_range_sec": [0.0, 1.0]
    }
  ],
  "semantic_role_summary": {
    "primary_subject_ids": [],
    "supporting_subject_ids": [],
    "incidental_subject_ids": [],
    "background_subject_ids": [],
    "unknown_subject_ids": []
  },
  "semantic_role_observations": [
    {
      "subject_id": "S1|unknown|new_subject_candidate",
      "importance_tier": "primary|supporting|incidental|background|unknown",
      "local_salience": "focus|active|reacting|present|background|unknown",
      "narrative_functions": [],
      "time_range_sec": [0.0, 1.0],
      "screen_position": "left|center|right|full_frame|off_screen|unknown",
      "visual_identity_cues": [],
      "confidence": "high|medium|low|unknown"
    }
  ],
  "visual_events": [
    {
      "time_range_sec": [0.0, 1.0],
      "subject_ids": ["S1"],
      "action": "visible action",
      "state_change": "visible change or none",
      "event_phase": "introduction|continuation|escalation|turning_point|resolution|transition",
      "continuation": "starts_here|continues_from_previous_shot|new_development|turning_point|resolves_here"
    }
  ],
  "key_object_candidates": [
    {
      "local_object_id": "OC1",
      "canonical_name": "record sleeve",
      "grounding_phrases": ["record sleeve held by the man", "record sleeve"],
      "priority": "must|high|optional|ignore",
      "reason_codes": ["character_interaction|narrative_prop|recurring|state_change|closeup|distinctive|text_information"],
      "visual_attributes": {"color": "visible color or unknown"},
      "related_subject_ids": ["S1"],
      "interactions": [
        {
          "subject_id": "S1",
          "action": "holding",
          "time_range_sec": [0.2, 1.0]
        }
      ],
      "visible_time_range_sec": [0.2, 1.0],
      "possible_states": ["on rack", "held"],
      "observed_state": "held|unknown",
      "recurrence_hint": false,
      "confidence": "high|medium|low|unknown",
      "proposal_status": "unverified"
    }
  ],
  "dialogue": [
    {
      "speaker_id": "S1|unknown",
      "time_range_sec": [0.0, 1.0],
      "language": "language code or unknown",
      "text": "verbatim speech",
      "delivery": "calm speech|loud speech|whisper|singing|shouting|unknown",
      "is_off_screen": false,
      "asr_text": "",
      "asr_time_range_sec": [0.0, 1.0]
    }
  ],
  "on_screen_text": [],
  "diegetic_audio": "factual description or unknown",
  "candidate_subject_ids": [],
  "continuity_hint": "continues_from_previous_shot|starts_here|new_development",
  "annotation_confidence": "high|medium|low|unknown"
}
```

Every visible person must occur in `subjects`, but only registered semantic
people belong in the role summary. Keep shot-local identity ambiguity explicit.
Return `key_object_candidates: []` when nothing meets the atomic key-object
criteria.
