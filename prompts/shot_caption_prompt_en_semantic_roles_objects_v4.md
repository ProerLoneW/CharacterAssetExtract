# MiMo Shot Caption + Semantic Roles + Key Object Proposals V4

Watch the complete supplied continuous shot. Produce a factual shot caption,
local semantic-role observations, and a small set of high-value object
proposals. Reuse the authoritative global person registry supplied in
`GLOBAL PERSON CONTEXT`. Output exactly one JSON object, without Markdown or
reasoning.

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

## Key-object proposal rules

`key_object_candidates` is a nomination list, not a detection result and not
an Object Registry. Never emit an `OBJ` identity here.

Include only objects satisfying at least one of these reasons:

1. directly manipulated, held, worn-and-removed, operated, driven, passed, or
   inspected by a narratively relevant person;
2. visibly causes or advances an event;
3. visibly changes state;
4. is deliberately emphasized by framing or close-up;
5. has distinctive visual identity useful across shots;
6. carries visible text or information important to the event;
7. is explicitly supported by supplied global context as recurring.

Do not list generic furniture, walls, floors, ordinary background decoration,
distant vehicles, every item on a table, or clothing/accessories that remain
part of a person's appearance. A hat, coat, glasses, or bag becomes an object
candidate only when independently manipulated or narratively important.

- Prefer 0-5 precise candidates. An empty list is correct when no key object
  exists.
- Use `must` only for direct character interaction or a clearly indispensable
  narrative prop; use `high` for strong state/recurrence/emphasis evidence.
- `grounding_phrases` must contain 1-4 short noun phrases ordered from specific
  to general, such as `["black smartphone", "smartphone", "phone"]`. Do not
  put relations, actions, S IDs, or full sentences in grounding phrases.
- Put relations in `interactions`, using a visible action and time interval.
- `related_subject_ids` may contain only supplied S IDs, `unknown`, or
  `new_subject_candidate`.
- `recurrence_hint=true` requires explicit global context or visible continuity
  evidence; do not infer recurrence from object category alone.
- Every candidate remains `unverified`; later GroundingDINO/SAM stages decide
  whether the physical instance exists and which instance is correct.

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
          "time_range_sec": [0.2, 1.0]
        }
      ],
      "visible_time_range_sec": [0.2, 1.0],
      "possible_states": ["screen_off", "screen_on"],
      "observed_state": "screen_on|unknown",
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
Return `key_object_candidates: []` when nothing meets the strict key-object
criteria.
