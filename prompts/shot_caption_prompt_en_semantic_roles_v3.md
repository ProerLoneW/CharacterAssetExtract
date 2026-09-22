# MiMo Shot Caption + Semantic Role Evidence V3

Watch the complete supplied continuous shot. Produce a factual shot caption
and local semantic-role observations that reuse the authoritative global
person registry supplied in `GLOBAL PERSON CONTEXT`. Output exactly one JSON
object, without Markdown or reasoning.

## Rules

- Describe only visible/audible evidence in this shot; never infer identity,
  emotion, intention, relationship, occupation, or events outside the clip.
- Reuse global `S1`, `S2`, ... IDs. Never create a local S ID. Use `unknown` or
  `new_subject_candidate` when no registered identity can be supported.
- Copy each registered person's global `importance_tier`; do not promote a
  background person merely because this shot is a close-up or centers them.
- `local_salience` describes this shot only and is independent of global
  importance: `focus`, `active`, `reacting`, `present`, `background`, or
  `unknown`.
- A body part, handheld object, animal, reflection, poster, screen image,
  silhouette without person evidence, or foreground obstruction is not a
  semantic person.
- Record visual cues that can align a person to a detector Tracklet: screen
  side, clothing color/type, hair, face pose, accessories, body scale, motion,
  and the time interval where the cue is visible.
- Times are seconds relative to this shot unless absolute ASR times are
  explicitly supplied. Preserve supplied ASR text and timestamps verbatim.
- All generated prose is factual English; retain speech in its original
  language. Use `unknown` when evidence is insufficient.

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

Every person visible in the shot must occur in `subjects`, but only registered
semantic people belong in the role summary. Keep shot-local identity ambiguity
explicit. Do not use a generic phrase such as “a person” when an S ID is
visually supported, and do not force an S ID when it is not.
