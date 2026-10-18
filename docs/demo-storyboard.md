# Demo storyboard

Phase 3 builds the site. Phase 2 writes the replay export that site will play.

A judge-facing run is a recorded export (`mode` stays whatever the run actually
was: `mock` or `hybrid`). Playback must not be labeled live. The lineage view
reads `parent_ids`, `operator`, `family`, `species`, fitness, feasibility,
split, and provenance from the event log. It does not synthesize nodes.

Suggested two-minute order once those exports exist: cloud lineage, mutation
and crossover inspection, held-out score against the one-shot arm, then the
same controller on shopping. That UI is intentionally absent here.
