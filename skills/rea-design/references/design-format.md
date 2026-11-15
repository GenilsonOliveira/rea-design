# DESIGN.md contract (v0.1)

Produce a concise top-level file with links to supporting evidence. Coordinates use a top-left origin; bounding boxes are `[x, y, width, height]` in pixels and optionally normalized 0–1. Values are estimates unless measured. Use the user's language.

## Source and scope
Source identifier, dimensions, hash if available, output goal, and limitations. Link the evidence directory. Do not include secrets or private source paths in a shared report.

## Visual summary
Purpose, overall hierarchy, visual mood, reading order and composition. Separate visible observations from interpretation.

## Canvas and style tokens
Aspect ratio; background; palette with sampled colors and where they appear; type roles, approximate sizing, weight, tracking, leading, alignment; spacing; recurring shapes; image treatment, lighting and texture. Mark font matches as hypotheses. Sampling blended pixels does not reveal original colors or opacities.

## Layer map (back to front)
For each inferred layer: stable ID, visible role, bbox, evidence (crop/file or coordinates), confidence (high/medium/low), possible mask/occlusion, probable effects and editable reconstruction method. Represent groups and overlapping relationships explicitly, for example `subject > headline > background`; `A > B` means A visually covers B. Describe where order is ambiguous. The map is a reconstruction hypothesis, not recovered source layers. A single visible object may need multiple reconstruction layers.

## Reconstruction recipe
Ordered steps with dependencies, position, scale, type treatment, effects and assets. List exact values only when measured; otherwise label ranges or estimates. Include a way to substitute missing assets and preserve style when changing content.

## Validation and open questions
Link render and comparison files if produced. Identify the largest visible mismatches, what was actually verified, unknowns, and next experiments. Do not describe an unrendered plan as a finished design.
