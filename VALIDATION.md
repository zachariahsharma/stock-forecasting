# Packaging checks

Checked while preparing this portfolio on October 21, 2026:

- All seven included notebooks parse as JSON; their code outputs and execution counts were cleared.
- All four scripts in `src/` pass Python syntax parsing. Imports and model execution were not tested.
- Four distinct hardcoded credential values identified in the included originals have no remaining matches in the publication files. A separate credential-pattern scan found no additional matches.
- Nine code screenshots were rendered in a browser and reviewed for readability. Two saved plot images were extracted directly from the original notebook; the forecast plot is incomplete and is not recommended for the slide.
- Source paths and SHA-256 hashes of the original source bytes are recorded in `SOURCE_MANIFEST.json`.

These checks concern the archival packaging. They do not establish model correctness, dependency compatibility, leakage-free evaluation, statistical significance, or trading performance.
