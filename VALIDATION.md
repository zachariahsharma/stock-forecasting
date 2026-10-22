# Packaging checks

Checked while preparing this portfolio on October 21, 2026:

- All seven included notebooks parse as JSON; their code outputs and execution counts were cleared.
- All four scripts in `src/` pass Python syntax parsing. Imports and model execution were not tested.
- Four distinct hardcoded credential values identified in the included originals have no remaining matches in the publication files. A separate credential-pattern scan found no additional matches.
- Nine code screenshots were rendered in a browser and reviewed for readability. Two saved plot images were extracted directly from the original notebook; the forecast plot is incomplete and is not recommended for the slide.
- Source paths and SHA-256 hashes of the original source bytes are recorded in `SOURCE_MANIFEST.json`.

These checks concern the archival packaging. They do not establish model correctness, dependency compatibility, leakage-free evaluation, statistical significance, or trading performance.

## Added October 22: presentation visuals

- Six small model-family demonstration runs completed using the RTX 4090. Five used MSE and one used the Hugging Face Autoformer with NLL.
- Recomputed test MAE from every prediction file; verified finite histories, matching held-out targets, expected split sizes, and source-series hash.
- Verified scaler bounds use the training portion only. Normal inputs end before the target; V3 is explicitly an intentionally leaked demonstration.
- Source row indices are unique and increasing. Calendar dates were not recovered, so the documented split follows archive row order.
- New charts are explicitly labeled NEW RUN; diagrams are SCHEMATIC; an original TensorBoard loss curve is ORIGINAL LOG with model assignment unverified.
- Six wide images are 1800 × 960; twelve primary/alternative square images are 1200 × 1200. PNGs have transparent backgrounds, and SVGs are also included. Bonus original-log figures use the same dimensions.
- The new valid model runs all perform worse than the carry-forward baseline on held-out MAE; the deliberately leaked run is not treated as a valid forecasting result.
