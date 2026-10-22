# Screenshot guide

**Updated October 22:** use the [new slide visuals](visuals/README.md) for the presentation. Those charts and diagrams are designed for the small picture areas. The guide below refers to the older source-code screenshots, retained as code evidence.

Use the numbered images in `screenshots/`. The first six follow the slide's V1–V6 reading order. Keep the image subtitle visible: it records what was actually recovered.

| Slide section | Primary image | Optional second image | Evidence limit |
| --- | --- | --- | --- |
| V1 MLP | `01-v1-mlp-foundations.png` | — | Related classification learning exercise, not the original stock MLP. |
| Normalized V2 | `02-v2-minmax-normalization.png` | — | Later normalization code from the GRU notebook, not the original V2 MLP. |
| V3 GRU | `03-v3-gru-architecture.png` | — | Recovered GRU baseline; exact V3 correspondence unverified. |
| GRU 2.0 V4 | `04-v4-data-pipeline.png` | `07-v4-dataset-audit.png` | Revised preprocessing code; does not prove all leakage issues were fixed. |
| V5 CNN-BiGRU | `05-v5-cnn-bigru.png` | `08-v5-forward-pass.png` | Recovered TCN/BiGRU variant uses BatchNorm/ReLU, not MaxPool. |
| Autoformer V6 | `06-v6-autoformer.png` | `09-v6-custom-transformer.png` | Separate experimental architectures; no verified predictive accuracy. |

`10-v6-saved-forecast.png` is an incomplete original plot: its legend lists a forecast, but no useful forecast is visible. It is retained as archive evidence and is not recommended for the slide. `11-v6-saved-metrics.png` is an original saved MASE/sMAPE scatter, included for completeness; the points alone do not establish a valid evaluation.

The last two images are extracted original plots. The other nine are browser screenshots of formatted source excerpts, with line numbers local to each excerpt. `SOURCES.json` maps each image to its original notebook cell or file.

## Slide wording checks

- V1/V2: the write-up describes these stages, but their exact stock MLP code was not found. Keep the caveat if using the related examples.
- V4: “reworked the pipeline to address target leakage” is supported by the project history. “Prevented data leakage” would imply verification beyond this archive review.
- V5: the recovered variant uses three dilated convolution blocks before a BiGRU. The slide can describe local feature extraction and bidirectional sequence modeling without specifying MaxPool.
- V6: the archive contains a Hugging Face Autoformer configuration and separate custom transformer experiments. An end-to-end, validated Autoformer training run was not established.
- Results: the supplied write-up and recovered archive do not establish profitable trading or statistically significant accuracy.

Suggested layout: one large code image per stage, with a small second image only where there is room. At the slide's small image sizes, an extra code image is less readable than a single close crop.
