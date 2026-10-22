# Slide visuals

**Preferred version: [native TensorBoard screenshots](tensorboard/).** These replace the styled chart images for the current slide. They retain TensorBoard's own formatting and include the run legend/value table. See that folder for six primary panels, two bonus panels, full-window screenshots, and replayable scalar event files.

The remaining guide describes the earlier Matplotlib figures, retained as alternatives.

These replace the small code screenshots with charts and diagrams that remain readable in the forecasting slide.

## Use on the slide

Use one image from `slide-visuals/wide/` across the space currently occupied by each pair of photos. Each is **1800 × 960 pixels**, matching the approximately 1.875:1 space. Preserve its aspect ratio rather than stretching it. The PNGs have transparent backgrounds so the slide's gray panels show through.

If keeping two separate picture slots, use the **1200 × 1200** images in `slide-visuals/square/`.

| Stage | Recommended wide image | Optional second square image |
| --- | --- | --- |
| V1 MLP | `01-v1-mlp-loss.png` | `01b-mlp-architecture.png` |
| V2 normalization | `02-v2-normalization.png` | `02b-normalized-mlp-loss.png` |
| V3 early backtest | `03-v3-backtest-results.png` | `03b-leakage-comparison.png` |
| V4 revised pipeline | `04-v4-gru-forecast.png` | `04b-corrected-gru-loss.png` |
| V5 CNN-BiGRU | `05-v5-cnn-bigru-loss.png` | `05b-cnn-bigru-architecture.png` |
| V6 Autoformer | `06-v6-autoformer-loss.png` | `06b-autoformer-architecture.png` |

The primary six also have square versions with the same filenames. SVG versions are included for resizing without loss of quality. For the two existing V5 slots, the square pipeline and square loss curve make a useful pair; the same pairing works for V6.

## What the visuals represent

- **NEW RUN:** real, short experiments performed on an NVIDIA GeForce RTX 4090 on October 22, 2026. These are small reconstructions of the model families, not the original checkpoints or exact V1–V6 implementations. Curves show actual recorded epoch values without smoothing or fabricated points.
- **DATA DEMO:** one observed training window from the archived Apple series, before and after Min-Max scaling. This illustrates the transformation, not improved prediction accuracy.
- **SCHEMATIC:** an explanatory diagram, not measured output. The Autoformer schematic separates trend and remainder, applies autocorrelation to the remainder path, and combines the paths for forecasting. It is a simplified view, not a complete layer diagram.
- **ORIGINAL LOG:** a bonus curve extracted from the original `Jul26_12-24-28_homelab` TensorBoard run. `Loss/train` and `Loss/Test` are preserved as logged. The exact model version, data split, and evaluation quality have not been verified, so this is not assigned to one of the six slide stages.

For the new forecast chart, the black line is the actual held-out price and the blue line is the model's one-step prediction. Each prediction receives the preceding observed prices; this is rolling one-step evaluation, not a 90-step forecast made at the start of the plot. The displayed 90 points are the final 90 held-out observations, selected as a fixed length for readability.

The optional leakage experiment intentionally puts the target in the final input slot. Its results are invalid as forecasting evidence; it illustrates the error. The corrected GRU keeps the target outside the input. In this short run, the leaked and corrected curves are similar: the curve does not reproduce the original project's dramatic false success.

The revised V3 primary visual shows the actual portfolio values from the archived `MLPTrader_2024-07-02_09-22-09_stats.csv`: **$500 to $75,482.38**, about **151×**. This is an unvalidated historical simulation, not real account growth. Its exact model-stage assignment and whether this specific run used the target-leakage bug have not been verified. The user's remembered billion-dollar spike was not found: all 89 archived stats CSVs and 47 saved tear sheets were checked. The largest positive total return reported in those tear sheets was 14,996%. The plotted dollar values come directly from the stats CSV, not from extrapolation.

## Fresh-run method

- Last 3,000 non-missing AAPL prices from the archived Feather file.
- Sequential 70% / 15% / 15% training, validation, and test split in archived row order. The original row index is unique and increasing; calendar dates were not recovered for this extracted series. Ordering follows the archive rather than independently verified dated records.
- Normalization fits only the training prices. A separate normalization illustration scales one observed training window to 0–1.
- 20 previous observations per next-price prediction. Autoformer receives 23 observations to support a 20-observation context plus its three lags.
- Five MSE models run for 65 epochs each; the reduced Hugging Face Autoformer runs for 35 epochs with negative log likelihood (NLL).
- Validation selects a checkpoint. Test observations are evaluated afterward and do not select the checkpoint.
- One fixed seed, no hyperparameter search, no uncertainty study. CUDA operations may not be fully deterministic across environments.

MSE curves use a logarithmic vertical axis. The unscaled MLP's MSE is in squared dollars; normalized-model MSE is in normalized units. Do not compare their vertical scales directly. The Autoformer uses NLL, a different objective that can be negative; lower is better.

## Results and limits

| Fresh model | Held-out MAE ($) |
| --- | ---: |
| Yesterday's price baseline | 1.954 |
| Unscaled MLP | 3.489 |
| Normalized MLP | 3.602 |
| Corrected GRU | 7.794 |
| CNN-BiGRU | 39.182 |
| Autoformer | 2.863 |

None of these valid fresh runs beat the baseline. They are useful illustrations of training and generalization, not evidence of a profitable strategy or an improvement from every model iteration. The deliberately leaked GRU is excluded from this comparison.

`results/` records metrics and the price values used to reproduce the charts. `train_slide_visuals.py` contains the fresh model definitions and training procedure; it expects the archived data path and a CUDA-enabled PyTorch environment. `render_slide_visuals.py` renders the figures from those records. No trading program was executed. Training used a new task directory on the server and left the source archive unchanged.
