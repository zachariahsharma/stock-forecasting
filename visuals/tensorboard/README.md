# Native TensorBoard screenshots

These are actual browser screenshots of TensorBoard's native Time Series panels. The chart styling, axes, colors, legends, and tables come from TensorBoard; the images were not redrawn or restyled.

Use the numbered PNGs directly on the slide. Each is a screenshot of an expanded chart card, including its native run legend/value table. `full-ui/` contains the corresponding full TensorBoard window screenshots.

| Slide section | Screenshot | Data source |
| --- | --- | --- |
| V1 MLP | `01-mlp-loss.png` | Fresh demonstration run, train/validation MSE |
| V2 normalized MLP | `02-normalized-mlp-loss.png` | Fresh demonstration run, train/validation MSE |
| V3 early backtest | `03-archived-backtest.png` | Original archived portfolio values; unvalidated simulation |
| V4 GRU | `04-gru-forecast.png` | Actual vs. predicted prices from the fresh held-out demonstration |
| V5 CNN-BiGRU | `05-cnn-bigru-loss.png` | Fresh demonstration run, train/validation MSE |
| V6 Autoformer | `06-autoformer-loss.png` | Fresh demonstration run, train/validation NLL |
| Optional historical loss | `07-original-2024-loss.png` | Original 2024 logged train/test loss; exact model version unverified |
| Optional V4 loss | `04b-gru-loss.png` | Fresh demonstration run, train/validation MSE |

## Display settings and provenance

- TensorBoard's native **Time Series** interface, light theme, expanded chart cards, linear axes.
- Smoothing set to **0**. Outlier exclusion disabled. Recorded points are shown directly.
- The recorded JSON metrics under `../results/` were exported to TensorBoard scalar event files for display. These are screenshots captured now, not historical screenshots taken during the original training sessions.
- `New_demo_*` run names identify the October 22, 2026 demonstration runs. `Original_2024_*` identify the recovered historical values. The model versions for the historical backtest and historical loss have not been independently verified.
- TensorBoard scalar values use float32, so the displayed last digits can differ slightly from the JSON records. For example, the backtest endpoint displays $75,482.375 rather than $75,482.37801000799.
- TensorBoard's **Relative** column reflects the metric-export event timestamps. It does not represent training runtime.
- The historical backtest is an unvalidated simulation, not real account growth or verified evidence of target leakage. Its recorded values rise from $500 to approximately $75,482; the remembered billion-dollar run was not located.

## Why Autoformer uses NLL

The fresh Autoformer run used Hugging Face's `AutoformerForPrediction` with `distribution_output="student_t"` and `loss="nll"`. It predicts a Student-t probability distribution rather than only a single price. Negative log-likelihood (NLL) rewards assigning high probability density to the observed target, accounting for both location and uncertainty. Lower NLL is better; for a continuous distribution, NLL can be negative.

The other demonstration models were trained to output one price using MSE. Their MSE values and the Autoformer's NLL values are different objectives and should not be compared numerically. No values were converted or relabeled to make the objectives look the same.

## Reopen the panels

The `events/` directory contains the scalar event files used for these screenshots. From the repository root, start TensorBoard with:

```sh
tensorboard --logdir visuals/tensorboard/events
```

The metric export and local server procedure is also preserved in `../serve_tensorboard_visuals.py`. It writes only its generated `tensorboard-recorded-metrics` directory next to the script; the source JSON and original server archive are not modified.
