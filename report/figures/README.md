# Figures

Drop the image files from the original report here using these filenames.
Any file that is missing renders as a labelled placeholder box, so the PDF
always builds.

| File | Original figure | Content |
|---|---|---|
| `fig1_pipeline.png` | Fig. 1 | Adaptive Mask Learning pipeline |
| `fig2_architecture.png` | Fig. 2 | Adaptive Mask Learning architecture |
| `fig_metrics_comparison.png` | Fig. 7 | Bar chart of Table 3 metrics |
| `fig_mask_ratio_schedule.png` | Fig. 4 / Fig. 6 | Learned ρ over training |
| `fig_val_loss.png` | Fig. 5 | Validation loss vs. training steps |
| `fig_qualitative_tracking_text2motion.png` | Fig. 9 | Baseline vs. AML frames: full-body tracking and text-to-motion |
| `fig_h1_steering.png` | Fig. 10 | H1 steering results |
| `fig_mask_generation_impl.png` | Fig. 11 | Mask-generation code screenshot |

Not carried over: the original Fig. 3 (nine-panel qualitative grid) and Fig. 8
(results summary), because their captions claimed experiments that were not run
or duplicated Table 3. If you have frames from the cartwheel or HumanML3D runs,
they can go in `fig_qualitative_tracking_text2motion.png`.

PDF or PNG both work; `\includegraphics` picks up either.
