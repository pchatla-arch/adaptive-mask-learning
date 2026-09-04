# Adaptive Mask Learning for MaskedMimic via Meta-RL

**Stanford CS224R (Deep Reinforcement Learning), Spring 2025 — final project**
Prasuna Chatla · pchatla@stanford.edu

[**Read the report (PDF)**](report/main.pdf)

## Summary

[MaskedMimic](https://research.nvidia.com/labs/par/maskedmimic/) trains a physics-based character controller by inpainting full-body motion from partially masked conditioning (joints, keyframes, text, objects). The masking schedule it uses during training is fixed in advance or sampled at random, and never responds to how the model is doing.

This project replaces that schedule with a **learned masking policy**. A small PPO-trained policy observes training signals — the change in validation loss, the gradient norm, and the entropy of the sampled mask — and outputs the masking ratio ρ for the next batch. Its reward is the improvement in validation loss plus a downstream imitation-accuracy term. The result is an emergent easy-to-hard curriculum: ρ starts near 0.2 and rises to ≈ 0.68 as the inpainting controller gets stronger.

Built on NVIDIA's [ProtoMotions](https://github.com/NVlabs/ProtoMotions) codebase; ~400 lines of changes across the training loop and mask generator.

## Results (AMASS, vs. cosine masking baseline)

| Method | L1 loss ↓ | MPJPE ↓ | FID ↓ | Accuracy ↑ | Slip err. ↓ |
|---|---|---|---|---|---|
| Random masking | 0.36 | 47.2 | 29.5 | 76.4% | 12.8% |
| Cosine masking | 0.31 | 42.7 | 23.1 | 80.5% | 10.3% |
| **Adaptive masking (ours)** | **0.10** | **29.6** | **12.4** | **92.1%** | **3.2%** |

- MPJPE −31%, FID −46%, action accuracy +11.6 points, foot-slip error ≈ 3× lower, all relative to the cosine schedule.
- Ablation: removing the validation-loss delta from the policy input is the most damaging change (MPJPE 29.6 → 33.1), identifying it as the dominant scheduling signal. Every learned variant beats a fixed ρ = 0.4 schedule (MPJPE 38.4).
- Generalization: coherent full-body motion on a held-out cartwheel sequence (MPJPE 45 → 39 mm from the fixed-mask checkpoint to the adaptive model) and on unseen HumanML3D text prompts.

Single training seed per configuration; see the Limitations section of the report.

## Repository layout

```
report/
  main.tex          LaTeX source of the report
  references.bib    bibliography
  figures/          figure images (see figures/README.md for the expected filenames)
  main.pdf          compiled report
```

Build the report with `cd report && latexmk -pdf main.tex` (TeX Live 2023 or later). Any figure whose image file is missing renders as a labelled placeholder so the document always compiles.

## Note on the course-site version

The copy of this report hosted on the CS224R course site is the original submission. It contains bibliography entries with placeholder author names, a leftover Lorem-ipsum appendix, some LaTeX artifacts, and a few internally inconsistent numbers (e.g. the MPJPE improvement was quoted three different ways). The version in this repository corrects those issues; the experiments, tables, and ablation are unchanged. A full list of corrections is in [ERRATA.md](ERRATA.md).

## Citation

```bibtex
@techreport{chatla2025aml,
  author = {Chatla, Prasuna},
  title  = {Adaptive Mask Learning for {MaskedMimic} via Meta-RL},
  note   = {Stanford CS224R course project},
  year   = {2025}
}
```
