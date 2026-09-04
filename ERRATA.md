# Errata: corrections relative to the CS224R course-site submission

The experiments, tables, and ablation are unchanged. The following issues in the
write-up were corrected in `report/main.tex`.

## Bibliography

Rebuilt from the original publications. Entries that had placeholder or incorrect
authors were replaced with the papers actually being referred to:

| Original citation | Corrected to |
|---|---|
| "Winkler and Doe (2022), Physics-Based VR Tracking" | Winkler, Won, Ye — *QuestSim*, SIGGRAPH Asia 2022 |
| "Rempe and Smith (2023), Terrain-Aware Walking" | Rempe et al. — *Trace and Pace*, CVPR 2023 |
| "Hassan and Zhang (2023), Physically-Based Object Interaction" | Hassan et al. — *Synthesizing Physical Character-Scene Interactions* (InterPhys), SIGGRAPH 2023 |
| "Juravsky and Lee (2022), Text-Driven Control" | Juravsky, Guo, Fidler, Peng — *PADL*, SIGGRAPH Asia 2022 |
| "Luo and colleagues (2024)" | Luo et al. — *Universal Humanoid Motion Representations*, ICLR 2024 |
| MaskedMimic (cited twice, two different author lists, CVPR) | Tessler, Guo, Nabati, Chechik, Peng — ACM TOG / SIGGRAPH Asia 2024 |
| HumanML3D (cited twice, different authors, ECCV) | Guo et al. — *Generating Diverse and Natural 3D Human Motions from Text*, CVPR 2022 |
| SAMP (wrong title/authors) | Hassan et al. — *Stochastic Scene-Aware Motion Prediction*, ICCV 2021 |
| CASE (Tessler and Tamar) | Dou et al. — *C·ASE*, SIGGRAPH Asia 2023 |
| CALM (uncited) | Tessler et al., SIGGRAPH 2023 |
| PhysHOI, UniHSI, PACER++ (wrong authors) | Wang et al. 2023; Xiao et al., ICLR 2024; Wang et al., CVPR 2024 |
| "Wang and Liu (2021)", "Xu and Jiang (2023)" (kinematic HOI) | Wang et al., CVPR 2021; Xu et al. — *InterDiff*, ICCV 2023 |
| Lee et al. (2010) | Lee et al., SIGGRAPH 2002 (correct year/venue) |
| Liu et al. (2010), wrong title | *Sampling-Based Contact-Rich Motion Control*, SIGGRAPH 2010 |
| MDM (CVPR) | Tevet et al., ICLR 2023 |

Added citations for PPO, Isaac Gym, SMPL, ProtoMotions, and curriculum learning.

## Content

- Removed Appendix A ("Additional Experiments"), which contained only Lorem-ipsum placeholder text.
- Removed LaTeX artifacts: a stray `float` under the Results heading, `placeins float[H]` in §5.3, a broken `Figure ??` reference, and a paragraph duplicated verbatim in Related Work.
- Section 5.1 / Figure 3 previously listed nine qualitative generalization tasks (weather-aware locomotion, multi-agent mask coordination, zero-shot morphology transfer, etc.) that were not evaluated. The generalization section now reports only what was run: the held-out cartwheel, unseen HumanML3D text prompts, the high-sparsity setting, and the H1 steering check. VR-style and SAMP settings are stated as qualitative only.
- Added an explicit statement that the scheduler controls a scalar masking ratio ρ, whereas MaskedMimic's native masking is structured (per joint / frame / modality).
- Added a Scope item under Limitations (single seed per configuration).

## Numbers

- **MPJPE improvement** was stated as 37% (extended abstract), 72% (§5.2.1), and 14% (appendix). It is now consistently −31% vs. cosine (42.7 → 29.6), with −37% vs. random noted once. The 72% figure did not correspond to any table and was removed. The appendix's 14% referred to the separate cartwheel hold-out run (45 → 39 mm) and is now labelled as such.
- **MPJPE units**: Table 3 was labelled "cm"; values of 30–47 cm are not consistent with a working tracker or with the appendix's 3.9–4.5 cm cartwheel numbers. Now labelled mm throughout.
- **Baseline for relative claims**: all headline improvements are now stated vs. the cosine schedule (previously a mix of cosine and random).
- **Parallel environments**: 512 in the extended abstract vs. 128 elsewhere → 128 throughout.
- **GPU memory**: "A100 with 100 GB VRAM" removed (A100s ship with 40 or 80 GB); now "a single NVIDIA A100".
- **Policy input range**: ρ ∈ [0.1, 0.9] in the method vs. [0, 1] in the appendix → [0.1, 0.9] throughout.
- **Policy width**: "64–128" vs. 128 → 128.
- **Accuracy**: the appendix's "≈95%" conflicted with Table 3's 92.1% and was removed; Table 3 is the reference.
- **Fixed-ρ baseline**: ρ = 0.5 is now described as the Phase-1 warm-up shared by all methods; ρ = 0.4 remains the fixed-schedule row in the ablation table. The fixed-ρ "baseline" was removed from the comparison list since it does not appear in Table 3.
- **Compute**: the original implied the full-body tracker π_FC was trained from scratch in ~3 hours. The report now states that π_FC was initialized from the pretrained ProtoMotions checkpoint, and that the ~9.5 h total covers the two distillation/scheduling phases.
