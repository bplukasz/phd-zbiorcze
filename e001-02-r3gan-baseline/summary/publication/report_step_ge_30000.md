# Publication summary (step >= 30000)

- Scope: `artifacts-*/logs.csv` z filtrem `max(step) >= 30000`.
- Fair comparison point: metryki przy `step <= 30000` (preferowane dokladnie `30000`).
- Qualified runs: **18**.

## Top runs by FID@eval

| artifact_dir | recipe | max_step | fid_at_eval | kid_at_eval | precision_at_eval | recall_at_eval | rpse_at_eval | wbed_at_eval | sec_per_iter_tail20 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| artifacts-03-20-05-phase_b_r0_baseline_32 | phase_b_r0_baseline_32 | 30000 | 21.1904 | 0.0146 | 0.6396 | 0.4586 | 0.0002 | 0.0316 | 0.6569 |
| artifacts-03-10-01-phase_b_r0_baseline_32 | phase_b_r0_baseline_32 | 30000 | 21.3355 | 0.0154 | 0.6390 | 0.4492 | 0.0002 | 0.0233 | 1.0274 |
| artifacts-03-21-01-phase_b_r0_baseline_32 | phase_b_r0_baseline_32 | 30000 | 21.8369 | 0.0153 | 0.6355 | 0.4517 | 0.0002 | 0.0400 | 1.0392 |
| artifacts-03-16-01-phase_e_r8_waved_wavereg_sched_32 | phase_e_r8_waved_wavereg_sched_32 | 30000 | 21.9593 | 0.0153 | 0.6410 | 0.4542 | 0.0006 | 0.0175 | 1.0600 |
| artifacts-03-22-03-phase_f_r9_waved_gatewarm_32_seed43 | phase_f_r9_waved_gatewarm_32_seed43 | 30000 | 22.2751 | 0.0165 | 0.6171 | 0.4660 | 0.0005 | 0.0078 | 0.6890 |
| artifacts-03-16-02-phase_e_r9_waved_gatewarm_32 | phase_e_r9_waved_gatewarm_32 | 30000 | 22.3503 | 0.0164 | 0.6451 | 0.4543 | 0.0002 | 0.0507 | 1.0384 |
| artifacts-03-22-02-phase_g_r0_baseline_32_seed44 | phase_g_r0_baseline_32_seed44 | 30000 | 22.4049 | 0.0159 | 0.6247 | 0.4428 | 0.0002 | 0.0476 | 0.6493 |
| artifacts-03-14-01-phase_c_r6_fftreg_32 | phase_c_r6_fftreg_32 | 30000 | 22.4578 | 0.0167 | 0.6381 | 0.4655 | 0.0006 | 0.0181 | 1.0133 |
| artifacts-03-11-02-phase_b_r2_waved_32 | phase_b_r2_waved_32 | 30000 | 22.9342 | 0.0160 | 0.6222 | 0.4575 | 0.0002 | 0.0171 | 1.0492 |
| artifacts-03-11-01-phase_b_r1_matched_capacity_32 | phase_b_r1_matched_capacity_32 | 30000 | 22.9752 | 0.0171 | 0.6252 | 0.4568 | 0.0002 | 0.0144 | 1.0575 |

## Composite ranking (quality balance)

| artifact_dir | recipe | rank_mean | rank_fid | rank_kid | rank_precision | rank_recall | rank_rpse | rank_wbed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| artifacts-03-20-05-phase_b_r0_baseline_32 | phase_b_r0_baseline_32 | 4.667 | 1.000 | 1.000 | 5.000 | 5.000 | 3.000 | 13.000 |
| artifacts-03-16-01-phase_e_r8_waved_wavereg_sched_32 | phase_e_r8_waved_wavereg_sched_32 | 7.000 | 4.000 | 2.000 | 4.000 | 11.000 | 15.000 | 6.000 |
| artifacts-03-11-02-phase_b_r2_waved_32 | phase_b_r2_waved_32 | 7.000 | 9.000 | 6.000 | 14.000 | 7.000 | 1.000 | 5.000 |
| artifacts-03-10-01-phase_b_r0_baseline_32 | phase_b_r0_baseline_32 | 7.167 | 2.000 | 4.000 | 6.000 | 14.000 | 7.000 | 10.000 |
| artifacts-03-21-01-phase_b_r0_baseline_32 | phase_b_r0_baseline_32 | 7.333 | 3.000 | 3.000 | 10.000 | 12.000 | 2.000 | 14.000 |
| artifacts-03-16-02-phase_e_r9_waved_gatewarm_32 | phase_e_r9_waved_gatewarm_32 | 7.500 | 6.000 | 7.000 | 1.000 | 10.000 | 4.000 | 17.000 |
| artifacts-03-22-03-phase_f_r9_waved_gatewarm_32_seed43 | phase_f_r9_waved_gatewarm_32_seed43 | 7.667 | 5.000 | 8.000 | 16.000 | 1.000 | 14.000 | 2.000 |
| artifacts-03-14-01-phase_c_r6_fftreg_32 | phase_c_r6_fftreg_32 | 8.500 | 8.000 | 10.000 | 8.000 | 2.000 | 16.000 | 7.000 |
| artifacts-03-15-01-phase_e_r7_wavereg_sched_32 | phase_e_r7_wavereg_sched_32 | 8.833 | 11.000 | 13.000 | 3.000 | 6.000 | 12.000 | 8.000 |
| artifacts-03-11-01-phase_b_r1_matched_capacity_32 | phase_b_r1_matched_capacity_32 | 9.500 | 10.000 | 14.000 | 12.000 | 9.000 | 8.000 | 4.000 |

## Claim readiness by recipe

| recipe | n_runs | n_unique_seeds | median_fid | best_fid | median_kid | claim_status | claim_note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| phase_b_r0_baseline_32 | 4 | 1 | 21.5862 | 21.1904 | 0.0154 | baseline_anchor | Referencja porownawcza. |
| phase_e_r8_waved_wavereg_sched_32 | 1 | 1 | 21.9593 | 21.9593 | 0.0153 | promising_single_seed | Blisko baseline, ale brak mocy statystycznej (za malo seedow). |
| phase_f_r9_waved_gatewarm_32_seed43 | 1 | 1 | 22.2751 | 22.2751 | 0.0165 | promising_single_seed | Blisko baseline, ale brak mocy statystycznej (za malo seedow). |
| phase_e_r9_waved_gatewarm_32 | 1 | 1 | 22.3503 | 22.3503 | 0.0164 | promising_single_seed | Blisko baseline, ale brak mocy statystycznej (za malo seedow). |
| phase_g_r0_baseline_32_seed44 | 1 | 1 | 22.4049 | 22.4049 | 0.0159 | promising_single_seed | Blisko baseline, ale brak mocy statystycznej (za malo seedow). |
| phase_c_r6_fftreg_32 | 1 | 1 | 22.4578 | 22.4578 | 0.0167 | promising_single_seed | Blisko baseline, ale brak mocy statystycznej (za malo seedow). |
| phase_b_r2_waved_32 | 1 | 1 | 22.9342 | 22.9342 | 0.0160 | ablation_only | Brak przewagi i/lub za malo seedow. |
| phase_b_r1_matched_capacity_32 | 1 | 1 | 22.9752 | 22.9752 | 0.0171 | ablation_only | Brak przewagi i/lub za malo seedow. |
| phase_e_r7_wavereg_sched_32 | 1 | 1 | 23.1124 | 23.1124 | 0.0169 | ablation_only | Brak przewagi i/lub za malo seedow. |
| phase_c_r4_waved_fftreg_32 | 1 | 1 | 23.1217 | 23.1217 | 0.0167 | ablation_only | Brak przewagi i/lub za malo seedow. |
| phase_f_r8_waved_wavereg_sched_32_seed43 | 1 | 1 | 23.1272 | 23.1272 | 0.0165 | ablation_only | Brak przewagi i/lub za malo seedow. |
| phase_e_r10_waved_wavereg_combo_32 | 1 | 1 | 23.5177 | 23.5177 | 0.0167 | ablation_only | Brak przewagi i/lub za malo seedow. |
| phase_e_r11_wavereg_fidgate_32 | 1 | 1 | 24.4362 | 24.4362 | 0.0176 | ablation_only | Brak przewagi i/lub za malo seedow. |
| phase_c_r5_wavereg_32 | 1 | 1 | 24.7034 | 24.7034 | 0.0192 | ablation_only | Brak przewagi i/lub za malo seedow. |
| phase_c_r3_waved_wavereg_32 | 1 | 1 | 24.8504 | 24.8504 | 0.0185 | ablation_only | Brak przewagi i/lub za malo seedow. |

## Interpretation

- Best run at eval step: `artifacts-03-20-05-phase_b_r0_baseline_32` with FID `21.1904` and KID `0.014559`.
- Best non-baseline run: `artifacts-03-16-01-phase_e_r8_waved_wavereg_sched_32`; gap vs best baseline: `+0.7689` FID.
- Long-run instability detected in `artifacts-03-19-02-phase_b_r0_baseline_32` (FID@30000 `28.3037` -> final `78.4275`).
- Publication stance: ablation-ready now; main-claim readiness requires >=3 seeds for top candidate recipes.
