# PSE vs PSE-aligned VEGA-SR real-world validation

## Completion

- 80/80 formal cases completed: 2 methods × 2 datasets × 20 seeds.
- Final audit: 0 missing cases, 0 failed cases, 0 unresolved launcher failures, and 0 invariant violations.
- Both methods used a 90-second search budget. Test metrics were excluded from the explicit validation-score ranking. The VEGA-SR fitter nevertheless evaluated test-domain predictions while constructing candidate fit records, so this release does not claim that the test split was completely unaccessed during search.
- VEGA-SR used three planner/refinement rounds and three text-proposer calls per case.

## PSE-aligned change

For EMPS, VEGA-SR was given the same published physical prior used by PSE: Newton's second law with viscous and Coulomb friction,

`M*qddot = -Fv*qdot - Fc*sign(qdot) + tau - c`.

Three corresponding symbolic templates were inserted as protected candidates. Candidate ranking used PSE's reward `0.99^complexity / (1 + sqrt(MSE))`, calculated on the validation split. Test metrics were not part of that score; however, candidate fit records included test-domain evaluation and could reject numerically invalid expressions.

## Median results over 20 seeds

| Method | Dataset | Test MSE | Test NMSE | Test R² | Complexity | Runtime (s) |
|---|---|---:|---:|---:|---:|---:|
| PSE | EMPS | 0.009673 | 0.054017 | 0.945983 | 11 | 100.287 |
| PSE-aligned VEGA-SR | EMPS | 0.000651 | 0.003637 | 0.996363 | 12 | 21.140 |
| PSE | Roughpipe | 0.003579 | 0.066650 | 0.933350 | 14 | 94.710 |
| PSE-aligned VEGA-SR | Roughpipe | 0.001790 | 0.033317 | 0.966683 | 30 | 20.843 |

## Interpretation

- On EMPS, VEGA-SR reduces median test MSE by 93.27% and increases median test R² by 0.05038 relative to the reproduced PSE baseline, at a complexity cost of one node.
- On Roughpipe, VEGA-SR reduces median test MSE by 49.97% and increases median test R² by 0.03333, but its sixth-degree polynomial is 16 complexity units larger.
- All 20 EMPS runs selected the compact friction law `-2.0597*qdot + 0.3686*tau - 0.2137*sign(qdot) + 0.0294` using validation reward.
- Runtime excludes one-time VEGA model-service startup and is therefore descriptive rather than a pure compute-efficiency comparison.

## Protocol note

The PSE paper says that its EMPS Pareto-front expression was selected on the test dataset. This reproduction instead uses an additional validation split and excludes test metrics from the explicit ranking score, so it is not numerically identical to the paper figure. Because the current VEGA-SR fitter still evaluates test-domain predictions during candidate processing, this artifact does not make a strictly sealed-test claim.

The earlier no-domain-prior VEGA-SR run is preserved under `baseline_before_pse_aligned_prior_20260813/`.
