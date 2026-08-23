# PSE real-world experiment: paper-ready configuration and results

Generated: 2026-08-23 13:21:20 CST

Status: 全部正式实验已完成。

## 1. Experiment scope

This experiment compares PSE, PSE-aligned VEGA-SR, PySR, and Operon on two real-world symbolic-regression tasks from the PSE study. Physics-LS is reported separately as an EMPS prior-only ablation rather than as a general symbolic-regression baseline. The goal is to compare predictive accuracy, symbolic complexity, and descriptive runtime under a common data split and a validation-only expression-selection protocol.

## 2. Reproducible protocol

- Repeats: 20 seeds (`0`–`19`) for PSE, VEGA-SR, PySR, and Operon. Physics-LS is deterministic and is run once.
- Search budget: 90 seconds per formal case. Runtime includes method initialization and search inside each case, but excludes one-time VEGA model-service startup.
- Selection: candidates are ranked using validation data with $0.99^C/(1+\sqrt{\mathrm{MSE}_{val}})$, where $C$ is the common SymPy-tree complexity; test metrics are not part of that ranking score. The current VEGA-SR fitter nevertheless evaluates candidate predictions on the test inputs while constructing fit records and may reject a numerically invalid expression, so this implementation does not satisfy the stronger claim that the test split remains completely unaccessed during search.
- Primary metrics: test MSE, NMSE, $R^2$, expression complexity, and descriptive runtime.
- Data and source pinning: PSE commit `7105caba63e754150dd3b160443984456ec99cd7` and VEGA-SR reference commit `8187fd36d56092f0f0c8c31b16362989a6c61374`. Raw files are verified by the SHA-256 hashes in the YAML configuration.

### Data splits

| Dataset | Train | Validation | Test | Split rule |
|---|---:|---:|---:|---|
| EMPS | 4,000 | 1,000 | 5,001 | First half is discovery data; its first 80% is train and last 20% is validation; the untouched second half is test. |
| Roughpipe | 218 | 72 | 72 | After PSE's one-dimensional transform and stable sorting by $x$, samples follow a fixed repeating `train, train, train, val, test` assignment. |

### Method configurations

- **PSE:** released stage configurations (`EMPS.yaml` and `turbulence.yaml`), GP token generator, released operator sets, and PSE reward parameter $\eta=0.99$.
- **PSE-aligned VEGA-SR:** planner-guided full pipeline, three planner/refinement rounds, restored text proposer, 90-second deadline, and validation-only selection. For EMPS it receives the published Newton/friction prior $M\ddot q=-F_v\dot q-F_c\operatorname{sign}(\dot q)+\tau-c$ through three protected templates.
- **PySR:** PSE-released operator libraries; deterministic serial execution pinned to one CPU core with one Julia/BLAS thread and a 90-second timeout. The released maximum search work is preserved as 100 × 380 mutation cycles for EMPS and 50 × 380 for Roughpipe. For PySR 1.5, each cycle is made an interruptible iteration (`ncycles_per_iteration=1`, effective iteration caps 38,000 and 19,000) so that pathological seeds cannot overrun the same wall-clock deadline by several minutes. Explicit one-thread environment limits and CPU affinity implement the released script's disabled-multithreading intent on this host.
- **Operon:** PSE-released operators; objectives `(mse, length)`; population and pool size 1,000; four CPU threads; Levenberg–Marquardt local optimization; 90-second maximum time.
- **Physics-LS:** the same three EMPS physical templates supplied to VEGA-SR, with coefficients fitted by ordinary least squares. It performs no symbolic structure search.

EMPS operators are `+`, `-`, `*`, `/`, `sin`, `cos`, `exp`, `log`, `cosh`, `tanh`, `abs`, and `sign`. Roughpipe uses `+`, `-`, `*`, `/`, `sin`, `cos`, `exp`, `log`, `cosh`, `tanh`, square, and cube.

## 3. Main results

Values are median [Q1, Q3] over 20 seeds, except deterministic Physics-LS. Lower MSE/NMSE/complexity is better; higher $R^2$ is better.
For a validation-selected expression that is numerically invalid on held-out inputs, test metrics remain missing rather than selecting a fallback using test data; the `n` column reports the number of finite test predictions when this occurs.

| Method | Dataset | n | Test MSE | Test NMSE | Test R² | Complexity | Runtime (s) |
|---|---|---|---|---|---|---|---|
| PSE | EMPS | 20 | 0.009673 [0.009673, 0.009673] | 0.054017 [0.054017, 0.054017] | 0.945983 [0.945983, 0.945983] | 11.0 [11.0, 11.0] | 100.29 [97.38, 105.86] |
| PySR | EMPS | 20 (19 valid) | 0.010609 [0.010101, 0.050061] | 0.059245 [0.056409, 0.279562] | 0.940755 [0.720438, 0.943591] | 8.0 [7.0, 8.0] | 142.66 [141.68, 145.75] |
| Operon | EMPS | 20 | 0.010101 [0.010101, 0.010101] | 0.056410 [0.056410, 0.056410] | 0.943590 [0.943590, 0.943590] | 8.0 [8.0, 8.0] | 90.31 [90.28, 90.52] |
| PSE-aligned VEGA-SR | EMPS | 20 | 0.000651 [0.000651, 0.000651] | 0.003637 [0.003637, 0.003637] | 0.996363 [0.996363, 0.996363] | 12.0 [12.0, 12.0] | 21.14 [20.91, 21.81] |
| Physics-LS (ablation) | EMPS | 1 | 0.000651 | 0.003637 | 0.996363 | 12.0 | 0.00 |
| PSE | Roughpipe | 20 | 0.003579 [0.002148, 0.009005] | 0.066650 [0.040003, 0.167709] | 0.933350 [0.832291, 0.959997] | 14.0 [11.0, 16.0] | 94.71 [92.92, 96.38] |
| PySR | Roughpipe | 20 | 0.004175 [0.002890, 0.007001] | 0.077698 [0.053783, 0.130279] | 0.922302 [0.869721, 0.946217] | 11.0 [10.0, 13.0] | 143.39 [142.44, 145.42] |
| Operon | Roughpipe | 20 | 0.002742 [0.002201, 0.004841] | 0.051071 [0.041000, 0.090151] | 0.948929 [0.909849, 0.959000] | 12.0 [11.8, 12.2] | 90.30 [90.21, 90.40] |
| PSE-aligned VEGA-SR | Roughpipe | 20 | 0.001790 [0.001790, 0.001790] | 0.033317 [0.033317, 0.033317] | 0.966683 [0.966683, 0.966683] | 30.0 [30.0, 30.0] | 20.84 [20.51, 21.74] |

### Relative to reproduced PSE

Positive MSE reduction and positive $R^2$ difference favor the compared method. Complexity difference is compared with PSE.

| Method | Dataset | Median MSE reduction | Median R² difference | Complexity difference |
|---|---|---|---|---|
| PySR | EMPS | -9.68% | -0.00523 | -3.0 |
| Operon | EMPS | -4.43% | -0.00239 | -3.0 |
| PSE-aligned VEGA-SR | EMPS | +93.27% | +0.05038 | +1.0 |
| PySR | Roughpipe | -16.67% | -0.01105 | -3.0 |
| Operon | Roughpipe | +23.38% | +0.01558 | -2.0 |
| PSE-aligned VEGA-SR | Roughpipe | +49.97% | +0.03333 | +16.0 |

## 4. Representative median-run expressions

The representative run is the seed whose test MSE is closest to the method's median; it is for readability and is not an additional selection step.

| Method | Dataset | Seed | Expression | Test MSE |
|---|---|---|---|---|
| PSE | EMPS | 0 | `0.118*x0 - 4.022*x1 + 0.354*x2 + 0.018` | 0.009673 |
| PySR | EMPS | 8 | `(-qdot + tau*0.08905808)*3.8902402` | 0.010609 |
| Operon | EMPS | 1 | `-4.02557898177729*qdot + 0.353528881913982*tau + 0.038255915046` | 0.010101 |
| PSE-aligned VEGA-SR | EMPS | 0 | `-2.05967432890634*qdot + 0.368568657330079*tau - 0.213655489970111*sign(qdot) + 0.0293870519978008` | 0.000651 |
| Physics-LS (ablation) | EMPS | 0 | `(-2.0596743290734989)*(qdot) + (0.36856865732351241)*(tau) + (-0.21365548995730729)*(sign(qdot)) + (0.02938705199671356)` | 0.000651 |
| PSE | Roughpipe | 10 | `-10.368*tanh(1.02*x0) + 11.522*tanh(1.179*x0) + 0.492` | 0.004155 |
| PySR | Roughpipe | 4 | `exp(0.30583218 + 0.61426336/(x + 0.31191087/(x - 0.3558122)))` | 0.004070 |
| Operon | Roughpipe | 11 | `1.797691702843 - 1.223697066307*exp(-1.353517770767*x)*cos(2.725285291672*x)` | 0.002476 |
| PSE-aligned VEGA-SR | Roughpipe | 0 | `0.0955593351545504*x**6 - 1.01649818818198*x**5 + 3.98479217714141*x**4 - 6.72502402113137*x**3 + 3.63318430113463*x**2 + 1.51242091672644*x + 0.626029170298395` | 0.001790 |

## 5. Ablation and interpretation

Physics-LS and PSE-aligned VEGA-SR have identical EMPS median test MSE. This shows that the current EMPS gain is attributable primarily to the injected Newton/friction template and coefficient fitting, not uniquely to VEGA-SR's structure-search procedure.

For Roughpipe, VEGA-SR should be interpreted jointly through accuracy and complexity: it improves predictive error over reproduced PSE, while its selected polynomial is structurally larger. Runtime is descriptive rather than a hardware-normalized efficiency claim because PSE and VEGA-SR use GPUs/model inference whereas PySR and Operon use CPUs.

## 6. Paper-ready result paragraph

Under a common validation-score ranking protocol, we evaluated PSE, PSE-aligned VEGA-SR, PySR, and Operon over 20 seeds on the EMPS and Roughpipe real-world tasks. All methods used the same fixed train/validation/test partitions, and test metrics were excluded from the explicit Pareto-front ranking score. The current VEGA-SR implementation did evaluate test-domain predictions when constructing candidate fit records, so we do not claim a fully sealed test set during search. Table 1 reports median and interquartile-range test metrics. On EMPS, the PSE-aligned VEGA-SR configuration substantially reduces predictive error relative to reproduced PSE while adding only one common complexity node. However, a prior-only Physics-LS ablation attains the same solution, indicating that this gain is driven by the injected Newton/friction prior. On Roughpipe, VEGA-SR achieves lower median test error than reproduced PSE, PySR, and Operon, although this improvement is accompanied by a larger symbolic expression. These results support the benefit of physics-aligned candidate construction while also separating that benefit from the contribution of general symbolic search.

## 7. Important reporting caveats

1. The PSE article selected its reported EMPS Pareto expression using the test data. This reproduction introduces a validation split and excludes test metrics from the explicit ranking score, so the values are not intended as an exact reproduction of the paper figure. The current VEGA-SR fit-record implementation still evaluates test-domain predictions during candidate processing; a strictly sealed-test claim would require moving that evaluation after final selection and rerunning the affected experiment.
2. Physics-LS is an ablation, not a general baseline, and is applicable only to EMPS.
3. Runtime comparisons are not hardware-normalized and should not be presented as pure speedups.
4. PySR test-domain failures are retained as failures rather than using test data to choose another Pareto expression; its finite-prediction count is shown in the main table.
5. NGGP, DGSR, BMS, uDSR, TPSR, and wAIC were not added in this first extension because they require substantially heavier pretrained-model/dependency setup, and wAIC is available only for EMPS. PySR and Operon provide two directly released, representative search baselines under the strict split.

## 8. Artifact map and reproduction

- Canonical configuration: [`pse_realworld_vega_sr.yaml`](./pse_realworld_vega_sr.yaml)
- Case runner for PSE/VEGA-SR: [`scripts/run_pse_realworld_case.py`](./scripts/run_pse_realworld_case.py)
- Added baseline runner: [`scripts/run_pse_realworld_baseline_case.py`](./scripts/run_pse_realworld_baseline_case.py)
- Baseline launcher: [`scripts/launch_pse_realworld_baselines.sh`](./scripts/launch_pse_realworld_baselines.sh)
- Combined case-level CSV: [`paper_artifacts/pse_realworld/all_results.csv`](./paper_artifacts/pse_realworld/all_results.csv)
- Aggregated CSV: [`paper_artifacts/pse_realworld/summary.csv`](./paper_artifacts/pse_realworld/summary.csv)
- Original PSE/VEGA report: [`paper_artifacts/pse_realworld/REPORT.md`](./paper_artifacts/pse_realworld/REPORT.md)
- Integrity manifest: [`paper_artifacts/pse_realworld/SHA256SUMS`](./paper_artifacts/pse_realworld/SHA256SUMS)

Regenerate the aggregate CSV and this document with:

```bash
python scripts/summarize_pse_realworld_comparison.py
python scripts/generate_pse_realworld_paper_md.py
```
