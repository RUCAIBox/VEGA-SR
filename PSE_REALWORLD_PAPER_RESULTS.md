# PSE real-world experiment: paper-ready configuration and results

Generated: 2026-09-09 08:45:08 CST

Status: 全部正式实验已完成。

## 1. Experiment scope

This experiment compares PSE-aligned VEGA-SR with PSE, PySR, Operon, gplearn, DSO, and LLM-SR on two real-world symbolic-regression tasks from the PSE study. DSO and LLM-SR are carried over from the paper's 895-task comparator set and rerun here on the measured systems; no result is transferred across datasets. Physics-LS is reported separately as an EMPS prior-only ablation rather than as a general symbolic-regression baseline. The goal is to compare predictive accuracy, symbolic complexity, and descriptive runtime under a common data split and a validation-only expression-selection protocol.

## 2. Reproducible protocol

- Repeats: 20 runs (`0`–`19`) for every search method. Linear-LS and Physics-LS are deterministic and are run once per applicable dataset.
- Search budget: the nominal search deadline is 90 seconds per formal case, with method-specific enforcement granularity. PSE, PySR, Operon, gplearn and LLM-SR stop at their next safe stage, iteration, generation or request boundary; DSO retains the paper configuration's fixed 5,000-sample work budget with a best-effort 90-second alarm. Actual runtime is always reported, and one-time VEGA-SR and LLM-SR model-service startup is excluded.
- Selection: methods exposing candidate sets are ranked using validation data with $0.99^C/(1+\sqrt{\mathrm{MSE}_{val}})$, where $C$ is the common SymPy-tree complexity. DSO retains its native risk-seeking objective and stopping rule, consistent with the baseline protocol used elsewhere in the paper. Test metrics are not part of any ranking score. The current VEGA-SR fitter nevertheless evaluates candidate predictions on the test inputs while constructing fit records and may reject a numerically invalid expression, so this implementation does not satisfy the stronger claim that the test split remains completely unaccessed during search.
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
- **gplearn:** genetic programming with population size 1,000, tournament size 20, parsimony coefficient 0.001, one CPU thread, and a 90-second target checked between generations. Because a generation is atomic, one Roughpipe seed required 163.75 seconds of search time; no test-based fallback or reselection was performed.
- **DSO:** the official deep-symbolic-optimization implementation and the configuration used in the paper's 895-task comparison; protected operators, risk-seeking policy-gradient search, one CPU thread and 5,000 sampled expressions. A best-effort 90-second alarm bounds the search, but TensorFlow's atomic calls can delay signal delivery; the maximum observed fit runtime was 130.51 seconds. DSO retains its native final-program selection.
- **LLM-SR:** the official LLM-SR evolutionary pipeline used in the paper's 895-task comparison, driven by Qwen2.5-32B-Instruct. Candidate constants are fitted on training data and final candidates are ranked on validation data under the common score. Generation stops at the first API request boundary after 90 seconds; post-search candidate refitting and summary construction account for total runtimes above 90 seconds.
- **Physics-LS:** the same three EMPS physical templates supplied to VEGA-SR, with coefficients fitted by ordinary least squares. It performs no symbolic structure search.

For the PSE-aligned operator-library comparisons, EMPS operators are `+`, `-`, `*`, `/`, `sin`, `cos`, `exp`, `log`, `cosh`, `tanh`, `abs`, and `sign`; Roughpipe uses `+`, `-`, `*`, `/`, `sin`, `cos`, `exp`, `log`, `cosh`, `tanh`, square, and cube. DSO and LLM-SR retain their own released operator/program spaces.

## 3. Main results

Values are median [Q1, Q3] over 20 runs, except deterministic Linear-LS and Physics-LS. Lower MSE/NMSE/complexity is better; higher $R^2$ is better.
For a validation-selected expression that is numerically invalid on held-out inputs, test metrics remain missing rather than selecting a fallback using test data; the `n` column reports the number of finite test predictions when this occurs.

| Method | Dataset | n | Test MSE | Test NMSE | Test R² | Complexity | Runtime (s) |
|---|---|---|---|---|---|---|---|
| Linear-LS | EMPS | 1 | 0.009689 | 0.054106 | 0.945894 | 11.0 | 0.04 |
| PSE | EMPS | 20 | 0.009673 [0.009673, 0.009673] | 0.054017 [0.054017, 0.054017] | 0.945983 [0.945983, 0.945983] | 11.0 [11.0, 11.0] | 100.29 [97.38, 105.86] |
| PySR | EMPS | 20 (19 valid) | 0.010609 [0.010101, 0.050061] | 0.059245 [0.056409, 0.279562] | 0.940755 [0.720438, 0.943591] | 8.0 [7.0, 8.0] | 142.66 [141.68, 145.75] |
| Operon | EMPS | 20 | 0.010101 [0.010101, 0.010101] | 0.056410 [0.056410, 0.056410] | 0.943590 [0.943590, 0.943590] | 8.0 [8.0, 8.0] | 90.31 [90.28, 90.52] |
| gplearn | EMPS | 20 | 0.012242 [0.009968, 0.050308] | 0.068365 [0.055664, 0.280945] | 0.931635 [0.719055, 0.944336] | 7.0 [6.0, 8.0] | 89.57 [89.31, 89.73] |
| DSO | EMPS | 20 | 0.091602 [0.085646, 0.098162] | 0.511549 [0.478286, 0.548184] | 0.488451 [0.451816, 0.521714] | 19.0 [13.5, 24.2] | 48.53 [47.81, 49.71] |
| LLM-SR | EMPS | 20 | 0.009689 [0.009689, 0.009689] | 0.054106 [0.054106, 0.054106] | 0.945894 [0.945894, 0.945894] | 11.0 [11.0, 11.0] | 92.66 [92.05, 93.15] |
| PSE-aligned VEGA-SR | EMPS | 20 | 0.000651 [0.000651, 0.000651] | 0.003637 [0.003637, 0.003637] | 0.996363 [0.996363, 0.996363] | 12.0 [12.0, 12.0] | 21.14 [20.91, 21.81] |
| Physics-LS (ablation) | EMPS | 1 | 0.000651 | 0.003637 | 0.996363 | 12.0 | 0.00 |
| Linear-LS | Roughpipe | 1 | 0.054418 | 1.012713 | -0.012713 | 5.0 | 0.03 |
| PSE | Roughpipe | 20 | 0.003579 [0.002148, 0.009005] | 0.066650 [0.040003, 0.167709] | 0.933350 [0.832291, 0.959997] | 14.0 [11.0, 16.0] | 94.71 [92.92, 96.38] |
| PySR | Roughpipe | 20 | 0.004175 [0.002890, 0.007001] | 0.077698 [0.053783, 0.130279] | 0.922302 [0.869721, 0.946217] | 11.0 [10.0, 13.0] | 143.39 [142.44, 145.42] |
| Operon | Roughpipe | 20 | 0.002742 [0.002201, 0.004841] | 0.051071 [0.041000, 0.090151] | 0.948929 [0.909849, 0.959000] | 12.0 [11.8, 12.2] | 90.30 [90.21, 90.40] |
| gplearn | Roughpipe | 20 | 0.014994 [0.014800, 0.015920] | 0.279247 [0.275641, 0.296506] | 0.720753 [0.703494, 0.724359] | 7.0 [6.0, 7.5] | 89.55 [89.34, 89.71] |
| DSO | Roughpipe | 20 | 0.044521 [0.035174, 0.061640] | 0.829168 [0.655085, 1.147994] | 0.170832 [-0.147994, 0.344915] | 12.5 [7.2, 16.0] | 38.71 [38.16, 39.51] |
| LLM-SR | Roughpipe | 20 | 0.012117 [0.012117, 0.023497] | 0.225503 [0.225503, 0.437268] | 0.774497 [0.562732, 0.774497] | 11.0 [11.0, 11.0] | 100.07 [98.21, 103.41] |
| PSE-aligned VEGA-SR | Roughpipe | 20 | 0.001790 [0.001790, 0.001790] | 0.033317 [0.033317, 0.033317] | 0.966683 [0.966683, 0.966683] | 30.0 [30.0, 30.0] | 20.84 [20.51, 21.74] |

### Relative to reproduced PSE

Positive MSE reduction and positive $R^2$ difference favor the compared method. Complexity difference is compared with PSE.

| Method | Dataset | Median MSE reduction | Median R² difference | Complexity difference |
|---|---|---|---|---|
| PySR | EMPS | -9.68% | -0.00523 | -3.0 |
| Operon | EMPS | -4.43% | -0.00239 | -3.0 |
| gplearn | EMPS | -26.56% | -0.01435 | -4.0 |
| DSO | EMPS | -847.02% | -0.45753 | +8.0 |
| LLM-SR | EMPS | -0.17% | -0.00009 | +0.0 |
| PSE-aligned VEGA-SR | EMPS | +93.27% | +0.05038 | +1.0 |
| PySR | Roughpipe | -16.67% | -0.01105 | -3.0 |
| Operon | Roughpipe | +23.38% | +0.01558 | -2.0 |
| gplearn | Roughpipe | -318.97% | -0.21260 | -7.0 |
| DSO | Roughpipe | -1144.06% | -0.76252 | -1.5 |
| LLM-SR | Roughpipe | -238.60% | -0.15885 | -3.0 |
| PSE-aligned VEGA-SR | Roughpipe | +49.97% | +0.03333 | +16.0 |

## 4. Representative median-run expressions

The representative run is the seed whose test MSE is closest to the method's median; it is for readability and is not an additional selection step.

| Method | Dataset | Seed | Expression | Test MSE |
|---|---|---|---|---|
| Linear-LS | EMPS | 0 | `(0.11783025529126621)*(q) + (-4.0217493684107604)*(qdot) + (0.3544932869764566)*(tau) + (0.01781131374518197)` | 0.009689 |
| PSE | EMPS | 0 | `0.118*x0 - 4.022*x1 + 0.354*x2 + 0.018` | 0.009673 |
| PySR | EMPS | 8 | `(-qdot + tau*0.08905808)*3.8902402` | 0.010609 |
| Operon | EMPS | 1 | `-4.02557898177729*qdot + 0.353528881913982*tau + 0.038255915046` | 0.010101 |
| gplearn | EMPS | 13 | `-3.423*qdot + 0.311817898347365*tau` | 0.012307 |
| DSO | EMPS | 6 | `q*(-qdot + tau)*log(q*(-tau + exp(tau))*cos(q + tau) - tau)` | 0.092666 |
| LLM-SR | EMPS | 0 | `0.11783255401659069*q - 4.0217512865461401*qdot + 0.35449316962378039*tau + 0.017810605851169109` | 0.009689 |
| PSE-aligned VEGA-SR | EMPS | 0 | `-2.05967432890634*qdot + 0.368568657330079*tau - 0.213655489970111*sign(qdot) + 0.0293870519978008` | 0.000651 |
| Physics-LS (ablation) | EMPS | 0 | `(-2.0596743290734989)*(qdot) + (0.36856865732351241)*(tau) + (-0.21365548995730729)*(sign(qdot)) + (0.02938705199671356)` | 0.000651 |
| Linear-LS | Roughpipe | 0 | `(0.038921066266160528)*(x) + (1.7282217249593421)` | 0.054418 |
| PSE | Roughpipe | 10 | `-10.368*tanh(1.02*x0) + 11.522*tanh(1.179*x0) + 0.492` | 0.004155 |
| PySR | Roughpipe | 4 | `exp(0.30583218 + 0.61426336/(x + 0.31191087/(x - 0.3558122)))` | 0.004070 |
| Operon | Roughpipe | 11 | `1.797691702843 - 1.223697066307*exp(-1.353517770767*x)*cos(2.725285291672*x)` | 0.002476 |
| gplearn | Roughpipe | 19 | `exp(tanh(sin(cos(log(Abs(x))))))` | 0.014994 |
| DSO | Roughpipe | 7 | `exp(cos(exp(exp(2*x + exp(x) - exp(exp(x)) + 1))))` | 0.044359 |
| LLM-SR | Roughpipe | 0 | `-0.2406929906256654*x + 2.336029639662128 - 1.7735940642856349*exp(-2.5324591502435085*x)` | 0.012117 |
| PSE-aligned VEGA-SR | Roughpipe | 0 | `0.0955593351545504*x**6 - 1.01649818818198*x**5 + 3.98479217714141*x**4 - 6.72502402113137*x**3 + 3.63318430113463*x**2 + 1.51242091672644*x + 0.626029170298395` | 0.001790 |

## 5. Ablation and interpretation

Physics-LS and PSE-aligned VEGA-SR have identical EMPS median test MSE. This shows that the current EMPS gain is attributable primarily to the injected Newton/friction template and coefficient fitting, not uniquely to VEGA-SR's structure-search procedure.

On EMPS, LLM-SR converges to the same linear family as Linear-LS and closely matches PSE, whereas DSO and gplearn are weaker. On Roughpipe, LLM-SR improves over gplearn and DSO but remains below PSE, PySR and Operon. VEGA-SR achieves the lowest median error on both datasets, although its Roughpipe polynomial is structurally larger. Runtime is descriptive rather than a hardware-normalized efficiency claim because methods use different CPU/GPU hardware and stopping granularities.

## 6. Paper-ready result paragraph

We evaluated PSE-aligned VEGA-SR, PSE, PySR, Operon, gplearn, DSO and LLM-SR over 20 runs on the EMPS and Roughpipe measured-system tasks. Every method used the same fixed train/validation/test partitions, and test metrics were excluded from expression ranking; DSO retained its released native objective and stopping rule. On EMPS, LLM-SR recovered the linear baseline family and closely matched PSE, while VEGA-SR achieved the lowest median test MSE. The identical Physics-LS result shows that this EMPS improvement is explained primarily by the injected Newton/friction prior. On Roughpipe, VEGA-SR also achieved the lowest median test MSE, ahead of Operon, PSE and PySR, but selected a more complex polynomial. DSO, gplearn and LLM-SR were less accurate on Roughpipe under this protocol.

## 7. Important reporting caveats

1. The PSE article selected its reported EMPS Pareto expression using the test data. This reproduction introduces a validation split and excludes test metrics from the explicit ranking score, so the values are not intended as an exact reproduction of the paper figure. The current VEGA-SR fit-record implementation still evaluates test-domain predictions during candidate processing; a strictly sealed-test claim would require moving that evaluation after final selection and rerunning the affected experiment.
2. Physics-LS is an ablation, not a general baseline, and is applicable only to EMPS.
3. Runtime comparisons are not hardware-normalized and should not be presented as pure speedups.
4. PySR test-domain failures are retained as failures rather than using test data to choose another Pareto expression; its finite-prediction count is shown in the main table.
5. DSO and LLM-SR reuse methods from the paper, but their EMPS/Roughpipe values are newly measured under this protocol; values from the 895-task comparison are not transplanted.
6. The 90-second budget is a target with method-specific safe stopping boundaries, not a claim that every recorded process wall time is at most 90 seconds. Exact observed runtimes are reported in the table and case JSON files.

## 8. Artifact map and reproduction

- Canonical configuration: [`pse_realworld_vega_sr.yaml`](./pse_realworld_vega_sr.yaml)
- Case runner for PSE/VEGA-SR: [`scripts/run_pse_realworld_case.py`](./scripts/run_pse_realworld_case.py)
- Added baseline runner: [`scripts/run_pse_realworld_baseline_case.py`](./scripts/run_pse_realworld_baseline_case.py)
- Baseline launcher: [`scripts/launch_pse_realworld_baselines.sh`](./scripts/launch_pse_realworld_baselines.sh)
- Combined case-level CSV: [`paper_artifacts/pse_realworld/all_results.csv`](./paper_artifacts/pse_realworld/all_results.csv)
- Aggregated CSV: [`paper_artifacts/pse_realworld/summary.csv`](./paper_artifacts/pse_realworld/summary.csv)
- Original PSE/VEGA report: [`paper_artifacts/pse_realworld/REPORT.md`](./paper_artifacts/pse_realworld/REPORT.md)

Regenerate the aggregate CSV and this document with:

```bash
python scripts/summarize_pse_realworld_comparison.py
python scripts/generate_pse_realworld_paper_md.py
```
