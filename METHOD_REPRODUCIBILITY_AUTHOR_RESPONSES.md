# Methods 可复现性信息：作者补充答复

本文件基于当前仓库、实际实验配置、2026-08-13 的运行日志、模型缓存元数据及本机训练产物整理。可确认的信息给出精确值；不能由现有证据确认的内容单列说明，不作推测。

## 结论摘要

| 待补充项 | 当前状态 | 可提供的核心信息 |
|---|---|---|
| Qwen3-VL-32B-Instruct revision、精度和启动参数 | 可补充 | revision `0cfaf48183f594c314753d30a4c4974bc75f3ccb`；bf16；2 张 A800 80 GB；vLLM 0.17.1；TP=2；上下文 32,768 |
| 常数拟合初始化、边界、重启和容差 | 可补充 | SciPy `least_squares`；无边界；TRF；`ftol=xtol=gtol=1e-8`；确定性初始化与分阶段重启策略见下文 |
| NED 实现、编辑代价和规范化 | 可补充 | 仓库内自实现的有序树编辑距离；插入、删除、替换均为单位代价；常数统一为 `Const` |
| EMPS 三个受保护模板 | 可补充 | 三个完整表达式已从实验 YAML 核实，见下文 |
| LoRA、代码和数据固定版本及 DOI | 部分可补充 | 代码与 PSE 实验产物已提交并推送；LoRA adapter 已公开并标记 `v1.0.0`；代码与 adapter 均已确认 Apache-2.0；Zenodo Creator 已确认为 `RUCAIBox` |
| SFT 9,800/200 是否任务分组 | 不能作肯定声明 | 配置只记录 `val_size: 0.02`，没有 group-aware split 或划分清单；只能表述为样本级 9,800/200 划分 |

## 1. Qwen3-VL-32B-Instruct 推理配置

### 可直接写入 Methods 的内容

> The inference backend used Qwen3-VL-32B-Instruct. The local Hugging Face cache metadata identifies the model snapshot as revision `0cfaf48183f594c314753d30a4c4974bc75f3ccb`. The unquantized model was served with vLLM 0.17.1 in bfloat16 precision on two NVIDIA A800 80-GB PCIe GPUs using tensor parallelism of size 2. The server used a maximum model length of 32,768 tokens, a GPU-memory-utilization target of 0.72, a maximum of eight concurrent sequences, and per-prompt multimodal limits of 16 images and zero videos. Remote model code was enabled. The service was bound to `127.0.0.1:18001` and exposed under the name `Qwen3-VL-32B-Instruct`.

需要注意：vLLM 从本地模型目录启动，因此运行日志中的 `revision` 参数为 `None`；上面的 revision 是从该目录的 Hugging Face 缓存元数据恢复出的精确快照标识，而不是启动命令显式传入的参数。

实际服务器参数：

```text
model=<local-or-Hugging-Face-path>/Qwen3-VL-32B-Instruct
served_model_name=Qwen3-VL-32B-Instruct
CUDA_VISIBLE_DEVICES=8,9
tensor_parallel_size=2
pipeline_parallel_size=1
dtype=bfloat16
quantization=None
max_model_len=32768
gpu_memory_utilization=0.72
max_num_seqs=8
limit_mm_per_prompt={"image":16,"video":0}
trust_remote_code=true
engine_seed=0
```

每次 LLM 请求还使用了确定性的请求级种子。若调用方没有显式提供种子，代码将以下字段序列化后计算 SHA-256，并将前 8 个十六进制字符转换为 31 位非负整数：数据集名、实验重复种子、Agent 角色、该角色的调用序号、temperature、max tokens 和 top-p。由于不同 Agent/阶段会传入不同采样参数，不应把所有请求笼统写成固定 `temperature=0.2, top_p=0.95`。

### 运行环境

2026-08-23 对实际启动脚本所引用的本地 `sr` 环境复核得到：

| 项目 | 值 |
|---|---|
| Python | 3.10.20 |
| NumPy | 2.2.6 |
| pandas | 2.3.3 |
| SciPy | 1.15.3 |
| SymPy | 1.14.0 |
| scikit-learn | 1.7.2 |
| PyTorch | 2.10.0 |
| Transformers | 4.57.6 |
| vLLM | 0.17.1 |
| OS | Ubuntu 24.04.3 LTS，kernel 6.8.0-86-generic |
| CPU | 2 × Intel Xeon Platinum 8160；每颗 24 核、2 线程/核，共 96 个逻辑 CPU |
| GPU | GPU 8、9：NVIDIA A800 80GB PCIe，各 81,920 MiB |
| NVIDIA driver | 570.195.03 |

2026-08-13 的服务日志明确记录了 vLLM 0.17.1、bf16、无量化、TP=2 和上述启动参数。其余 Python 包版本是审计时从同一环境读取的当前值；若作者需要严格证明它们与运行当日完全一致，还应归档该环境的 `pip freeze` 或 Conda lock 文件。

证据位置：`scripts/launch_pse_realworld_vega_service.sh`、`results/pse_realworld_validation/logs/vega_service/vllm_qwen3vl32b.log` 和模型目录下的 `.cache/huggingface/download/*.metadata`。

## 2. 常数拟合、无效表达式判定和符号化简

### 可直接写入 Methods 的内容

> Free constants were identified as SymPy free symbols that were not observed feature names and were ordered lexicographically. Constants were fitted on the training partition only by minimizing the residual vector `prediction - y_train` with `scipy.optimize.least_squares` (SciPy 1.15.3). The implementation supplied only `x0` and `max_nfev`; consequently, it used SciPy's trust-region reflective method with numerical two-point Jacobians, unbounded parameters, linear loss, and `ftol=xtol=gtol=1e-8`. The default maximum number of function evaluations was 1,200 and was reduced to 600 for expressions of at least 120 characters or at least four fitted parameters. The fit with the smallest finite least-squares cost across the configured restarts was retained.

边界与求解器参数如下：

```text
method="trf"
jac="2-point"
bounds=(-inf, inf)
loss="linear"
ftol=1e-8
xtol=1e-8
gtol=1e-8
x_scale=1.0
max_nfev=1200  # 普通表达式
max_nfev=600   # 表达式长度 >=120 或参数数 >=4
```

### 初始化和重启

常数拟合使用 `numpy.random.default_rng(42)`。确定性基础初值按参数名生成：以 `a/scale/gain/amp/alpha` 或 `p/q/pow/exp` 开头的参数初始化为 1，其余初始化为 0。候选初值按以下顺序去重后使用：

1. 基础初值；
2. 全零；
3. `0.5 ×` 基础初值；
4. `2 ×` 基础初值；
5. `-1 ×` 基础初值；
6. 若仍未达到重启次数，再从零均值正态分布采样，标准差依次循环为 `0.25、1、2、4 × init_scale`。

搜索阶段的默认重启策略为：预筛选 1 次；快速种子、初始候选、多模态候选和 refinement 候选各 2 次；其他阶段 3 次。快速阶段的 `init_scale=0.5`，其余默认 `init_scale=0.8`。候选数达到 20 时重启最多 2 次，达到 32 时只重启 1 次。低维加性非线性专用路径例外地使用 8 次重启。PSE real-world 数据在代码中标为 `source_tag="pse_realworld"`，不会触发仅面向 benchmark source 的强制单重启规则。

### 无效表达式与化简

- SymPy 解析失败、拟合异常或所有重启均没有产生有限参数时，候选记为失败。
- 优化过程中若训练预测含 NaN 或 Inf，残差向量全部替换为 `1e6`。
- 最优参数代回 SymPy 表达式后，分别计算 train/validation/test MSE。
- 基础化简使用 `sympy.simplify`。表达式字符串超过 240 个字符或包含极大数字模式时跳过该步。
- V11 化简器把绝对值不超过 `1e-10` 的浮点数原子置零；对节点数不超过 80 的表达式，用 `sympy.nsimplify(..., [pi, E], tolerance=1e-8, full=False)` 做保守有理化，仅保留字符串长度足够紧凑的替换，之后再次调用 `sympy.simplify`。

### 论文表述必须保留的限制

优化目标本身只使用训练集，最终候选重排设置也明确为 `use_test_for_selection: false`。但是当前 `TemplateFillTool` 会在候选拟合阶段计算并保存 test MSE，而且 test 预测异常可使候选拟合失败。因此，现有代码证据支持“test score was not used as the numerical fitting objective or ranking metric”，但不支持更强的“the test set was never accessed during search”。如果论文要作后一个严格声明，需要先把候选拟合阶段的 test 评估移到最终选择之后并重新运行受影响的实验。

证据位置：`tools/template_fill_tool.py`、`tools/algebraic_simplify_tool.py`、`scripts/vl_loopsr_core.py` 和 `scripts/vega_sr.py`。

## 3. NED 的精确实现

### 可直接写入 Methods 的内容

> Normalized equation-tree edit distance (NED) was computed by a self-contained ordered tree-edit-distance implementation in our repository, following the normalization and metric convention of the public SRSD benchmark. No external tree-edit-distance package was called at runtime. The metric was `min(TED(T_pred,T_true), |T_true|) / |T_true|`, where `|T_true|` is the number of nodes in the normalized ground-truth tree. Insertions, deletions, and node relabellings each had unit cost. All numeric nodes shared the label `Const`, variable nodes retained their exact symbol names, and other nodes were labelled with the corresponding SymPy function/operator name. Children followed the order in `SymPy.Expr.args`. Structural similarity was reported as `1-NED`.

规范化顺序为：

1. 将 `^` 改写为 `**`；
2. 统一函数别名，包括 `abs/Abs`、`ln/log`、`sign/sgn`、`asin/arcsin`、`acos/arccos` 和 `atan/arctan`；
3. 使用 `sympy.sympify` 解析；
4. 重新从字符串解析一次；
5. 将 `pi` 替换为其浮点值；
6. 依次调用 `evalf()`、`factor()`、`simplify()`，并把 `1.0` 替换为整数 `1`；
7. 再次从字符串解析，随后构造有序表达式树。

真实表达式无法解析时状态为 `invalid_ground_truth`；预测表达式缺失或无法解析时状态为 `unavailable_prediction`，NED 记为缺失值。候选池最小 NED 是使用真式计算的 oracle 诊断量，只能用于分析候选池覆盖率，不能作为最终方法性能。

实现文件：`scripts/compute_multimodal_ned.py`。

## 4. EMPS 的三个受保护模板

配置中的物理先验来源为牛顿第二定律及黏性/库仑摩擦：

```text
M*qddot = -Fv*qdot - Fc*sign(qdot) + tau - c
```

三个受保护候选模板为：

```text
a*qdot + b*tau + c*sign(qdot) + d
a*q + b*qdot + c*tau + d*sign(qdot) + e
a*qdot + b*tau + c*sign(qdot) + d*abs(qdot) + e
```

它们在 Proposer 初始候选之前插入并去重，随后与其他候选经过相同的拟合和验证集选择流程。允许算子为 `+、-、*、/、sin、cos、exp、log、cosh、tanh、abs、sign`，选择指标为 `validation_reward_eta_0.99`，其中 `eta=0.99`。

配置位置：`pse_realworld_vega_sr.yaml`。

## 5. LoRA adapter、代码和数据版本

### 已公开的 adapter 与固定标识

LoRA adapter 公开地址：

```text
https://huggingface.co/liuyihong/qwen3-vl-32b-proposer-sr-lora
tag=v1.0.0
commit=21648fa93c594bb0f2ecc43c345f96b53f3ea3bb
```

本地 LoRA 产物：

```text
artifact=qwen3-vl-32b-proposer-sft-10000-qlora
adapter_model.safetensors size=537005424 bytes
adapter_model.safetensors sha256=0a8b2c55edd3c9900a94aa55a0170979c748fd11b6e5fa88531c4fe88f693dbb
adapter_config.json sha256=f9b73ced57f0553ff36de4e1fc9be4b1f0ee3660422a745a8425e0a18fbbd5a5
```

adapter 配置记录了 LoRA rank 16、alpha 32、dropout 0.05，目标模块为 `q_proj、k_proj、v_proj、o_proj、gate_proj、up_proj、down_proj`。顶层产物对应训练完成的最终 step 1,226；`best_model_checkpoint` 和 `best_metric` 均为 `null`，因此不能写成“按最低验证损失选择最佳 checkpoint”。

已推送的代码仓库及 PSE 实验快照提交：

```text
https://github.com/RUCAIBox/VEGA-SR
d5d15047946717c585bc8baa0886efb517cf9df5
```

PSE 固定提交：

```text
7105caba63e754150dd3b160443984456ec99cd7
```

原始实测数据哈希：

```text
EMPS sha256=9c77776a8cdc56354e72ea6ed8e5cf2d94e1c72910d585d2d31742113695b2ee
rough-pipe sha256=f5ea004e268dd182285ec782805a4af69373417bf18183e2df65fdb3fbe352b1
```

SFT 语料归档哈希：

```text
proposer_mllm_sft.json.gz sha256=1c7383b4078540bba7e0e282647356ba6af88e18160a84cdaca8e250c5087826
images.tar sha256=d1e709bd2cfa04fd526818d22748304e11d534e6148156c2f96ca53d8791e9c1
csv.tar sha256=24aa77353b036c6962de7d339916f07c7b675ed230b182ccaa4b8d8a4385295c
```

### 目前不能声称已经完成的事项

1. 尚未建立论文对应的 GitHub release 和 Zenodo DOI。
2. SFT adapter 的 `adapter_config.json` 中 `revision=null`。当前本地基础模型目录可恢复出 revision `0cfaf...`，但没有证据证明 SFT 训练开始时使用的基础模型快照也已被显式固定到该 revision。

代码与 adapter 均已由作者确认为 Apache-2.0，Zenodo Creator 已确认为机构 `RUCAIBox`。在论文中填写 DOI 前，只剩在 Zenodo 中开启 GitHub 仓库归档，再创建 GitHub release，最后把 DOI 和归档提交号回填论文。

### 许可证核查与最小作者输入

截至 2026-08-23，已核对的许可状态如下：

| 对象 | 当前许可状态 | 对本项目的含义 |
|---|---|---|
| VEGA-SR 代码仓库 | Apache-2.0 | 根目录 `LICENSE` 已收录完整 Apache License 2.0 文本；第三方依赖、benchmark 数据和上游资产仍保留各自条款。 |
| Qwen3-VL-32B-Instruct | Apache-2.0 | 这是 LoRA 基础模型的上游许可，但不会自动替 VEGA-SR 原创代码或 SFT 语料选定许可证。 |
| PSE | MIT | 当前实验固定并调用上游 PSE；如将 PSE 代码实质复制进发行物，必须保留其 MIT 版权和许可通知。 |
| 已公开 LoRA adapter | Apache-2.0 | 模型卡元数据和许可说明已更新，并与基础模型许可保持一致。 |

Zenodo 对公开记录的 `License` 和 `Creators` 都规定为必填字段。本次已根据作者确认将两者分别固定为 `Apache-2.0` 和 `RUCAIBox`，并写入根目录 `.zenodo.json`。Zenodo 归档将以软件为主资源；第三方依赖、benchmark 数据和其他上游资产仍保留各自条款，不因根仓库许可证而被重新授权。

作者信息的最小要求为：

- **必须提供**：至少一个 Creator，可以是个人姓名或机构名称；如有多个 Creator，还需给出引文中的顺序。Creator 会出现在 Zenodo 自动生成的学术引文中。
- **不是必填**：ORCID、单位/ROR、邮箱、通讯作者标记。本次最小发布不再要求这些信息。
- `CITATION.cff` 不是 Zenodo DOI 的必备文件，可以不创建；但无论是通过 `.zenodo.json`、GitHub 集成还是 Zenodo 页面填写，Zenodo 记录本身仍必须有 Creator。

官方依据：[Zenodo Creators](https://help.zenodo.org/docs/deposit/describe-records/creators/)、[Zenodo Licenses and rights](https://help.zenodo.org/docs/deposit/describe-records/licenses/)、[GitHub Licensing a repository](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository)、[Qwen3-VL-32B-Instruct model card](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct)。

本次作者已确认：

```text
license: Apache-2.0（VEGA-SR 代码和 LoRA adapter）
creators: RUCAIBox
```

其余元数据已从仓库、发布说明和已固定的版本信息生成，不再要求作者手工提供。

在此之前，建议只保留如下占位表述，不要填写虚假 DOI：

> The code, experiment configurations, integrity-checked result artifacts, SFT corpus, and versioned LoRA adapter are publicly available from the VEGA-SR GitHub and Hugging Face repositories. A DOI-backed archival code snapshot will be added after the GitHub release is deposited through Zenodo. [DOI to be added after archival release.]

## 6. SFT 9,800/200 划分的正确表述

语料包含 10,000 个样本，来自 5,000 个任务；每个任务恰好对应一个 proposal 样本和一个 Critic-conditioned repair 样本。SFT 配置设置 `val_size: 0.02`，训练日志确认最终为 9,800 个训练样本和 200 个验证样本。

但现有配置没有 task ID 分组字段，产物中也没有保存 train/eval ID 清单。因此不能证明两个配对样本始终落在同一侧，也不能把该划分描述为 task-disjoint 或 group-aware。

建议正文采用：

> The SFT corpus contained 10,000 example records derived from 5,000 paired tasks, with one proposal record and one Critic-conditioned repair record per task. A 2% example-level evaluation split produced 9,800 training records and 200 evaluation records. Because a task-grouped split manifest was not retained, we do not claim that the two partitions were task-disjoint. We therefore treat the archived SFT comparison as descriptive rather than as a causal checkpoint comparison.

如果后续希望作严格 SFT 效果声明，应按 5,000 个基础 task ID 分组后重新划分和训练，并保存两侧 task/example ID 清单及其 SHA-256。

## 建议回传给论文修改方的最简答复

前四项现在均已有可核实的精确答案，可据本文件直接补入 Methods/Supplement。第五项的代码、PSE 结果与 LoRA adapter 已公开；代码与 adapter 均已确认为 Apache-2.0，Zenodo Creator 已确认为 `RUCAIBox`，并已写入发布元数据。现在只剩通过 Zenodo 归档 GitHub release 生成 DOI。SFT 的 9,800/200 划分只能称为 2% 样本级划分，不能称为按任务分组；建议保留描述性分析限定。
