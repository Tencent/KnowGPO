<h1 align="center">
  KnowGPO: Knowledge-Grouped Policy Optimization for Search Agents
</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2610.22331"><img alt="Paper" src="https://img.shields.io/badge/arXiv-2610.22331-b31b1b"></a>
  <a href="LICENSE"><img alt="License: Apache 2.0" src="https://img.shields.io/badge/License-Apache%202.0-blue.svg"></a>
</p>

<p align="center">
  Siqi Chen<sup>1,2</sup>, Zhengxi Lu<sup>1</sup>, Yuchen Yan<sup>1</sup>, Xingyu Wu<sup>1,2</sup>,
  Qipeng Chen<sup>1</sup>, Xin Zhang<sup>2</sup>, Aiting Liu<sup>2</sup><br>
  Chao Deng<sup>2</sup>, Jie Liu<sup>2</sup>, Jin Ma<sup>2</sup>, Weiming Lu<sup>1</sup>, Jun Xiao<sup>1</sup>, Yongliang Shen<sup>1&dagger;</sup>
</p>

<p align="center">
  <sup>1</sup> Zhejiang University &nbsp;&nbsp;&middot;&nbsp;&nbsp;
  <sup>2</sup> Tencent &nbsp;&nbsp;&middot;&nbsp;&nbsp;
  <sup>&dagger;</sup> Corresponding author
</p>

## 🔥 Overview

**KnowGPO** (Knowledge-Grouped Policy Optimization) is a reinforcement-learning method for long-horizon deep-search agents. Instead of matching surface observations across search trajectories, it groups steps by the task-relevant knowledge acquired so far and assigns process credit for newly verified knowledge conditions.

<p align="center">
  <img src="assets/intro.png" alt="Outcome-only, observation-grouped, and knowledge-grouped credit assignment" width="95%">
</p>

**Release scope.** The method implementation was historically labeled V7 RGC-IR in the experiment code; the public method name is KnowGPO. This repository contains the **KnowGPO training code path**, adapted from `verl`: multi-turn search rollouts, outcome/knowledge verification, knowledge-grouped step advantages, their fusion with GRPO outcome advantages, and a single KnowGPO launcher. It does **not** contain training or evaluation datasets, knowledge-condition generation scripts, SFT checkpoints, trained model weights, hosted retrieval/judge services, or supported baseline-training launchers. The inherited `verl/` tree retains general-purpose utility and baseline code; the legacy parent reward managers are retained because KnowGPO subclasses and reuses them.

## 🎉 News

- **October 15, 2026:** [KnowGPO: Knowledge-Grouped Policy Optimization for Search Agents](https://arxiv.org/abs/2610.22331) is available on arXiv.

## 🌟 Highlights

- **Knowledge-state alignment.** The paper reports 97.8% effective step alignment with knowledge-state grouping versus 51.8% using observation-based grouping.
- **Environment-grounded process credit.** A step is rewarded for increasing the number of independently verified knowledge conditions, not for boosting the model's own answer confidence.
- **Deep-search benchmarks.** The paper reports 48.0% on Xbench-2505, 38.8% on BrowseComp-ZH, and 40.2% on GAIA using a Qwen3-8B agent. **These are results reported in the paper, not scores independently reproduced with this repository snapshot.**

## 🤖 Method

<p align="center">
  <img src="assets/method.png" alt="KnowGPO knowledge-condition construction and grouped training pipeline" width="95%">
</p>

1. **Knowledge conditions:** A two-stage process defines a query-specific reasoning skeleton with entity placeholders, then grounds it using the correct answer and a successful reference trajectory. **This release expects the conditions to have been prepared externally**; it does not implement the two-stage data-generation pipeline.
2. **Knowledge states:** An LLM verifier checks which conditions are supported at each search step. Satisfied conditions accumulate monotonically, so different search paths can be compared from the same prior knowledge state.
3. **Knowledge-grouped credit:** Non-final steps for the same question and prior knowledge state are grouped. Their incremental knowledge gain is normalized within that group and combined with an episode-level GRPO outcome advantage. The final-answer step has zero process advantage.

**Training chain:** `SFT model + prompts → SGLang multi-turn search rollout → final-answer grader + knowledge-state verifier → group-by-(query UID, pre-action knowledge state) step advantage → GRPO outcome advantage → convex_final_exempt token fusion → PPO actor update`.

The code preserves historical `gig_*` batch-field names to match the trainer contract. On non-final tokens, KnowGPO combines normalized outcome and step advantages; final-answer tokens retain the **raw, unattenuated outcome advantage**. This reflects the released code and should not be mistaken for a verbatim implementation of every simplified equation in the paper. The KnowGPO configuration disables length penalties.

## 📊 Performance

<p align="center">
  <img src="assets/experiments.png" alt="Benchmark results reported in the KnowGPO paper" width="95%">
</p>

The paper compares KnowGPO against outcome-only GRPO and process-supervision baselines on Xbench, BrowseComp-ZH, and GAIA. It reports **38.8% on BrowseComp-ZH** (1.3 percentage points below DeepMiner-32B-RL) and **40.2% on GAIA**; the published benchmark tables and evaluation protocol, rather than the dry-run checks below, are the source of these figures. Results from this snapshot have **not** been re-evaluated. Extra print-resolution source figures are available in [`assets/`](assets/).

## 🔖 Repository Layout

```text
KnowGPO/
├── assets/                             # Paper graphics and print-resolution sources
├── LICENSE                             # Apache License 2.0 from the original release
├── README.md
├── pyproject.toml                     # Python package and runtime dependencies
├── scripts/train_knowgpo.sh           # KnowGPO training launcher
├── verl/trainer/config/               # PPO/Hydra and KnowGPO reward settings
├── verl/workers/reward_manager/       # KnowGPO manager and inherited dependencies
├── verl/utils/reward_score/           # Knowledge-state verification and step rewards
├── verl/trainer/ppo/ray_trainer.py     # GRPO advantage and KnowGPO fusion
├── verl/tools/                       # Search-tool interface (external backend required)
└── tests/knowgpo/                     # Dependency-light offline contract tests
```

## 🛠️ Setup and Inputs

Use a Linux CUDA environment compatible with `verl`, PyTorch, Ray, and SGLang. A possible installation starting point is `pip install -e '.[sglang]'`. Verify PyTorch/CUDA and SGLang versions for your host; dependency installation and GPU training have **not** been validated for this snapshot.

Supply all of the following **outside the repository**:

| Setting | Required input |
| --- | --- |
| `KNOWGPO_MODEL_PATH` | Compatible SFT checkpoint capable of ReAct-style `<think>`, `<tool_call>`, and `<answer>` output. |
| `KNOWGPO_TRAIN_FILES`, `KNOWGPO_VAL_FILES` | Paths to train/validation parquet datasets. The code still loads validation data when `trainer.val_before_train=False`. |
| `KNOWGPO_SEARCH_URL` | Search service endpoint accepting HTTP POST JSON `{"queries": ["..."], "topk": 10, "return_scores": true}` and returning a `result` array in the format expected by `verl/tools/utils/search_r1_like_utils.py`. |
| `KNOWGPO_JUDGE_BASE_URL`, `KNOWGPO_JUDGE_API_KEY`, `KNOWGPO_JUDGE_MODEL` | OpenAI-compatible step-state verifier and final-answer grader. Optional `KNOWGPO_JUDGE_PROXY`. |

Each dataset row must provide the `verl` prompt and data-source fields, `reward_model.ground_truth.target`, and `extra_info.question`, `extra_info.constraints_gt` (list of knowledge conditions), and `extra_info.entities_gt` (placeholder-to-entity map such as `E0 → name`). The dataset schema should be checked against the restored V3 reward manager before running a real job. This repository does not construct or distribute these records. The bundled generic search tool does not add an authorization header; if your endpoint requires one, put an authenticated gateway/adapter in front of it. The inherited answer grader defaults to a Chinese-language judgment prompt; adapt it for English-only tasks before claiming benchmark parity.

## 🚀 Train KnowGPO

```bash
export KNOWGPO_MODEL_PATH=/path/to/sft-checkpoint
export KNOWGPO_TRAIN_FILES=/path/to/train.parquet
export KNOWGPO_VAL_FILES=/path/to/validation.parquet
export KNOWGPO_SEARCH_URL=https://your-search-service.example/api
export KNOWGPO_JUDGE_BASE_URL=https://your-openai-compatible-judge.example/v1
export KNOWGPO_JUDGE_API_KEY=...   # Keep private; never commit
export KNOWGPO_JUDGE_MODEL=your-judge-model

bash scripts/train_knowgpo.sh --dry-run
bash scripts/train_knowgpo.sh trainer.n_gpus_per_node=8 trainer.nnodes=1
```

The launcher supports additional Hydra overrides; a supplied key **replaces** its default and switching to another reward manager is rejected. It defaults to **40 prompts per batch**, matching the historical training script, and **16 rollouts per prompt**. The paper reports **32 prompts per batch**; to request that batch size, add `data.train_batch_size=32`. The historical script's 128K context and tensor-parallel-size-8 defaults are retained as a starting point: reduce them if your available hardware cannot support them. Learning rate, KL settings, validation cadence, and service/model versions must be assessed against your intended experiment. Neither the default nor an override is a claim of benchmark reproduction. Ray cluster provisioning and live services are not bundled.

## ✅ Offline Verification and Limits

```bash
python3.12 -m unittest discover -s tests/knowgpo -v
bash -n scripts/train_knowgpo.sh
bash scripts/train_knowgpo.sh --dry-run
```

The offline tests cover relevant file/config paths, knowledge grouping and immediate rewards on synthetic states, KnowGPO Phase-B dispatch/restoration, launcher preflight, and token-level advantage fusion. They do **not** run a model, fetch search results, call a judge, start SGLang/Ray training, or establish that reported paper scores are reproduced. Real training still requires user-provided data, weights, services, a compatible accelerator environment, and integration testing. Any credentials present in older source snapshots should be rotated or revoked if still valid; removing them from this release does not invalidate them.

## 🤝 Acknowledgements and License

KnowGPO builds on [veRL](https://github.com/volcengine/verl). The paper compares against [GiGPO](https://github.com/langfengQ/verl-agent) and [IGPO](https://github.com/GuoqingWang1/IGPO), and evaluates on Xbench, BrowseComp-ZH, and GAIA. This release retains the original repository's [Apache License 2.0](LICENSE). Third-party notices and rights in inherited `verl` components and other borrowed code should be reviewed before publishing the merged repository.
