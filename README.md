# MAWAM: Joint Imagination for Collaborative Robot Manipulation with a Multi-Agent World Action Model

<p align="center">
  <a href="https://embodiedcity.github.io/MAWAM/"><b>🌐 Project Page</b></a> &nbsp;|&nbsp;
  <a href="https://arxiv.org/abs/2611.29501"><b>📚 arXiv</b></a> &nbsp;|&nbsp;
  <a href="https://arxiv.org/pdf/2611.29501"><b>📄 Paper (PDF)</b></a> &nbsp;|&nbsp;
  <a href="https://huggingface.co/EmbodiedCity/MAWAM"><b>🤗 Models</b></a> &nbsp;|&nbsp;
  <a href="https://huggingface.co/datasets/EmbodiedCity/MA-Bench"><b>📦 MA-Bench</b></a>
</p>

**Rongze Tang**<sup>1,3</sup>, **Chenyu Zhou**<sup>2,3</sup>, **Jianjie Fang**<sup>4</sup>, **Ziming Liu**<sup>4</sup>, **Zhaolu Wang**<sup>3</sup>, **Ziyou Wang**<sup>4</sup>, **Xvyuan Liu**<sup>4</sup>, **Wei Wu**<sup>5</sup>, **Chen Gao**<sup>3,4,&dagger;</sup>, **Yong Li**<sup>4</sup>, **Zhibo Chen**<sup>1,3,&dagger;</sup>

<sup>1</sup>University of Science and Technology of China &nbsp;&middot;&nbsp; <sup>2</sup>Nanjing University &nbsp;&middot;&nbsp; <sup>3</sup>Zhongguancun Academy &nbsp;&middot;&nbsp; <sup>4</sup>Tsinghua University &nbsp;&middot;&nbsp; <sup>5</sup>Manifold AI  
<sup>&dagger;</sup>Corresponding authors (`chgao96@gmail.com`, `chenzhibo@ustc.edu.cn`)

---

## Overview

World action models (WAMs) advance robotic manipulation by grounding control in visual foresight, yet existing formulations remain single-embodiment. **MAWAM** introduces the **agent** as a first-class modeling axis inside one weight-shared causal diffusion transformer:

1. **MA-Task Harness (`ma_task_harness/`)**: A three-stage coding-agent co-evolution framework that formalizes *multi-agent-necessary* tasks along three axes (*why*, *when*, *who*) into a `TaskSpec`, co-evolves generation ($`G^{(n)}`$) and evaluation ($`E^{(n)}`$) sub-agents under deterministic LLM-free gates, and certifies labels with operational verification probes.
2. **MA-Bench (`ma_bench/`)**: Seven closed-loop multi-robot tasks spanning asynchronous handoffs, cross-table partial observability, partially synchronous holds, and synchronous $`SE(3)`$ / dynamic co-manipulation across homogeneous and heterogeneous robot teams, scored by 43 deterministic milestones ($`S_k`$, $`S_{\mathrm{collab}}`$, $`\mathrm{SR}`$).
3. **MAWAM Model, Training & Inference (`mawam/`)**: Joint spatiotemporal imagination via weight-shared joint causal self-attention, hierarchical gated global + local language conditioning, fixed Fourier physical ID + normalized role encodings, two-stage agent-axis curriculum (Stage 1 $`N = 1`$ → Stage 2 $`N > 1`$), and both benchmark (`25/50` steps) and accelerated real-time (`4/8` steps, `0.72 s/chunk`) closed-loop inference.

---

## Repository Structure

```text
MAWAM/
├── configs/
│   ├── tasks/*.yaml               # Canonical configs for all 7 MA-Bench tasks
│   ├── training/
│   │   ├── stage1.yaml            # N=1, 8x H20 FSDP2 BF16, lr=1e-5, 3,000 updates
│   │   └── stage2.yaml            # N=2, 8x H20 FSDP2 BF16, lr=1e-5 / gate 1e-4, 3,000 updates
│   └── inference/
│       ├── benchmark.yaml         # C=2, W=64, 25 video (CFG 5.0) / 50 action (CFG 1.0)
│       └── realtime.yaml          # C=2, W=64, 4 video / 8 action + fused QKV + CUDA Graphs
├── ma_task_harness/               # [1] Task-agnostic coding-agent co-evolution harness (§3.2, Appendix A–B)
│   ├── spec/                      # Open N-agent TaskSpec schema (supports arbitrary robots & predicates)
│   ├── runtime/                   # SapienMultiRobotWorld (real RoboTwin/SAPIEN+PhysX+mplib), state types, subprocess runner
│   ├── gates/                     # Universal physical checks (i)–(iv), Eq.(4–5) scorer, Appendix A probe arbiter
│   ├── agents/                    # SPECIFIER / GENERATOR / EVALUATOR charters, skills, code templates
│   ├── orchestrator/              # Backends (claude | codex | openai), Stage 1 specifier, Stage 2–3 driver & CLI
│   ├── viz/                       # Unified chronological dialogue HTML trace viewer with SAPIEN RGB playback
│   ├── examples/mabench/          # The 7 MA-Bench TaskSpecs (reference examples)
│   └── tests/                     # Task-agnostic 3-stage & N=3 unit tests
├── ma_bench/                      # [2] MA-Bench benchmark & evaluators (§3.3, Appendix C)
│   ├── registry.py                # 7 MA-Bench tasks & 3 real-robot tasks metadata
│   ├── action_codec.py            # Shared 30-D masked action codec & single-arm view masks
│   ├── metrics.py                 # Role Score S_k, Collaboration Score S_collab, SR, 95% Wilson CI
│   ├── evaluators/                # Deterministic, LLM-free evaluators for all 7 tasks (43 milestones)
│   ├── envs/                      # RoboTwin 2.0 / SAPIEN multi-robot environment interface
│   └── eval_runner.py             # Synchronous 100-trial closed-loop evaluation runner
├── mawam/                         # [3] MAWAM model, training & inference (§4, Appendix D & G)
│   ├── model/                     # Joint causal mask (Eq. 7), Stage2WanTransformer3DModel, masks
│   ├── data/                      # Stage 1 & Stage 2 datasets, pair builder, frozen T5 text cache
│   ├── training/                  # Eq. (9) loss, AdamW groups, FSDP2 DCP ckpts, Stage 1/2 trainers
│   └── inference/                 # Sliding joint KV cache, fused kernels, streaming VAE, policy server
├── scripts/                       # Checkpoint download, 8-GPU training, serving, and eval launchers
└── tests/                         # CPU unit tests for masks, evaluators, configs, and hygiene
```

---

## Installation

For the local SAPIEN scenes and the Git-based server workflow, see [deployment and training](DEPLOYMENT.md) and [simulation and recording](LOCAL_SETUP.md).

### 1. Core Package & Offline Verification (CPU-only, no GPU or API key required)

```bash
git clone https://github.com/EmbodiedCity/MAWAM.code.git
cd MAWAM.code
pip install -e .
```

### 2. Model Training & Inference Environment (CUDA GPU)

Requires CUDA-enabled PyTorch (≥ 2.5, tested with PyTorch 2.9 on NVIDIA H20 GPUs) and an upstream [LingBot-VA](https://github.com/Robbyant/lingbot-va) checkout (`--upstream-root ./third_party/LingBot-VA`):

```bash
pip install -r requirements-model.txt
```

### 3. MA-Bench Simulation Environment (RoboTwin 2.0 + SAPIEN)

Requires [RoboTwin 2.0](https://github.com/RoboTwin-Platform/RoboTwin) (`ROBOTWIN_ROOT`) and 3D task assets (`MA_BENCH_ASSETS`):

```bash
pip install -r requirements-sim.txt
```

---

## 1. MA-Task Harness

MA-Task Harness operationalizes the *Why–When–Who* task taxonomy (§3.1–3.2, Appendix A–B) for **any user-defined multi-robot task** ($`N \ge 2`$ homogeneous or heterogeneous robots; our 7 MA-Bench specs live in `ma_task_harness/examples/mabench/` as reference examples):
- **Stage 1 (Specification — `specify`)**: Clarifies team capabilities $`\mathcal{E}_k = (\mathcal{O}_k, \mathcal{W}_k, \mathcal{U}_k)`$, hierarchical prompts $`(g, \{\ell^k\})`$, named thresholds, open-ended milestones $`(M_k, M_{\mathrm{collab}})`$, and probes into a validated `TaskSpec.yaml`.
- **Stage 2 (Bidirectional Code Co-Evolution — `coevolve`)**: Spawns isolated `generator` ($`G^{(n)}`$, writing `gen/scene.py` and `gen/expert.py`) and `evaluator` ($`E^{(n)}`$, writing deterministic LLM-free `eval/evaluator.py`) workspaces exchanging only disk artifacts (`audit_report.json`, `counter_audit.json`), verified every round in an isolated subprocess against universal physical gates (i)–(iv), Eq. (4)–(5) contracts, and Appendix A probes.
- **Stage 3 (Certification, Freeze & Trace Viewer — `verify-pkg`, `render-trace`)**: Certifies `single_<k>`, $`O \prec W \prec U`$ relaxations, `serialize`, delay sweeps, and `role_swap` before freezing a SHA-256 verified task package and rendering a standalone interactive HTML trace report.

### Author Any Custom Multi-Robot Task

```bash
# 1. Stage 1: Clarify any custom task idea -> validated TaskSpec.yaml
python3 -m ma_task_harness specify \
  --backend claude --model sonnet \
  --work-dir ./runs/my_custom_task \
  --interactive \
  --idea "Describe your N-robot collaborative task..."

# 2. Stage 2 & 3: Real SAPIEN + PhysX + mplib dual-agent G^(n) <-> E^(n) co-evolution + probes + freeze + dialogue HTML
ROBOTWIN_ROOT=./third_party/RoboTwin \
python3 -m ma_task_harness coevolve \
  --spec ./runs/my_custom_task/TaskSpec.yaml \
  --backend claude --model sonnet \
  --sim-backend sapien \
  --sim-python /path/to/sapien_env/bin/python \
  --work-dir ./runs/my_custom_task \
  --max-rounds 10 \
  --render-html ./runs/my_custom_task/trace_dialogue.html

# 3. Independently re-verify the frozen package on real SAPIEN + PhysX
python3 -m ma_task_harness verify-pkg ./runs/my_custom_task/package \
  --sim-backend sapien --sim-python /path/to/sapien_env/bin/python
```

### Offline Checks & MA-Bench Examples (Pure CPU)

```bash
python3 ma_task_harness/tests/test_harness_generic.py
python3 -m ma_task_harness list-examples
python3 -m ma_task_harness validate
```

---

## 2. MA-Bench

MA-Bench comprises **7 multi-agent-necessary tasks** (**700** accepted demonstration pairs total, 100 per task; **100** independent closed-loop evaluation trials per task; 250 Hz physics / 25 Hz control; 43 deterministic milestones):

| Task | When | Why | Who (Team) | Milestones | Duration | Released Checkpoint (`EmbodiedCity/MAWAM`) |
| --- | --- | --- | --- | :---: | :---: | --- |
| **Joint bowl placing** | Async. | Spatial | Homo. (ALOHA + ALOHA) | 6 | 90 s | `ckpt/MAWAM-Joint-Bowl-Placing` |
| **Find common color** | Async. | Information | Homo. (ALOHA + ALOHA) | 6 | 13 s | `ckpt/MAWAM-Find-Common-Color` |
| **Hold and scan** | Partial | Spatial | Homo. (ALOHA + ALOHA) | 6 | 56 s | `ckpt/MAWAM-Hold-and-Scan` |
| **Open door, place pot** | Partial | Capability | Hetero. (Panda + ALOHA) | 6 | 62 s | `ckpt/MAWAM-Open-Door-Place-Pot` |
| **Lift tray, keep blocks** | Sync. | Capability | Homo. (ALOHA + ALOHA) | 7 | 40 s | `ckpt/MAWAM-Lift-Tray-Keep-Blocks` |
| **Lift tray, pour food** | Sync. | Capability | Homo. (ALOHA + ALOHA) | 6 | 39 s | `ckpt/MAWAM-Lift-Tray-Pour-Food` |
| **Pour and catch** | Sync. | Capability | Hetero. (Panda + UR5) | 6 | 38 s | `ckpt/MAWAM-Pour-and-Catch` |

### Download MA-Bench Dataset (`EmbodiedCity/MA-Bench`)

All **700 synchronized demonstration pairs** (100 episodes per task, 852,514 joint control steps, 3,900 multi-view RGB videos, and 1,400 HDF5 trajectories) are publicly hosted on Hugging Face:

```bash
pip install -U huggingface_hub

# Download a single task (e.g., Open door, place pot)
hf download EmbodiedCity/MA-Bench \
  --repo-type dataset \
  --include "tasks/open_door_place_pot/*" \
  --local-dir ./datasets/ma_bench

# Or download all 7 tasks (700 demonstration pairs, ~5.6 GB)
hf download EmbodiedCity/MA-Bench \
  --repo-type dataset \
  --local-dir ./datasets/ma_bench
```

### Main Simulation Results (Table 2, 100 Independent Trials per Task)

| Method | Mean Role $`S_k`$ (%) | Mean Collab $`S_{\mathrm{collab}}`$ (%) | Mean Joint $`\mathrm{SR}`$ (%) |
| --- | :---: | :---: | :---: |
| $`\pi_{0.5}`$ (per-robot) | 31.4 | 15.1 | 10.0 |
| LingBot-VA (per-robot) | 81.9 | 60.6 | 52.0 |
| $`\pi_{0.5}`$ (joint) | 4.4 | 0.1 | 0.0 |
| LingBot-VA (joint) | 40.9 | 18.4 | 13.7 |
| MA-VLA | 41.6 | 21.1 | 14.9 |
| VLA-Fabric | 45.5 | 24.0 | 17.1 |
| **MAWAM (ours)** | **94.3** (+12.4) | **87.2** (+26.6) | **82.7** (+30.7) |

---

## 3. MAWAM Training & Inference Pipeline

### Model & Hyper-Parameters (Paper Section 4 & Appendix Table 6)

| Component | Setting |
| --- | --- |
| **Backbone** | `Stage2WanTransformer3DModel` built on `robbyant/lingbot-va-base` (30 DiT blocks, 24 heads × 128, FFN 14336, ~6.24B params with global CA) |
| **Encodings & Gates** | Fixed unit-RMS Fourier physical IDs $`\phi(n_k)`$ + RMS-normalized mean role vector $`r_k`$ ($`\beta = 0.01`$); global CA gate $`\gamma_b = 10^{-3}`$ |
| **Unified Action Slots** | Shared 30-D slot layout with binary embodiment mask $`m_{e_k}`$: ALOHA 16-D (`[0..13, 28, 29]`), Panda/UR5/FR3 8-D (`[0..6, 28]`), Cobot 14-D (`[0..5, 7..12, 28, 29]`) |
| **Video Canvas** | 3 RGB views/robot tiled onto `24×20` latent canvas (head `16×20`, wrists `8×10` each, `48` channels, patch `1×2×2` → 120 tokens/frame); missing single-arm right wrist view-masked in loss |
| **Optimization** | 8× NVIDIA H20 GPUs, FSDP2 `bfloat16`, full activation checkpointing, batch size 8 episodes, AdamW $`\beta = (0.9, 0.95)`$, weight decay `0.1`, grad clip `2.0`, constant LR `1e-5` (`1e-4`, zero decay for $`\gamma_b`$) |
| **Curriculum** | **Stage 1** ($`N = 1`$): `3,000` updates/task (1:1 hetero balancing) → **Stage 2** ($`N = 2`$): `3,000` updates/task, joint CFG dropout `0.1`, $`C \in \{1..4\}`$, $`W \in \{4..64\}`$ |
| **Flow Schedule** | 1,000 levels; SNR shift `5.0` (video), `1.0` (action); $`C = 2`$ (32 actions/chunk), sliding joint KV cache $`W = 64`$ |
| **Inference Presets** | **Benchmark**: `25` video steps (CFG `5.0`) / `50` action steps (CFG `1.0`) · **Real-Time**: `4` video / `8` action steps + fused QKV + compiled 3-D RoPE + cached text KVs + CUDA Graphs (`0.72 s/chunk` single GPU) |

### Download Released Checkpoints

```bash
# Download a single task checkpoint (e.g., Open door, place pot)
python3 scripts/download_checkpoints.py --task door --local-dir ./mawam_ckpts

# Or download all 7 MA-Bench checkpoints
python3 scripts/download_checkpoints.py --task all --local-dir ./mawam_ckpts
```

### Data Preparation (Frozen T5 Cache & Stage 2 Pairing)

```bash
# 1. Pre-encode global prompt g and local role prompts l^k with frozen T5
python3 -m mawam.data.prepare_text_cache \
  --task open_door_place_pot \
  --base-model ./checkpoints/lingbot-va-base \
  --output ./datasets/open_door_place_pot/stage2

# 2. Index & audit synchronized Stage 2 multi-robot episode pairs from Stage 1 episodes
python3 -m mawam.data.prepare_pairs \
  --task open_door_place_pot \
  --stage1-root ./datasets/open_door_place_pot/stage1 \
  --output-root ./datasets/open_door_place_pot/stage2
```

### Two-Stage Training (8× GPU FSDP2)

```bash
# Stage 1 (N=1, 3,000 updates)
DATASET_ROOT=./datasets/open_door_place_pot/stage1 \
UPSTREAM_ROOT=./third_party/LingBot-VA \
BASE_CKPT=./checkpoints/lingbot-va-base \
RUN_ROOT=./runs/open_door_place_pot/stage1 \
bash scripts/train_stage1.sh

# Stage 2 (N=2, 3,000 updates initialized from Stage 1)
DATASET_ROOT=./datasets/open_door_place_pot/stage2 \
UPSTREAM_ROOT=./third_party/LingBot-VA \
STAGE1_CKPT=./runs/open_door_place_pot/stage1/checkpoints/checkpoint_step_3000 \
RUN_ROOT=./runs/open_door_place_pot/stage2 \
bash scripts/train_stage2.sh
```

### Closed-Loop Inference & MA-Bench Evaluation

Start the MAWAM policy server (Algorithm 2):

```bash
# Benchmark preset (25 video / 50 action steps, W=64, C=2)
TASK=open_door_place_pot \
CHECKPOINT=./mawam_ckpts/ckpt/MAWAM-Open-Door-Place-Pot \
BASE_MODEL=./checkpoints/lingbot-va-base \
TEXT_CACHE=./datasets/open_door_place_pot/stage2/text/text_cache.pt \
UPSTREAM_ROOT=./third_party/LingBot-VA \
PRESET=benchmark \
bash scripts/serve_policy.sh

# Real-time preset (4 video / 8 action steps + fused QKV + compiled RoPE + CUDA Graphs, 0.72 s/chunk)
PRESET=realtime \
TASK=open_door_place_pot \
CHECKPOINT=./mawam_ckpts/ckpt/MAWAM-Open-Door-Place-Pot \
BASE_MODEL=./checkpoints/lingbot-va-base \
TEXT_CACHE=./datasets/open_door_place_pot/stage2/text/text_cache.pt \
UPSTREAM_ROOT=./third_party/LingBot-VA \
bash scripts/serve_policy.sh
```

Run 100 independent closed-loop evaluation trials on MA-Bench:

```bash
TASK=open_door_place_pot NUM_TRIALS=100 bash scripts/eval_mabench.sh
```

### Real-Robot Deployment (Franka FR3 + AgileX Cobot Magic)

Our heterogeneous real-robot cell pairs a 7-DoF **Franka FR3** (`8` active channels) with a dual-arm **AgileX Cobot Magic** (`14` active channels, 2×6-DoF PiPER arms) at **30 Hz** control / **7.5 Hz** RGB ingestion (`mawam/inference/real_robot_client.py`). Using the `realtime` preset (`4/8` steps, `0.72 s/chunk < 1.07 s` budget), MAWAM achieves **63.3%** mean joint success across *Fruit basket* (`75.0%`), *Toy handoff & sort* (`50.0%`), and *Funnel cube pour* (`65.0%`) over 20 trials per task (vs. `13.3%` per-robot and `0.0%` joint LingBot-VA).

---

## Verification Tests (CPU)

Run the full offline verification suite to check TaskSpecs, probes, co-evolution, Eq. (7) causal masks, 30-D embodiment codecs, Eq. (9) masked loss, Table 6 configs, all 7 MA-Bench evaluators, and repository path hygiene:

```bash
python3 ma_task_harness/tests/test_harness_generic.py
python3 tests/test_attention_and_masks.py
python3 tests/test_mabench_evaluators.py
python3 tests/test_leak_hygiene.py
```

---

## License & Acknowledgements

- Released under the [Apache-2.0 License](LICENSE).
- Built upon [LingBot-VA](https://github.com/Robbyant/lingbot-va), [Wan](https://github.com/Wan-Video/Wan2.2), [RoboTwin 2.0](https://github.com/RoboTwin-Platform/RoboTwin), and [ALOHA](https://tonyzhaozh.github.io/aloha/).

## Citation

```bibtex
@article{tang2026mawam,
  title   = {MAWAM: Joint Imagination for Collaborative Robot Manipulation
             with a Multi-Agent World Action Model},
  author  = {Rongze Tang and Chenyu Zhou and Jianjie Fang and Ziming Liu and
             Zhaolu Wang and Ziyou Wang and Xvyuan Liu and Wei Wu and
             Chen Gao and Yong Li and Zhibo Chen},
  journal = {arXiv preprint arXiv:2611.29501},
  year    = {2026}
}
```
