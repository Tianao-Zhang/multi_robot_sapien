# 本地开发与服务器运行

首次在服务器的项目父目录执行 `git clone https://github.com/Tianao-Zhang/multi_robot_sapien.git`。后续本地在 `main` 提交、执行 `git push`，服务器在 `multi_robot_sapien` 目录执行 `git pull --ff-only`。代码、配置和脚本随 Git 同步。两套 Python 环境、机器人资产、数据、录像、日志和模型权重保存在各自机器的项目目录。

服务器到 GitHub 的连接不稳定时，可以从本地经 SSH 同步。首次在服务器仓库运行 `git config --local receive.denyCurrentBranch updateInstead`，在本地添加 `git remote add server9003 ssh://SERVER/ABSOLUTE/PATH/TO/multi_robot_sapien`。之后在本地执行下面两条命令，第二条推送会在服务器工作区干净时同步检出的代码。

```bash
git push origin main && git push server9003 main
```

## 环境

安装 `uv`、Python 3.10、`unzip` 和支持 Vulkan/CUDA 的 NVIDIA 驱动。仿真环境和模型环境分别使用 `.venv` 和 `.venv-train`。可用 `UV_BIN` 指定项目目录里的 `uv`，例如 `export UV_BIN="$PWD/.tools/bin/uv"`。

```bash
# 准备好 third_party/downloads/embodiments.zip 后执行
bash scripts/setup_sim.sh

# Linux x86-64 和 CUDA 12.8
bash scripts/setup_train.sh
source .venv-train/bin/activate
HF_HOME="$PWD/.tools/hf" hf download robbyant/lingbot-va-base \
  --revision 68b7bc1b35da6ddc67ea94c4ceb58d768fbb3f9c \
  --include 'tokenizer/*' 'text_encoder/*' 'vae/*' 'transformer/*' \
  --local-dir checkpoints/lingbot-va-base
```

仿真资产来源和输出格式见 [本地仿真说明](LOCAL_SETUP.md)。训练脚本安装 Python 3.12、PyTorch 2.9.0 CUDA 12.8、FlashAttention 和指定版本的 LingBot-VA 源码。LingBot-VA 的 LeRobot 数据加载只在调用其原版训练器时导入，MAWAM 使用自己的数据集。 [LingBot-VA 基础权重](https://huggingface.co/robbyant/lingbot-va-base/tree/68b7bc1b35da6ddc67ea94c4ceb58d768fbb3f9c)

## 场景、采集与数据准备

从仓库根目录执行。用新的 `run_id` 为每批结果保留独立目录。

```bash
run_id="$(date -u +%Y%m%dT%H%M%SZ)"
raw="datasets/color_${run_id}"
prepared="datasets/color_${run_id}_training"

env -u PYTHONPATH .venv/bin/python -m ma_bench.run_scenes \
  --seed 0 --output "runs/scenes_${run_id}"
env -u PYTHONPATH .venv/bin/python -m ma_bench.collect_demos \
  --task find_common_color --seed-start 0 --num-episodes 3 --output "$raw"
env -u PYTHONPATH .venv-train/bin/python -m mawam.data.prepare_collected \
  --input-root "$raw" --output "$prepared"
```

七个场景可分别运行和录像。同步数据采集与上述转换入口支持 `find_common_color` 和 `joint_bowl_placing`。转换只读取已验收的示范，按控制 tick 检查双机器人和三路镜头的同步。RGB 以 6.25 Hz 输入 Wan VAE。输出包含 Stage 1 逐机器人清单、Stage 2 同步配对清单及冻结的 T5 文本缓存。

## 训练

在同一个终端中使用上一步的 `prepared` 和 `run_id`。根据本次分配的显卡设置 `CUDA_VISIBLE_DEVICES` 和 `NPROC`。启动器使用单机独立端口，默认八张卡，每阶段默认 3,000 次更新。Stage 2 从此次 Stage 1 的训练权重初始化。

9003 的双卡训练验证使用 NCCL 本机共享内存通信。启动训练前执行 `export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1`。

```bash
source .venv-train/bin/activate
stage1_run="runs/color_${run_id}_stage1"
DATASET_ROOT="$prepared/stage1" RUN_ROOT="$stage1_run" \
  bash scripts/train_stage1.sh
DATASET_ROOT="$prepared/stage2" RUN_ROOT="runs/color_${run_id}_stage2" \
  STAGE1_CKPT="$stage1_run/checkpoints/checkpoint_step_3000" \
  bash scripts/train_stage2.sh
```

## 四臂接力放碗

采集验收要求 A 左臂、A 右臂、B 右臂、B 左臂按顺序接触并抬升同一只碗，且通过 MA-Bench 的六项里程碑。保留所有失败尝试及其诊断。在 250 Hz 仿真、25 Hz 控制下记录全部控制指令和实测状态，并在每四个控制 tick 保存双机器人各三路 RGB 和总览。末端动作目标由实际关节驱动指令正运动学得到。

```bash
raw="datasets/bowl_${run_id}"
run="runs/bowl_${run_id}"
.venv/bin/python -m ma_bench.collect_demos --task bowl --num-episodes 200 \
  --max-attempts 320 --workers 4 --seed-start 0 --video-stride 4 --output "$raw"
.venv/bin/python -m ma_bench.audit_bowl_collection --collection "$raw" \
  --output "$run/collection_audit.json"
```

将已验收示范的前 100 对用于训练，后 100 对用于保留验证。动作归一化分位数由训练划分拟合。可用 `--instructions` 读取数据发布中的任务指令。

```bash
.venv-train/bin/python -m mawam.data.prepare_collected --input-root "$raw" \
  --output "${raw}_train" --first 0 --count 100 --fit-action-norm
.venv-train/bin/python -m mawam.data.prepare_collected --input-root "$raw" \
  --output "${raw}_holdout" --first 100 --count 100
```

论文模式闭环评测使用 `benchmark` 推理配置，预先由专家验证评测种子，单试验上限 90 秒。用 `ma_bench.eval_runner --task bowl --seed-file ... --num-trials 100 --video-dir ... --output ...` 运行、逐场保存视频与指标。本地资产、专家路径和硬件与论文原始环境有差异。 [论文与实验设置](https://arxiv.org/html/2611.29501v1)
