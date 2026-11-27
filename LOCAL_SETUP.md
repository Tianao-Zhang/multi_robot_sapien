# MAWAM 本地任务与数据采集

七项 MA-Bench 场景均有本地 SAPIEN 实现。场景参照公开 TaskSpec 和示范重建，使用 RoboTwin 官方机器人资产、PhysX 和 mplib 逆解，并沿用 MA-Bench 评估器。相关上游版本见 [MAWAM](https://github.com/EmbodiedCity/MAWAM.code/tree/5220b570f0e85b660602a934ada6f6025e5c4f3a) 和 [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin/tree/ea8b21121ebb3cd201ff5b3fe361944ac94eda3f)。

## 运行

在本仓库目录执行。SAPIEN 进程需要访问本机 Vulkan 设备。下列命令使用 Python 3.10 环境 `.venv`，机器人资产位于 `third_party/RoboTwin/assets/embodiments`。

```bash
env -u PYTHONPATH .venv/bin/python -m ma_task_harness validate
env -u PYTHONPATH .venv/bin/python -m ma_bench.run_scenes \
  --seed 0 --output runs/all_scenes_rerun
env -u PYTHONPATH .venv/bin/python -m ma_bench.collect_demos \
  --task find_common_color --seed-start 0 --num-episodes 3 \
  --output datasets/local_color_rerun
```

`run_scenes` 默认逐项执行七个专家脚本，每项写出原生 25 Hz 总览视频 `overview.mp4` 和 `report.json`。用 `--task` 可指定任务，命令会汇总每项里程碑、物理检查与进程退出状态。录像覆盖完整本地动作及终态保持。

`collect_demos` 支持 `find_common_color`（别名 `color`）的同步多视角数据采集。`--output` 指向新目录。成功数据写入 `tasks/find_common_color/episodeN`，失败数据及原因保留在 `rejected`。每组含 `agent_a`、`agent_b` 两份 `trajectory.h5`，每台机器人各三路 320×240 RGB 视频，并有 `overview.mp4` 和 `episode_report.json`。视频和动作、测量位姿、相机内外参使用同一控制 tick。动作是 14 维双臂关节及夹爪命令，测量末端位姿使用各自桌面坐标系和 `wxyz` 四元数。相机名及 HDF5 字段参照公开数据。 [MA-Bench 数据集](https://huggingface.co/datasets/EmbodiedCity/MA-Bench)

## 重建依赖

安装 Python 3.10 和 `uv`。将 RoboTwin 2.0 发布的 `embodiments.zip` 保存到 `third_party/downloads/embodiments.zip`，再运行 `bash scripts/setup_sim.sh`。脚本校验归档、解压机器人资产并安装仿真依赖。`ROBOTWIN_ASSET_ARCHIVE` 可指定归档路径，`ROBOTWIN_ROOT` 可指定已有的资产 checkout。 [RoboTwin 资产发布](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/blob/main/embodiments.zip)

专家运行使用仿真状态和本地运动控制，场景验证和采集可以离线执行。
