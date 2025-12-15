# domainAdaptation

该工程用于复现《Deep domain adaptation eliminates costly data required for task-agnostic wearable robotic control》中的无监督域适配流程，覆盖从原始 CSV 加载到 CycleGAN 训练及模态归一化损失计算的完整链路。下面按照“数据 → 归一化 → 模型 → 损失 → 训练脚本”的顺序进行说明。

## 数据加载与预处理
- **统一数据集**：`SensorDataset` 会遍历数据根目录下的所有 trial，真实域读取 `*_exo.csv`，模拟域读取 `*_exo_sim.csv`，二者共享同一套 `input_names`/`label_names`，仅通过文件后缀来区分数据来源（`domain_adaptation/data/sensor_dataset.py:15`）。
- **trial 缓存**：第一次遍历目录树时会将 trial 相对路径（含参与者/动作）缓存到 `cache/trials_*.json`，后续复用以避免重复 I/O（`sensor_dataset.py:92`）。
- **通道级 z-score**：加载 trial 时先通过 `_get_or_compute_channel_stats` 统计每个通道的均值/方差/标准差及“标准化后”的方差，再写入 `cache/channel_stats_*.json`（`sensor_dataset.py:126`）。取样阶段用 `(x-mean)/std` 对所有输入通道做归一化，保证 GAN 输入量纲一致。
- **随机窗口与 NaN 规避**：`scripts/train_pipeline.py` 在 `_collate_windows` 中为每个 batch 调用 `_random_window`，仅在无 NaN 的连续片段上裁剪；若当前片段含 NaN 就跳到下一段，否则在最后对不足长度的窗口补零以保持 `--seq-len` 定长（`scripts/train_pipeline.py:60`）。

## IMU 模态 σ<sub>max</sub> 计算
- **模态划分**：目前针对论文强调的 4 个 IMU 模态（大腿/小腿加速度、角速度）进行归一化映射，逻辑在 `_identify_modality`（`scripts/train_pipeline.py:98`）。若需要扩展到其它传感器，只要在该函数里追加匹配规则即可。
- **方差聚合**：`_aggregate_channel_stats` 会遍历 Dataset/ConcatDataset 内所有子集，按照 trial 数加权平均 “标准化后的方差”，得到每个通道的整体波动水平（`scripts/train_pipeline.py:117`）。
- **σ<sub>max</sub> 生成**：`_compute_modality_scales` 以模态为单位取对角方差中的最大值，生成 Hadamard 归一化所需的 `sim_sigma_max` / `real_sigma_max`，并在训练前注入 GAN 配置（`scripts/train_pipeline.py:133`）。

## 模型结构
- **生成器**：Sim→Real 与 Real→Sim 都采用 1D U-Net（金字塔深度、基础通道数可通过 `--unet-depth`、`--base-channels` 控制），卷积块包含 Conv1d+InstanceNorm+LeakyReLU，瓶颈后通过上采样和 skip connection 恢复时间分辨率（`domain_adaptation/models/gan.py:14`）。
- **判别器**：使用 PatchGAN 样式的 1D 卷积堆栈（逐层 stride=2 下采样）来评估局部一致性，既用于真实域也用于模拟域（`gan.py:94`）。
- **优化器**：生成器和判别器分别用 Adam，学习率及 β 来自 `GanConfig`，训练时自动移动到 `--device` 指定的硬件（`gan.py:146`）。

## 损失构成与模态归一化
- **对抗损失 `L_GAN`**：采用 LS-GAN 形式，迫使生成后的序列在对应判别器前与目标域分布匹配（`gan.py:178`）。
- **循环一致 `L_cycle`**：模拟→真实→模拟、真实→模拟→真实两条链路分别做平方误差，误差先按模态的 σ<sub>max</sub> 做 Hadamard 归一化，再平均到各通道，保证大方差模态不会压制其它模态（`gan.py:167`）。
- **身份损失 `L_id`**：当输入与输出的通道数匹配时，让生成器在“本域输入”上保持恒等，仍然使用模态归一化后的 MSE，并以 0.5 系数稳定训练（`gan.py:185`）。
- **判别器损失**：标准的平方误差形式，分别对真实/模拟域求和（`gan.py:207`）。
- **总体目标**：`gen_total = λ_gan·L_GAN + λ_cycle·L_cycle + λ_id·L_id`，权重可通过命令行覆盖（`train_pipeline.py` CLI 中的 `--lambda-*`）。

重构类损失（循环、身份，以及未来若添加监督损失）都依赖 `_normalized_mse`：
```python
loss = ((prediction - target) ** 2) / sigma_max
loss = loss.mean()
```
若模态未被识别到，则默认权重为 1，等价于普通 MSE。

## 训练脚本与日志
- **入口**：`python scripts/train_pipeline.py --seq-len 256 --batch-size 32 ...`
- **数据准备**：`_prepare_real_dataset` 与 `_prepare_sim_dataset` 分别加载真实/模拟域，并返回 `Subset`（真实域还会执行 NaN trial 过滤）。
- **DataLoader**：`_build_loader` 根据 `--seq-len`、`--num-workers` 构造等长窗口的 batch，并在 epoch 内不断循环迭代器以保证 real/sim 步调一致。
- **运行目录**：通过 `--output-dir` 与 `--run-name` 组合生成，若重名自动附加时间戳，所有 checkpoint（`best.pt` 与 `last.pt`）都会写入该目录。
- **MLflow**：传入 `--mlflow --mlflow-uri ./mlflowruns --mlflow-experiment debug --run-name demo` 即可启用，脚本会记录命令行参数、数据集规模、逐步 loss，并自动上传 checkpoint（`scripts/train_pipeline.py:230`）。
- **进度条**：每个 epoch 用 `tqdm` 展示 step 级别进度，`--log-interval` 控制打印/MLflow 写入频率，日志包含 `gen_total / adv_loss / cycle_loss / identity_loss / disc_loss`。

## 常见操作
1. **重新统计 trial/通道信息**：删除 `cache/trials_*.json` 或 `cache/channel_stats_*.json` 后重新运行脚本即可强制扫描与统计。
2. **切换输入模态**：在 `domain_adaptation/config/dataset_config.py` 修改 `real_sensor_names` 或传入 `--data-dirs / --side / --action-patterns`，`SensorDataset` 会在初始化时自动替换 `*` 占位符。
3. **扩展模态归一化**：若需要让更多传感器参与 Hadamard 归一化，只需在 `_identify_modality` 中匹配对应列名并重新生成缓存。

以上流程确保“真实域 / 模拟域 → 逐通道 z-score → IMU 模态 σ<sub>max</sub> 归一化 → CycleGAN”严格贴合论文描述的设定，方便后续对接 TCN 力矩估计器或半监督扩展。
