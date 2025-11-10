# Noise-DA 复现指南

本文档汇总了在本仓库中复现《Denoising as Adaptation: Noise-Space Domain Adaptation for Image Restoration》论文的关键步骤，涵盖环境配置、数据准备、训练、推理与指标复现流程。请结合对应任务目录下的配置文件按需调整具体路径与参数。

## 1. 环境配置

1. 克隆代码并进入项目目录：
   ```bash
   git clone https://github.com/KangLiao929/Noise-DA
   cd Noise-DA
   ```
2. （推荐）使用 Conda 创建独立环境并安装依赖：
   ```bash
   conda create -n Noise-DA python=3.9
   conda activate Noise-DA
   pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
   pip install -r requirements.txt
   ```
   > 代码基于 PyTorch 2.1.2、CUDA 12.1 进行开发，如需在其他硬件环境运行，请对应调整 CUDA 与 PyTorch 版本。【F:README.md†L29-L58】

## 2. 快速推理（Demo）

1. 从提供的 Google Drive 链接下载预训练模型，放入 `checkpoints/` 目录。
2. 根据任务在 `configs_demo/` 中的示例 JSON 调整：
   - `path.checkpoint`：预训练权重路径。
   - `datasets.test.which_dataset.args.data_root`：待处理的退化图像路径。
3. 运行测试脚本：
   ```bash
   sh test.sh ./configs_demo/denoising.json   # 或 deraining.json / deblurring.json
   ```
   结果默认保存在 `results/` 目录下。【F:README.md†L60-L105】【F:configs_demo/denoising.json†L1-L70】【F:test.sh†L1-L5】

## 3. 数据准备

针对三类任务需分别准备合成与真实数据集，并使用 `path_wrapper.py` 生成 `.flist` 文件以便统一读取。可选地，可准备大规模干净图像（如 COCO、ImageNet）用于扩展的无配对条件训练。

### 3.1 图像去噪（Denoising）

1. 下载合成训练集（DIV2K、Flickr2K、WED、BSD）并生成裁剪块：
   ```bash
   python Denoising/download_data.py --data train --noise gaussian
   python Denoising/generate_patches_dfwb.py
   ```
2. 下载真实 SIDD 训练/验证/测试数据并生成裁剪块：
   ```bash
   python Denoising/download_data.py --data train --noise real
   python Denoising/generate_patches_sidd.py
   python Denoising/download_data.py --noise real --data test --dataset SIDD
   ```
3. 运行 `path_wrapper.py` 将数据路径写入 `.flist`：
   ```bash
   python path_wrapper.py --folder_path 'dataset/denoise/train/DFWB/input_crops' --flist_file './flist_name/denoise/DFWB.flist'
   python path_wrapper.py --folder_path 'dataset/denoise/train/SIDD/input_crops' --flist_file './flist_name/denoise/SIDD_train_input.flist'
   python path_wrapper.py --folder_path 'dataset/denoise/val/SIDD/input_crops' --flist_file './flist_name/denoise/SIDD_val_input.flist'
   python path_wrapper.py --folder_path 'dataset/denoise/val/SIDD/target_crops' --flist_file './flist_name/denoise/SIDD_val_gt.flist'
   # 可选：COCO 等无配对干净图像
   ```
【F:Denoising/README.md†L1-L47】

### 3.2 图像去雨（Deraining）

1. 下载 Rain13K（合成）与 SPA（真实）数据集。
2. 使用 `path_wrapper.py` 将训练与测试数据目录打包为 `.flist`：
   ```bash
   python path_wrapper.py --folder_path 'dataset/derain/train/Rain13K/input' --flist_file './flist_name/derain/rain13k_input.flist'
   python path_wrapper.py --folder_path 'dataset/derain/train/Rain13K/target' --flist_file './flist_name/derain/rain13k_gt.flist'
   python path_wrapper.py --folder_path 'dataset/derain/train/SPA/input' --flist_file './flist_name/derain/SPA_train.flist' --subfolders
   python path_wrapper.py --folder_path 'dataset/derain/test/SPA/input' --flist_file './flist_name/derain/real_test_1000_input.flist'
   python path_wrapper.py --folder_path 'dataset/derain/test/SPA/target' --flist_file './flist_name/derain/real_test_1000_gt.flist'
   ```
【F:Deraining/README.md†L1-L32】

### 3.3 图像去模糊（Deblurring）

1. 下载 GoPro（合成）与 RealBlur_J（真实）数据集。
2. 对 GoPro 数据集生成裁剪块并创建 `.flist`：
   ```bash
   python Deblurring/download_data.py --data train
   python Deblurring/generate_patches_gopro.py
   python path_wrapper.py --folder_path 'dataset/deblur/train/GoPro/input_crops' --flist_file './flist_name/deblur/gopro_train_patch_input.flist'
   python path_wrapper.py --folder_path 'dataset/deblur/train/GoPro/target_crops' --flist_file './flist_name/deblur/gopro_train_patch_gt.flist'
   ```
3. 使用 `Deblurring/data_split.py` 按预定义划分生成 RealBlur_J 的 `.flist`：
   ```bash
   python Deblurring/data_split.py --source_file_path './datasets/split_names/RealBlur_J_train_list.txt' \
     --output_file_path1 './flist_name/deblur/RealBlur_J_train_gt.flist' \
     --output_file_path2 './flist_name/deblur/RealBlur_J_train_input.flist' \
     --base_path './dataset/deblur/RealBlur_J/'
   python Deblurring/data_split.py --source_file_path './datasets/split_names/RealBlur_J_test_list.txt' \
     --output_file_path1 './flist_name/deblur/RealBlur_J_test_gt.flist' \
     --output_file_path2 './flist_name/deblur/RealBlur_J_test_input.flist' \
     --base_path './dataset/deblur/RealBlur_J/'
   ```
【F:Deblurring/README.md†L1-L48】

## 4. 训练流程

1. 根据任务在 `Denoising/`, `Deraining/`, `Deblurring/` 的 `configs/options_train.json` 中指定 `.flist` 数据路径、训练超参和日志目录。
2. （可选）如需启用无配对干净图像的扩展，将配置中的 `diff_flag` 设为 2，并在 `ref_root` 写入对应 `.flist`。
3. 启动训练：
   ```bash
   sh train.sh Denoising/configs/options_train.json   # 或 Deraining/... / Deblurring/...
   ```
   `train.sh` 默认使用 8 张 GPU（`-gpu '0, 1, 2, 3, 4, 5, 6, 7'`），可按需修改脚本中的 `-gpu` 参数以匹配设备数量。【F:Denoising/README.md†L49-L69】【F:Deraining/README.md†L34-L54】【F:Deblurring/README.md†L34-L54】【F:train.sh†L1-L6】
4. 若训练过程中中断，可在对应 `options_train.json` 中取消注释 `resume_state`，填入最新的断点权重路径后重新运行上述脚本。【F:Denoising/README.md†L71-L78】【F:Deraining/README.md†L56-L63】【F:Deblurring/README.md†L56-L63】

## 5. 推理与评估

1. 在任务目录的 `configs/options_test.json` 中设置预训练权重与测试数据的 `.flist` 路径。
2. 运行：
   ```bash
   sh test.sh Denoising/configs/options_test.json   # 或 Deraining/... / Deblurring/...
   ```
   `test.sh` 默认单卡运行（`-gpu '0'`），可在脚本内修改。【F:Denoising/README.md†L80-L93】【F:Deraining/README.md†L65-L78】【F:Deblurring/README.md†L65-L78】【F:test.sh†L1-L5】
3. 指标复现：
   - 去噪（SIDD）：在 MATLAB 中运行 `Denoising/eva_denoise.m`，输入去噪结果 `Idenoised.mat` 与官方 `ValidationGtBlocksSrgb.mat` 计算 PSNR/SSIM。
   - 去雨（SPA）：运行 `Deraining/eva_derain.py` 评估。
   - 去模糊（RealBlur_J）：运行 `Deblurring/eva_deblur.py` 评估。
【F:Denoising/README.md†L94-L109】【F:Deraining/README.md†L80-L93】【F:Deblurring/README.md†L80-L93】

## 6. 目录结构提示

- `run.py`：统一的训练/测试入口，负责解析配置、构建数据集、模型与损失，并在分布式环境下启动进程。【F:run.py†L1-L111】
- `train.sh` / `test.sh`：封装 `run.py` 的 shell 脚本，分别用于训练与推理批处理。【F:train.sh†L1-L6】【F:test.sh†L1-L5】
- `configs_demo/`：提供推理示例 JSON，可作为自定义配置的参考。【F:configs_demo/denoising.json†L1-L70】
- `inputs/`：示例退化图像。
- `results/`：默认输出目录。

按照上述步骤准备数据、调整配置并执行脚本，即可在本仓库中完整复现论文实验流程。
