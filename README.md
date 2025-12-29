# AiVoiceAssistant

简体中文说明文档 — 本仓库目标是在 RKNN NPU（如 rk3588 / rk3588s / rk3588s+ 等）上运行离线语音识别与 TTS 相关组件，并提供一键化的环境构建流程（make 控制）。

## 项目概览

本项目包含：

- RKNN 格式模型（存放在 `models/`）用于在 NPU 上推理（Zipformer ASR、Qwen 推理等）。
- `libs/rknn/`：项目内置的 RKNN 运行时二进制（librknnrt.so）与离线安装包（whl），便于在无互联网或需要特定版本时使用。
- 一套 Makefile 用于自动化创建虚拟环境、安装依赖、注入运行时库路径、下载/转换模型、以及硬件冒烟测试。
- ASR/LLM/TT S 模块代码位于 `modules/`，并含 demo 脚本用于本地验证。

## 快速开始（推荐）

1. 克隆仓库并进入项目：

   git clone <your-repo-url>
   cd AiVoiceAssistant

2. 一键部署（会创建 `.venv`、安装系统依赖、安装 Python 包、注入运行时库补丁并下载/安装模型）：

   make setup

3. 激活虚拟环境（非常重要）：

   source .venv/bin/activate

   激活后 `.venv/bin/activate` 会自动注入运行时环境：
   - 自动把项目根加入 `PYTHONPATH`，使 `python -m`/脚本能正确导入本地包；
   - 自动把 `libs/rknn` 加入 `LD_LIBRARY_PATH`；
   - 若存在 `libs/rknn/librknnrt.so`，会把该库通过 `LD_PRELOAD` 预加载，优先使用项目内的 RKNN 运行时，避免系统已安装不同版本导致的问题。

   注：如果手动创建虚拟环境，请使用仓库 `Makefile` 的 `create-venv` 或确保激活脚本含上述注入内容。

## 主要 Make 目标说明

- `make setup`：完整流程（init-uv -> create-venv -> deps -> toolkit -> link-driver -> download -> convert -> test）。
- `make create-venv`：仅创建 `.venv` 并注入运行时库补丁（在 `.venv/bin/activate` 末尾写入环境变量配置）。
- `make deps`：安装系统依赖（apt）及 Python 包（使用 uv 管理的 pip，锁定部分关键包版本以保证兼容性）。
- `make toolkit`：安装仓库内提供的 RKNN-Toolkit wheel（ARM64 cp310 版本）。
- `make link-driver`：将项目内 `libs/rknn/librknnrt.so` 链接到系统路径 `/usr/lib` 和 `/usr/lib64`（需要 sudo）。谨慎使用，通常只在确实需要系统级指向项目库时运行。
- `make download`：下载/准备模型数据。
- `make convert`：运行模型转换脚本（例如 zipformer 的 ONNX -> RKNN 等）。
- `make test` / `make test-hw`：运行硬件冒烟测试，验证 RKNN NPU、ASR 流程是否可用。
- `make clean`：删除 `.venv` 和缓存文件。

## 运行示例

- 本地运行 Zipformer demo（在激活虚拟环境后）：

  python modules/asr/zipformer.py \
    --encoder_model_path models/zipformer/encoder-epoch-99-avg-1.rknn \
    --decoder_model_path models/zipformer/decoder-epoch-99-avg-1.rknn \
    --joiner_model_path models/zipformer/joiner-epoch-99-avg-1.rknn

- 运行硬件测试（已通过 Makefile 封装）：

  make test-hw

## 常见问题与排查指南

1. make create-venv / make setup 报错：
   - 请确认 `uv` 可执行（`which uv`），`make init-uv` 会尝试安装；
   - 检查 Makefile 中对路径变量的写入是否被权限或文件锁阻止；
   - 确认 `bash` 环境下执行 `source .venv/bin/activate`。

2. RKNN 运行时或模型报错（例如符号找不到、版本不匹配）：
   - 激活虚拟环境后运行 `echo $LD_LIBRARY_PATH` 与 `echo $LD_PRELOAD`，确认是否指向 `$(pwd)/libs/rknn`；
   - 若项目注入未生效，可手动设置：
     export LD_LIBRARY_PATH="$(pwd)/libs/rknn:$LD_LIBRARY_PATH"
     export LD_PRELOAD="$(pwd)/libs/rknn/librknnrt.so:$LD_PRELOAD"
   - `make link-driver` 会把项目内 librknnrt.so 链接到系统目录（需 sudo），谨慎使用。

3. Git 推送失败（Permission denied (publickey)）：
   - 这是因为本机未配置 SSH 公钥或 GitHub 未登记你的公钥。两种解决方案：
     - 推荐：添加 SSH 公钥到 GitHub（ssh-keygen -> ssh-add -> 将 ~/.ssh/id_ed25519.pub 内容粘贴到 GitHub）。
     - 临时：改为 HTTPS 远程并使用 PAT（personal access token）：
       git remote set-url origin https://github.com/<user>/AiVoiceAssistant.git
       git push -u origin main

4. Python 包版本冲突（例如 numpy/torch）：
   - Makefile 在 `deps` 中会安装特定版本（如 numpy==1.26.4 torch==2.2.0）。请不要在系统环境或全局 pip 中安装冲突版本；建议始终使用 `.venv`。

## 开发者提示

- 激活脚本注入设计为幂等：重复创建虚拟环境或多次激活不会重复添加相同条目。
- 如果你希望在 CI 中复用该流程，可在 CI 脚本中直接 source `.venv/bin/activate` 或在 Job 环境中设置等效的环境变量。

## 贡献与联系

欢迎提交 issue/PR。请保持变更与依赖声明清晰，尤其是涉及 RKNN、torch、numpy 版本的更新。

---

(TODO) 若需要，我可以把 README 中的示例命令改为适配你当前的模型文件名或补充运行日志示例。

3588上git的代理使用的是本地的win主机  记得unset
