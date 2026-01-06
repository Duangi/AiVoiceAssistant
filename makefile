# =================================================================
# AiVoiceAssistant 项目总控 Makefile (2026 硬件全加速版)
# =================================================================

PROJECT_ROOT := $(shell pwd)

# --- 环境定义 ---
VENV_BUILD   := $(PROJECT_ROOT)/.venv-build
VENV_RUN     := $(PROJECT_ROOT)/.venv-runtime

# --- 解释器 ---
PY_BUILD     := $(VENV_BUILD)/bin/python
PY_RUN       := $(VENV_RUN)/bin/python

UV           := $(shell command -v uv 2> /dev/null || echo $(HOME)/.local/bin/uv)
RKNN_LIB_DIR := $(PROJECT_ROOT)/libs/rknn

# 自动寻找 WHL
TOOLKIT2_WHL      := $(shell ls $(RKNN_LIB_DIR)/rknn_toolkit2-*-cp310-cp310-*aarch64.whl 2>/dev/null | head -n 1)
TOOLKIT_LITE2_WHL := $(shell ls $(RKNN_LIB_DIR)/rknn_toolkit_lite2-*-cp310-cp310-*aarch64.whl 2>/dev/null | head -n 1)

# LLM 相关
LLM_MODEL      := $(PROJECT_ROOT)/models/Qwen3-VL-4B/qwen3-vl-4b_vision_rk3588.rknn
LLM_SERVER_DIR := $(PROJECT_ROOT)/modules/llm/server
FIX_FREQ_SH    := $(PROJECT_ROOT)/scripts/hardware/fix_freq_rk3588.sh

.PHONY: help setup init-uv sys-deps fix-system-lib create-venv-build create-venv-run download convert clean test-hw run-llm-web

# -----------------------------------------------------------------
# 核心流程
# -----------------------------------------------------------------
setup: init-uv sys-deps fix-system-lib create-venv-build create-venv-run download convert test-hw
	@echo "🎉 [SUCCESS] 部署完成！"
	@echo "👉 开发/推理请使用: $(VENV_RUN)/bin/python"

init-uv:
	@if [ ! -x "$(UV)" ]; then curl -LsSf https://astral.sh/uv/install.sh | sh; fi

# 2. 安装系统依赖 (新增 OpenMP 和 OpenCL 库)
sys-deps:
	@echo "📦 正在安装系统级依赖与硬件加速库..."
	sudo apt-get update && sudo apt-get install -y \
		python3-dev gcc g++ cmake libxslt1-dev zlib1g-dev \
		libglib2.0-0 libsm6 libgl1-mesa-glx libprotobuf-dev \
		libasound2-dev \
		libomp-dev ocl-icd-libopencl1 opencl-headers ocl-icd-opencl-dev

# 3. 💥 [核心] 实体替换系统驱动 + GPU 映射 💥
fix-system-lib:
	@echo "☢️  正在检查并实体替换系统驱动 (需 sudo)..."
	@if [ ! -f "$(RKNN_LIB_DIR)/librknnrt.so" ]; then echo "❌ 缺少 libs/rknn/librknnrt.so"; exit 1; fi
	
	@# A. 处理 ASR 驱动 (librknnrt.so)
	@if ! cmp -s "$(RKNN_LIB_DIR)/librknnrt.so" "/usr/lib/librknnrt.so"; then \
		echo "   🔧 物理替换 librknnrt.so (ASR)..."; \
		if [ -f "/usr/lib/librknnrt.so" ]; then sudo mv /usr/lib/librknnrt.so /usr/lib/librknnrt.so.bak; fi; \
		sudo cp -f "$(RKNN_LIB_DIR)/librknnrt.so" /usr/lib/librknnrt.so; \
	fi

	@# B. 处理 LLM 驱动 (librkllmrt.so)
	@if [ -f "$(RKNN_LIB_DIR)/librkllmrt.so" ]; then \
		if ! cmp -s "$(RKNN_LIB_DIR)/librkllmrt.so" "/usr/lib/librkllmrt.so"; then \
			echo "   🔧 物理替换 librkllmrt.so (LLM)..."; \
			if [ -f "/usr/lib/librkllmrt.so" ]; then sudo mv /usr/lib/librkllmrt.so /usr/lib/librkllmrt.so.llm.bak; fi; \
			sudo cp -f "$(RKNN_LIB_DIR)/librkllmrt.so" /usr/lib/librkllmrt.so; \
		fi; \
	fi

	@# C. 映射 Mali GPU 驱动到 OpenCL 标准路径 (解决 "Cannot load clGetPlatformIDs" 问题)
	@echo "   🔗 映射 Mali GPU 驱动到 OpenCL 路径..."
	@sudo ln -sf /usr/lib/aarch64-linux-gnu/libmali-valhall-g610-g13p0-x11-gbm.so /usr/lib/libOpenCL.so
	@sudo ln -sf /usr/lib/aarch64-linux-gnu/libmali-valhall-g610-g13p0-x11-gbm.so /usr/lib/libOpenCL.so.1
	
	@# D. 刷新缓存
	@sudo ldconfig
	@echo "✅ 系统驱动与硬件加速环境配置完毕。"

# 4. 构建环境
create-venv-build: init-uv
	@if [ ! -d "$(VENV_BUILD)" ]; then \
		echo "🏗️  创建构建环境..."; \
		$(UV) venv $(VENV_BUILD) --python 3.10; \
		$(UV) pip install --python $(PY_BUILD) "$(TOOLKIT2_WHL)" onnx setuptools; \
	fi

# 5. 运行环境 (请确保 requirements.txt 中已包含 flask==2.2.2 和 werkzeug==2.2.2)
create-venv-run: init-uv
	@if [ ! -d "$(VENV_RUN)" ]; then \
		echo "🚀 创建运行环境..."; \
		$(UV) venv $(VENV_RUN) --python 3.10; \
		$(UV) pip install --python $(PY_RUN) "$(TOOLKIT_LITE2_WHL)" numpy==1.26.4 torch==2.2.0; \
		TORCH_DIR=$$($(PY_RUN) -c "import torch; print(torch.utils.cmake_prefix_path)") && \
		export KALDIFEAT_CMAKE_ARGS="-DCMAKE_CXX_STANDARD=17 -DTORCH_DIR=$$TORCH_DIR" && \
		$(UV) pip install --python $(PY_RUN) --no-build-isolation kaldifeat; \
		$(UV) pip install --python $(PY_RUN) -r requirements.txt; \
	fi

# 6. 下载
download: create-venv-run
	@echo "📥 下载模型..."
	@export PYTHONPATH=$(PROJECT_ROOT) && $(PY_RUN) scripts/download/Zipformer_download.py
	@export PYTHONPATH=$(PROJECT_ROOT) && $(PY_RUN) scripts/download/Qwen2.5-3B-Instruct-RKNN3_download.py

# 7. 转换
convert: create-venv-build
	@echo "🔄 转换模型..."
	@export PYTHONPATH=$(PROJECT_ROOT) && $(PY_BUILD) scripts/convert/convert_zipformer.py

# 8. 测试
test-hw: create-venv-run
	@echo "🔍 硬件验证..."
	@export PYTHONPATH=$(PROJECT_ROOT) && $(PY_RUN) tests/hardware/test_asr.py
	@export PYTHONPATH=$(PROJECT_ROOT) && $(PY_RUN) tests/hardware/test_npu.py

test: test-hw

clean:
	rm -rf $(VENV_BUILD) $(VENV_RUN)

# 启动 LLM Web
run-llm-web: create-venv-run
	@echo "⚡ 正在优化硬件频率..."
	@sudo bash $(FIX_FREQ_SH)
	
	@echo "🔗 准备 Server 内部库与脚本环境..."
	@mkdir -p $(LLM_SERVER_DIR)/lib
	@# 指向 /usr/lib 下已经替换好的 2.3.2/1.2.3 驱动
	@ln -sf /usr/lib/librkllmrt.so $(LLM_SERVER_DIR)/lib/librkllmrt.so
	@# 将定频脚本同步到当前目录，防止 python 内部 subprocess 找不到
	@cp -f $(FIX_FREQ_SH) $(LLM_SERVER_DIR)/
	
	@echo "🚀 启动 LLM Gradio Server (端口 8080)..."
	@cd $(LLM_SERVER_DIR) && \
	 export PYTHONPATH=$(PROJECT_ROOT):$$PYTHONPATH && \
	 $(PY_RUN) gradio_server.py \
		--rkllm_model_path $(LLM_MODEL) \
		--target_platform rk3588