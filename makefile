# =================================================================
# AiVoiceAssistant 项目总控 Makefile (2025 实体替换版)
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

.PHONY: help setup init-uv sys-deps fix-system-lib create-venv-build create-venv-run download convert clean test-hw

# -----------------------------------------------------------------
# 核心流程
# -----------------------------------------------------------------
setup: init-uv sys-deps fix-system-lib create-venv-build create-venv-run download convert test-hw
	@echo "🎉 [SUCCESS] 部署完成！"
	@echo "👉 开发/推理请使用: $(VENV_RUN)/bin/python"

init-uv:
	@if [ ! -x "$(UV)" ]; then curl -LsSf https://astral.sh/uv/install.sh | sh; fi

sys-deps:
	@dpkg -s python3-dev >/dev/null 2>&1 || (sudo apt-get update && sudo apt-get install -y python3-dev gcc g++ cmake libxslt1-dev zlib1g-dev libglib2.0-0 libsm6 libgl1-mesa-glx libprotobuf-dev libasound2-dev)

# 3. 💥 [核心] 实体替换系统驱动 💥
# 不使用软链接，直接物理复制文件，确保万无一失
fix-system-lib:
	@echo "☢️  检查并实体替换系统驱动 (需 sudo)..."
	@if [ ! -f "$(RKNN_LIB_DIR)/librknnrt.so" ]; then echo "❌ 缺少 libs/rknn/librknnrt.so"; exit 1; fi
	
	@# 1. 处理 /usr/lib
	@if ! cmp -s "$(RKNN_LIB_DIR)/librknnrt.so" "/usr/lib/librknnrt.so"; then \
		echo "   🔧 检测到驱动不一致，正在执行物理替换..."; \
		if [ -f "/usr/lib/librknnrt.so" ]; then \
			echo "   💾 备份原驱动 -> /usr/lib/librknnrt.so.bak"; \
			sudo mv /usr/lib/librknnrt.so /usr/lib/librknnrt.so.bak; \
		fi; \
		echo "   📝 复制: libs/rknn/librknnrt.so -> /usr/lib/"; \
		sudo cp -f "$(RKNN_LIB_DIR)/librknnrt.so" /usr/lib/librknnrt.so; \
	else \
		echo "   ✅ /usr/lib 驱动已是最新，无需替换。"; \
	fi

	@# 2. 处理 /usr/lib64 (如果目录存在)
	@if [ -d "/usr/lib64" ]; then \
		if ! cmp -s "$(RKNN_LIB_DIR)/librknnrt.so" "/usr/lib64/librknnrt.so"; then \
			echo "   🔧 同步更新 /usr/lib64..."; \
			if [ -f "/usr/lib64/librknnrt.so" ]; then sudo mv /usr/lib64/librknnrt.so /usr/lib64/librknnrt.so.bak; fi; \
			sudo cp -f "$(RKNN_LIB_DIR)/librknnrt.so" /usr/lib64/librknnrt.so; \
		fi; \
	fi
	
	@# 3. 刷新缓存
	@sudo ldconfig
	@echo "✅ 系统驱动替换完毕。"

# 4. 构建环境 (转换用)
create-venv-build: init-uv
	@if [ ! -d "$(VENV_BUILD)" ]; then \
		echo "🏗️  创建构建环境..."; \
		$(UV) venv $(VENV_BUILD) --python 3.10; \
		$(UV) pip install --python $(PY_BUILD) "$(TOOLKIT2_WHL)" onnx setuptools; \
	fi

# 5. 运行环境 (推理用)
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

# 全部测试
test: test-hw

clean:
	rm -rf $(VENV_BUILD) $(VENV_RUN)