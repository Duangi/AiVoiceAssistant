# =================================================================
# AiVoiceAssistant 项目总控 Makefile (2025 标准版)
# =================================================================

# 1. 基础变量定义
PROJECT_ROOT := $(shell pwd)
VENV         := $(PROJECT_ROOT)/.venv
PYTHON       := $(VENV)/bin/python

# 自动寻找 uv：优先找系统路径，找不到找 ~/.local/bin，再找不到就准备安装
UV           := $(shell command -v uv 2> /dev/null || echo $(HOME)/.local/bin/uv)

# 库文件路径
RKNN_LIB_DIR	 := $(PROJECT_ROOT)/libs/rknn

.PHONY: help setup init-uv create-venv deps toolkit link-driver download convert clean test-hw test

# 默认目标：显示帮助
help:
	@echo "🌟 AiVoiceAssistant 项目自动化工具"
	@echo "------------------------------------------------"
	@echo "一键指令:"
	@echo "  make setup           - [最推荐] 从零开始完成所有部署(UV/环境/依赖/模型)"
	@echo ""
	@echo "分步指令:"
	@echo "  make init-uv         - 仅安装 uv 工具"
	@echo "  make create-venv     - 创建本地 .venv 环境"
	@echo "  make deps            - 安装系统依赖和 Python 包"
	@echo "  make toolkit         - 安装 RKNN-Toolkit2 (ARM64)"
	@echo "  make download        - 执行模型下载脚本"
	@echo "  make convert         - 执行 Zipformer 模型转换"
	@echo "  make clean           - 删除环境和缓存"
	@echo "------------------------------------------------"

# --- 核心目标：一键部署 ---
setup: init-uv create-venv deps toolkit link-driver download convert test
	@echo "🎉 [SUCCESS] 整个开发环境已就绪！"
	@echo "👉 请在 VS Code 中选择解释器: $(VENV)/bin/python"

# 1. 自动安装/检查 UV
init-uv:
	@if [ ! -x "$(UV)" ]; then \
		echo "🚚 未检测到 uv，正在安装..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
		export PATH="$(HOME)/.local/bin:$$PATH"; \
	else \
		echo "✅ uv 已就绪: $(UV)"; \
	fi

# 2. 创建本地虚拟环境
create-venv: init-uv
	@if [ ! -d "$(VENV)" ]; then \
		echo "🔨 正在创建本地虚拟环境 .venv ..."; \
		$(UV) venv --python 3.10; \
		echo "🩹 正在注入驱动路径补丁..."; \
		echo '' >> $(VENV)/bin/activate; \
		echo '# --- AiVoiceAssistant Environment Patch ---' >> $(VENV)/bin/activate; \
		# 使用 $(RKNN_LIB_DIR) 获取 Makefile 变量，使用 $$ 获取 Shell 变量 \
		echo 'export LD_LIBRARY_PATH="$(RKNN_LIB_DIR):$$LD_LIBRARY_PATH"' >> $(VENV)/bin/activate; \
		echo 'export PYTHONPATH="$(PROJECT_ROOT):$$PYTHONPATH"' >> $(VENV)/bin/activate; \
		echo '✅ 补丁注入完成。'; \
	else \
		echo "✅ 虚拟环境已存在."; \
	fi

# 3. 安装依赖 (系统库 + Python库)
deps: create-venv
	@echo "📦 安装系统级编译依赖..."
	sudo apt-get update && sudo apt-get install -y \
		python3-dev gcc g++ cmake libxslt1-dev zlib1g-dev \
		libglib2.0-0 libsm6 libgl1-mesa-glx libprotobuf-dev \
		libasound2-dev 
	
	@echo "🐍 正在预装符合 RKNN 要求的固定版本 (1.26.4 & 2.2.0)..."
	# 这里强制锁死版本，不准自动升级
	$(UV) pip install --python $(PYTHON) numpy==1.26.4 torch==2.2.0 setuptools wheel pip
	
	@echo "🐍 正在编译安装 kaldifeat (基于固定版本)..."
	@TORCH_DIR=$$($(PYTHON) -c "import torch; print(torch.utils.cmake_prefix_path)") && \
	 export KALDIFEAT_CMAKE_ARGS="-DCMAKE_CXX_STANDARD=17 -DTORCH_DIR=$$TORCH_DIR" && \
	 export KALDIFEAT_MAKE_ARGS="-j4" && \
	 $(UV) pip install --python $(PYTHON) --no-build-isolation kaldifeat
	
	@echo "🐍 安装剩余依赖..."
	$(UV) pip install --python $(PYTHON) -r requirements.txt

# 4. 专项安装 RKNN-Toolkit2 (ARM64)
toolkit: create-venv
	@echo "💾 正在安装项目内置的 RKNN-Toolkit2 (ARM64)..."
	@# 寻找 libs/rknn_packages 下符合 Python 3.10 和 aarch64 的 whl 文件
	@WHL_PATH=$$(ls $(RKNN_LIB_DIR)/rknn_toolkit2-*-cp310-cp310-*aarch64.whl 2>/dev/null | head -n 1); \
	if [ -f "$$WHL_PATH" ]; then \
		echo "📦 找到安装包: $$WHL_PATH"; \
		$(UV) pip install --python $(PYTHON) "$$WHL_PATH"; \
		echo "✅ 已成功安装 RKNN-Toolkit2 到 .venv"; \
	else \
		echo "❌ 错误: 在 libs/rknn_packages/ 下找不到符合条件的 .whl 文件！"; \
		echo "请确保文件存在且命名包含 'cp310' 和 'aarch64'"; \
		exit 1; \
	fi

link-driver:
	@echo "🔗 正在创建系统级驱动链接 (2.3.2)..."
	@# 备份原始文件
	@if [ ! -f "/usr/lib/librknnrt.so.original" ]; then \
		sudo mv /usr/lib/librknnrt.so /usr/lib/librknnrt.so.original; \
	fi
	@# 创建指向项目内部的软链接
	sudo ln -sf $(PROJECT_ROOT)/libs/rknn/librknnrt.so /usr/lib/librknnrt.so
	sudo ln -sf $(PROJECT_ROOT)/libs/rknn/librknnrt.so /usr/lib64/librknnrt.so
	sudo ldconfig
	@echo "✅ 链接创建完成，系统现已指向项目内的 2.3.2 驱动。"

# 5. 下载模型
# 使用 export PYTHONPATH 来确保脚本中的 from utils.paths 正确生效
download: create-venv
	@echo "📥 启动模型下载任务..."
	@export PYTHONPATH=$(PROJECT_ROOT) && $(PYTHON) scripts/download/Zipformer_download.py
	@export PYTHONPATH=$(PROJECT_ROOT) && $(PYTHON) scripts/download/Qwen2.5-3B-Instruct-RKNN3_download.py

# 6. 转换模型
convert: create-venv
	@echo "🔄 启动模型转换任务..."
	@export PYTHONPATH=$(PROJECT_ROOT) && $(PYTHON) scripts/convert/convert_zipformer.py

# 清理环境
clean:
	@echo "🧹 正在清理环境..."
	rm -rf $(VENV)
	find . -type d -name "__pycache__" -exec rm -rf {} +
	@echo "✨ 已清理所有临时文件和虚拟环境。"

# 运行硬件冒烟测试
test-hw:
	@echo "🔍 开始硬件链路验证..."
	@export PYTHONPATH=$(PROJECT_ROOT) && $(PYTHON) tests/hardware/test_asr.py
	@export PYTHONPATH=$(PROJECT_ROOT) && $(PYTHON) tests/hardware/test_npu.py

# 运行所有测试（未来可以引入 pytest）
test: test-hw
	@echo "✅ 所有测试已完成。"