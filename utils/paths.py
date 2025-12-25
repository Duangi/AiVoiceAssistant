from pathlib import Path

# 获取当前文件 (paths.py) 的绝对路径
_current_file = Path(__file__).resolve()

# 逻辑：paths.py 在 utils 文件夹里，所以上一级就是根目录
PROJECT_ROOT = _current_file.parents[1]

# 统一导出常用的路径常量
MODELS_DIR = PROJECT_ROOT / "models"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
DATA_DIR = PROJECT_ROOT / "data"

# 顺便确保必要的文件夹已经存在
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def get_model_path(model_name: str) -> str:
    """获取具体模型的绝对路径字符串"""
    return str(MODELS_DIR / model_name)