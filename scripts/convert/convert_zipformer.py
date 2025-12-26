import sys
import os
from pathlib import Path
from utils.paths import MODELS_DIR

# 导入 RKNN API (前提是你已经运行过 make toolkit)
try:
    from rknn.api import RKNN
except ImportError:
    print("❌ 错误: 找不到 rknn-toolkit2 模块，请先确保执行过 'make toolkit'")
    sys.exit(1)

# 环境引导：确保能找到 utils
_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.argv:
    sys.path.append(str(_root))

def convert_model(onnx_path: str, platform='rk3588'):
    """执行单个模型的转换逻辑"""
    rknn_path = onnx_path.replace('.onnx', '.rknn')
    
    if os.path.exists(rknn_path):
        print(f"⏩ {Path(rknn_path).name} 已存在，跳过。")
        return

    print(f"\n" + "="*50)
    print(f"🔄 正在转换: {Path(onnx_path).name}")
    print(f"平台: {platform}")
    print("="*50)

    rknn = RKNN(verbose=False)

    # 1. 配置参数 (Zipformer 官方 Demo 建议不开启量化，即 do_quant=False)
    rknn.config(target_platform=platform)

    # 2. 加载 ONNX
    print('--> 加载 ONNX 模型...')
    ret = rknn.load_onnx(model=onnx_path)
    if ret != 0:
        print('❌ 加载模型失败！')
        return

    # 3. 构建模型 (do_quantization=False)
    print('--> 构建 RKNN 模型 (此过程可能较慢)...')
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        print('❌ 构建模型失败！')
        return

    # 4. 导出 RKNN
    print(f'--> 导出到: {rknn_path}')
    ret = rknn.export_rknn(rknn_path)
    if ret != 0:
        print('❌ 导出模型失败！')
        return

    rknn.release()
    print(f"✨ {Path(onnx_path).name} 转换完成！")

def main():
    # 自动定位到 models/zipformer 目录
    zipformer_dir = MODELS_DIR / "zipformer"
    
    # 定义 Zipformer 必须的三个子模型
    onnx_files = [
        "encoder-epoch-99-avg-1.onnx",
        "decoder-epoch-99-avg-1.onnx",
        "joiner-epoch-99-avg-1.onnx"
    ]

    found_any = False
    for filename in onnx_files:
        full_path = zipformer_dir / filename
        if full_path.exists():
            convert_model(str(full_path))
            found_any = True
        else:
            print(f"⚠️ 找不到文件: {full_path}")

    if not found_any:
        print("❌ 错误: 在 models/zipformer 中没有找到任何可转换的 ONNX 文件！")
        print("请检查是否运行过 'make download'")
        sys.exit(1)

if __name__ == "__main__":
    main()