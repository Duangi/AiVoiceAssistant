import os
import requests
from pathlib import Path
from dotenv import load_dotenv  # 新增导入
from utils.paths import MODELS_DIR

# 1. 在脚本运行之初，加载 .env 中的环境变量
# 它会自动寻找根目录下的 .env 文件
load_dotenv()

def download_file(url, save_path):
    if os.path.exists(save_path):
        print(f"⏩ 文件已存在，跳过: {save_path.name}")
        return

    print(f"📥 正在下载: {save_path.name}...")
    
    # 2. 直接从环境变量读取代理
    # 如果 .env 里没写，这里就是 None，请求会直连，代码健壮性更好
    proxies = {
        'http': os.getenv('HTTP_PROXY'),
        'https': os.getenv('HTTPS_PROXY')
    }

    try:
        # 即使 proxies 里的值是空的，requests 也能正常处理
        with requests.get(url, proxies=proxies, stream=True, timeout=30) as r:
            r.raise_for_status()
            with open(save_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"✅ 下载完成: {save_path.name}")
    except Exception as e:
        print(f"❌ 下载 {save_path.name} 失败: {e}")

def main():
    zipformer_dir = MODELS_DIR / "zipformer"
    zipformer_dir.mkdir(parents=True, exist_ok=True)

    base_url = "https://ftrg.zbox.filez.com/v2/delivery/data/95f00b0fc900458ba134f8b180b3f7a1/examples/zipformer"
    files = [
        "encoder-epoch-99-avg-1.onnx",
        "decoder-epoch-99-avg-1.onnx",
        "joiner-epoch-99-avg-1.onnx"
    ]

    for file_name in files:
        url = f"{base_url}/{file_name}"
        save_path = zipformer_dir / file_name
        download_file(url, save_path)

if __name__ == "__main__":
    main()