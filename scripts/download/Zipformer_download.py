import os
import requests
from dotenv import load_dotenv
from utils.paths import MODELS_DIR

# 加载环境变量
load_dotenv()

def get_proxies() -> dict[str, str]:
    """
    专门解决 Pylance 报错：
    Argument of type "dict[str, str | None]" cannot be assigned to parameter "proxies"
    """
    proxies = {}
    http_p = os.getenv('HTTP_PROXY')
    https_p = os.getenv('HTTPS_PROXY')
    
    # 只有值不为 None 时才加入字典
    if http_p:
        proxies['http'] = http_p
    if https_p:
        proxies['https'] = https_p
        
    return proxies

def download_file(url, save_path):
    if os.path.exists(save_path):
        print(f"⏩ 文件已存在，跳过: {save_path.name}")
        return

    print(f"📥 正在下载: {save_path.name}...")
    print(f"   🔗 源地址: {url}")
    
    # 使用修复后的代理获取函数
    proxies = get_proxies()
    if proxies:
        print(f"   🌐 使用代理: {proxies}")

    try:
        with requests.get(url, proxies=proxies, stream=True, timeout=30) as r:
            r.raise_for_status()
            with open(save_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"✅ 下载完成: {save_path.name}")
    except Exception as e:
        print(f"❌ 下载 {save_path.name} 失败: {e}")
        # 如果下载失败，删除可能损坏的文件
        if os.path.exists(save_path):
            os.remove(save_path)

def main():
    # 1. 设置目录
    zipformer_dir = MODELS_DIR / "zipformer"
    zipformer_dir.mkdir(parents=True, exist_ok=True)

    # 2. 定义下载任务列表
    tasks = []

    # --- 任务组 A: 你的 ONNX 模型 (保持原样，使用 ZBox 源) ---
    base_url_onnx = "https://ftrg.zbox.filez.com/v2/delivery/data/95f00b0fc900458ba134f8b180b3f7a1/examples/zipformer"
    onnx_files = [
        "encoder-epoch-99-avg-1.onnx",
        "decoder-epoch-99-avg-1.onnx",
        "joiner-epoch-99-avg-1.onnx"
    ]
    for name in onnx_files:
        tasks.append({
            "name": name,
            "url": f"{base_url_onnx}/{name}"
        })

    # --- 任务组 B: 词汇表 (新增，使用 GitHub 源) ---
    tasks.append({
        "name": "vocab.txt",
        "url": "https://raw.githubusercontent.com/airockchip/rknn_model_zoo/main/examples/zipformer/model/vocab.txt"
    })

    # 3. 执行下载
    print(f"🚀 开始下载 Zipformer 模型及资源 (共 {len(tasks)} 个文件)...")
    for task in tasks:
        save_path = zipformer_dir / task["name"]
        download_file(task["url"], save_path)

if __name__ == "__main__":
    main()