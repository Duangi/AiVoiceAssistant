from modelscope import snapshot_download
from utils.paths import get_model_path  # 直接导入！不再需要手动找根目录

def main():

    save_path = get_model_path('Qwen2.5-3B-Instruct-RKNN3')

    print(f"🚀 正在下载模型到: {save_path}")

    snapshot_download(
        model_id='radxa/Qwen2.5-3B-Instruct-RKNN3',
        local_dir=save_path
    )

if __name__ == "__main__":
    main()