import traceback
from pathlib import Path

import kaldifeat
import numpy as np
import soundfile as sf
import torch
from rknn.api import RKNN

from utils.paths import get_model_path


def test_asr_hardware():
    """验证 ASR 硬件全链路：读取音频 -> kaldifeat 提取特征 -> RKNN 初始化"""
    print("🚀 开始 ASR 硬件链路验证 (使用真实音频文件)...")

    # 1. 获取项目根目录（基于当前文件位置推算，用于定位测试音频）
    # 虽然没有 sys.path 引导，但读取本地文件仍建议使用绝对路径防止报错
    project_root = Path(__file__).resolve().parents[2]
    test_wav_path = project_root / "tests" / "data" / "test.wav"

    if not test_wav_path.exists():
        print(f"❌ 找不到测试音频文件: {test_wav_path}")
        return

    # 2. 提取特征验证 (kaldifeat OnlineFbank)
    try:
        audio_data, sample_rate = sf.read(str(test_wav_path))
        
        # 预处理：单声道转 float32 Tensor
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)
        audio_tensor = torch.from_numpy(audio_data).to(torch.float32)

        # 初始化参数（严格对齐生产环境 zipformer.py）
        opts = kaldifeat.FbankOptions()
        opts.frame_opts.samp_freq = 16000
        opts.mel_opts.num_bins = 80
        fbank = kaldifeat.OnlineFbank(opts)

        # 在线提取特征逻辑
        fbank.accept_waveform(sampling_rate=sample_rate, waveform=audio_tensor)
        num_frames = fbank.num_frames_ready
        
        if num_frames > 0:
            print(f"✅ kaldifeat 验证通过。已从音频生成 {num_frames} 帧特征。")
        else:
            print("❌ kaldifeat 未能生成特征帧。")
            return
    except Exception:
        print("❌ kaldifeat 运行崩溃：")
        traceback.print_exc()
        return

    # 3. NPU 运行时验证 (RKNN)
    try:
        # 获取 Encoder 模型路径
        zipformer_dir = Path(get_model_path("zipformer"))
        encoder_path = zipformer_dir / "encoder-epoch-99-avg-1.rknn"

        if not encoder_path.exists():
            print(f"❌ 找不到 RKNN 模型文件: {encoder_path}")
            return

        rknn = RKNN(verbose=False)
        print(f"📦 正在加载 NPU 模型: {encoder_path.name}")

        if rknn.load_rknn(str(encoder_path)) != 0:
            print("❌ 模型加载失败")
            return

        print("⚙️  正在初始化 RK3588 NPU 运行时...")
        if rknn.init_runtime(target='rk3588') != 0:
            print("❌ NPU 初始化失败，请检查驱动权限 (ls /dev/rknn)")
            return

        print("✅ RKNN 运行时初始化成功")
        print("\n✨ [SUCCESS] ASR 硬件测试通过：音频读写、特征提取、NPU 调用均正常。")
        
        rknn.release()
    except Exception:
        print("❌ RKNN 验证过程崩溃：")
        traceback.print_exc()


if __name__ == "__main__":
    test_asr_hardware()