import torch
import soundfile as sf
from utils.paths import PROJECT_ROOT, MODELS_DIR

# 导入你刚刚写好的模块
from modules.asr.zipformer import RKNNModel, read_vocab, run_model, post_process

def test_final():
    # 路径
    encoder = str(MODELS_DIR / "zipformer/encoder-epoch-99-avg-1.rknn")
    decoder = str(MODELS_DIR / "zipformer/decoder-epoch-99-avg-1.rknn")
    joiner  = str(MODELS_DIR / "zipformer/joiner-epoch-99-avg-1.rknn")
    vocab_path = str(MODELS_DIR / "zipformer/vocab.txt")
    wav_path = str(PROJECT_ROOT / "tests/data/test.wav")

    print("🚀 加载 Zipformer 模型...")
    # 这里的 target 和 device_id 已经没用了，传空即可
    model = RKNNModel(encoder, decoder, joiner, target="rk3588", device_id=None)
    model.init_encoder_input()

    print("🎵 读取音频...")
    audio, sr = sf.read(wav_path)
    # 简单的预处理
    if audio.ndim > 1: 
        audio = audio.mean(axis=1) # 转单声道
    # 这里应该调用 ensure_sample_rate，为演示从简
    audio_tensor = torch.tensor(audio, dtype=torch.float32)

    print("🏎️  开始推理...")
    hyp, timestamp = run_model(model, audio_tensor, sr)

    vocab = read_vocab(vocab_path)
    text, _ = post_process(hyp, vocab, timestamp)
    
    print(f"\n📝 识别结果: {text}")
    model.release_model()

if __name__ == "__main__":
    test_final()