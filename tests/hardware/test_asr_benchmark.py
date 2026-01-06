import os
import time
import torch
import soundfile as sf
import kaldifeat
from utils.paths import PROJECT_ROOT, MODELS_DIR

# 导入模块
from modules.asr.zipformer import RKNNModel, ensure_sample_rate, ensure_channels, read_vocab, post_process

# 路径配置
ZIPFORMER_DIR = MODELS_DIR / "zipformer"
ENCODER_PATH = str(ZIPFORMER_DIR / "encoder-epoch-99-avg-1.rknn")
DECODER_PATH = str(ZIPFORMER_DIR / "decoder-epoch-99-avg-1.rknn")
JOINER_PATH  = str(ZIPFORMER_DIR / "joiner-epoch-99-avg-1.rknn")
VOCAB_PATH   = str(ZIPFORMER_DIR / "vocab.txt")
TEST_WAV     = str(PROJECT_ROOT / "usb_mic_test.wav")

def benchmark_streaming():
    print("🚀 开始 Zipformer 性能基准测试 (Benchmark)...")
    print(f"📂 音频文件: {TEST_WAV}")

    # ------------------------------------------------------------------
    # 1. 测试模型加载时间
    # ------------------------------------------------------------------
    print("-" * 40)
    print("⏱️  正在加载模型...")
    t_load_start = time.time()

    model = RKNNModel(ENCODER_PATH, DECODER_PATH, JOINER_PATH, target="rk3588", device_id=None)
    model.init_encoder_input()
    
    t_load_end = time.time()
    load_time = (t_load_end - t_load_start) * 1000
    print(f"✅ 模型加载完成 | 耗时: {load_time:.2f} ms")

    try:
        # ------------------------------------------------------------------
        # 2. 准备音频和特征提取器
        # ------------------------------------------------------------------
        audio_data, sample_rate = sf.read(str(TEST_WAV))
        # 预处理
        if audio_data.ndim > 1: 
            audio_data, _ = ensure_channels(audio_data, audio_data.ndim)
        audio_data, sample_rate = ensure_sample_rate(audio_data, sample_rate)
        audio_tensor = torch.tensor(audio_data, dtype=torch.float32)

        # 计算音频总时长
        audio_duration_s = len(audio_data) / sample_rate
        print(f"🎵 音频时长: {audio_duration_s:.2f} 秒")

        # 初始化 Fbank (特征提取)
        opts = kaldifeat.FbankOptions()
        opts.frame_opts.samp_freq = 16000
        opts.mel_opts.num_bins = 80
        opts.mel_opts.high_freq = -400
        opts.frame_opts.dither = 0
        opts.frame_opts.snip_edges = False
        fbank = kaldifeat.OnlineFbank(opts)

        # ------------------------------------------------------------------
        # 3. 模拟流式推理 (核心循环)
        # ------------------------------------------------------------------
        print("-" * 40)
        print("🏎️  开始流式推理 (模拟真实数据流)...")
        
        # Zipformer 参数
        chunk_size = 103 # 每次送入 NPU 的帧数
        offset = 96      # 步长
        context_size = 2
        
        # 模拟“喂数据”
        fbank.accept_waveform(sampling_rate=16000, waveform=audio_tensor)
        fbank.input_finished()
        
        num_frames = fbank.num_frames_ready
        num_processed = 0
        
        # 解码状态
        hyp = None
        decoder_out = None
        timestamp = []
        frame_offset = 0
        
        # 统计变量
        chunk_count = 0
        total_inference_time = 0 # 纯 NPU + CPU 解码耗时
        
        t_pipeline_start = time.time()

        while num_frames - num_processed >= chunk_size:
            chunk_count += 1
            
            # A. 提取特征 (模拟前端耗时)
            frames = []
            for i in range(chunk_size):
                frames.append(fbank.get_frame(num_processed + i))
            frames = torch.cat(frames, dim=0).unsqueeze(0)
            
            # B. 执行推理 (计时核心)
            t_chunk_start = time.time()
            
            hyp, decoder_out, timestamp, frame_offset = model.run_greedy_search(
                frames, context_size, decoder_out, hyp, num_processed, timestamp, frame_offset
            )
            
            t_chunk_end = time.time()
            chunk_cost = (t_chunk_end - t_chunk_start) * 1000 # ms
            total_inference_time += (t_chunk_end - t_chunk_start)
            
            # 打印前几个 Chunk 的耗时，后面的省略防止刷屏
            if chunk_count <= 5:
                print(f"   👉 Chunk {chunk_count}: 耗时 {chunk_cost:.2f} ms")
            elif chunk_count == 6:
                print("   ... (后续 Chunk 省略日志)")

            num_processed += offset

        t_pipeline_end = time.time()
        
        # ------------------------------------------------------------------
        # 4. 结果分析报告
        # ------------------------------------------------------------------
        print("-" * 40)
        print("📊 性能分析报告")
        print("-" * 40)
        
        total_wall_time = t_pipeline_end - t_pipeline_start
        avg_chunk_time = (total_inference_time / chunk_count) * 1000 if chunk_count > 0 else 0
        
        # 计算 RTF (Real Time Factor)
        # RTF = 处理耗时 / 音频时长
        # RTF < 1 表示实时，越小越快
        rtf = total_wall_time / audio_duration_s
        
        print(f"✅ 处理 Chunk 总数 : {chunk_count}")
        print(f"⚡ 平均单次推理耗时: {avg_chunk_time:.2f} ms")
        print(f"⏳ 总流程物理耗时   : {total_wall_time:.4f} 秒")
        print(f"🎵 原始音频时长     : {audio_duration_s:.4f} 秒")
        print(f"🚀 实时率 (RTF)     : {rtf:.4f}")
        
        if rtf < 1.0:
            print(f"🌟 结论: 达标！处理速度比说话快 {1/rtf:.1f} 倍")
        else:
            print("❌ 结论: 不达标，无法满足实时性")
            
        # 输出识别结果
        if os.path.exists(str(VOCAB_PATH)):
            vocab = read_vocab(str(VOCAB_PATH))
            text, _ = post_process(hyp, vocab, timestamp)
            print(f"\n📝 识别内容: {text}")

    finally:
        model.release_model()

if __name__ == "__main__":
    benchmark_streaming()