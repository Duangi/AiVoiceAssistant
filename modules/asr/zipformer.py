from rknnlite.api import RKNNLite
import torch
import kaldifeat
import soundfile as sf
import numpy as np
import argparse
import scipy
import sys
from typing import Any, Optional, Tuple, List

class BaseModel():
    def __init__(self):
        self.model_config = {'x': [1, 103, 80], 'cached_len_0': [2, 1], 'cached_len_1': [2, 1], 'cached_len_2': [2, 1], 'cached_len_3': [2, 1],
                             'cached_len_4': [2, 1], 'cached_avg_0': [2, 1, 256], 'cached_avg_1': [2, 1, 256], 'cached_avg_2': [2, 1, 256],
                             'cached_avg_3': [2, 1, 256], 'cached_avg_4': [2, 1, 256], 'cached_key_0': [2, 192, 1, 192], 'cached_key_1': [2, 96, 1, 192],
                             'cached_key_2': [2, 48, 1, 192], 'cached_key_3': [2, 24, 1, 192], 'cached_key_4': [2, 96, 1, 192], 'cached_val_0': [2, 192, 1, 96],
                             'cached_val_1': [2, 96, 1, 96], 'cached_val_2': [2, 48, 1, 96], 'cached_val_3': [2, 24, 1, 96], 'cached_val_4': [2, 96, 1, 96],
                             'cached_val2_0': [2, 192, 1, 96], 'cached_val2_1': [2, 96, 1, 96], 'cached_val2_2': [2, 48, 1, 96], 'cached_val2_3': [2, 24, 1, 96],
                             'cached_val2_4': [2, 96, 1, 96], 'cached_conv1_0': [2, 1, 256, 30], 'cached_conv1_1': [2, 1, 256, 30], 'cached_conv1_2': [2, 1, 256, 30],
                             'cached_conv1_3': [2, 1, 256, 30], 'cached_conv1_4': [2, 1, 256, 30], 'cached_conv2_0': [2, 1, 256, 30], 'cached_conv2_1': [2, 1, 256, 30],
                             'cached_conv2_2': [2, 1, 256, 30], 'cached_conv2_3': [2, 1, 256, 30], 'cached_conv2_4': [2, 1, 256, 30]}

    def init_encoder_input(self):
        self.encoder_input = []
        self.encoder_input_dict = {}
        for input_name in self.model_config:
            if 'cached_len' in input_name:
                data = np.zeros((self.model_config[input_name]), dtype=np.int64)
                self.encoder_input.append(data)
                self.encoder_input_dict[input_name] = data
            else:
                data = np.zeros((self.model_config[input_name]), dtype=np.float32)
                self.encoder_input.append(data)
                self.encoder_input_dict[input_name] = data
    
    def update_encoder_input(self, out, model_type):
        for idx, input_name in enumerate(self.encoder_input_dict):
            if idx == 0:
                continue
            if idx > 10 and model_type == 'rknn':
                data = self.convert_nchw_to_nhwc(out[idx])
            else:
                data = out[idx]
            self.encoder_input[idx] = data
            self.encoder_input_dict[input_name] = data
    
    def convert_nchw_to_nhwc(self, src):
        dst = np.transpose(src, (0, 2, 3, 1))
        return dst

    def init_model(self, model_path, target, device_id):
        pass
    
    def release_model(self):
        pass

    def run_encoder(self, x):
        pass
    
    def run_decoder(self, decoder_input):
        pass
    
    def run_joiner(self, encoder_out, decoder_out):
        pass
    
    def run_greedy_search(self, frames, context_size, decoder_out, hyp, num_processed_frames, timestamp, frame_offset):
        encoder_out = self.run_encoder(frames)
        if encoder_out is None:
            raise RuntimeError("Encoder returned None")
        encoder_out = encoder_out.squeeze(0)

        blank_id = 0
        unk_id = 2
        if decoder_out is None and hyp is None:
            hyp = [blank_id] * context_size
            decoder_input = np.array([hyp], dtype=np.int64)
            decoder_out = self.run_decoder(decoder_input)

        T = int(encoder_out.shape[0])
        for t in range(T):
            cur_encoder_out = encoder_out[t: t + 1]
            joiner_result = self.run_joiner(cur_encoder_out, decoder_out)
            if joiner_result is None:
                raise RuntimeError("Joiner returned None")
            joiner_out = joiner_result.squeeze(0)
            y = np.argmax(joiner_out, axis=0)
            if y != blank_id and y != unk_id:
                timestamp.append(frame_offset + t)
                hyp.append(y)
                decoder_input = hyp[-context_size:]
                decoder_input = np.array([decoder_input], dtype=np.int64)
                decoder_out = self.run_decoder(decoder_input)
        frame_offset += T

        return hyp, decoder_out, timestamp, frame_offset

class RKNNModel(BaseModel):
    def __init__(
        self,
        encoder_model_path: str,
        decoder_model_path: str,
        joiner_model_path: str,
        target: str,
        device_id: Optional[str]
    ):
        super().__init__()

        # 明确类型，避免静态分析将属性视为 None
        self.encoder: Any = None
        self.decoder: Any = None
        self.joiner: Any = None

        self.encoder = self.init_model(encoder_model_path, target, device_id)
        self.decoder = self.init_model(decoder_model_path, target, device_id)
        self.joiner = self.init_model(joiner_model_path, target, device_id)

    def init_model(self, model_path: str, target: str, device_id: Optional[str]) -> Any:
        # Create RKNN object
        rknn = RKNNLite(verbose=False)

        # Load RKNN model
        print('--> Loading model')
        ret = rknn.load_rknn(model_path)
        if ret != 0:
            print('Load RKNN model "{}" failed!'.format(model_path))
            sys.exit(ret)
        print('done')

        # init runtime environment
        print('--> Init runtime environment')
        ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
        if ret != 0:
            print('Init runtime environment failed')
            sys.exit(1)

        return rknn
    
    def release_model(self) -> None:
        if self.encoder is not None:
            try:
                self.encoder.release()
            except Exception:
                pass
            self.encoder = None
        if self.decoder is not None:
            try:
                self.decoder.release()
            except Exception:
                pass
            self.decoder = None
        if self.joiner is not None:
            try:
                self.joiner.release()
            except Exception:
                pass
            self.joiner = None

    def run_encoder(self, x: torch.Tensor) -> Optional[np.ndarray]:
        if self.encoder is None:
            raise RuntimeError("Encoder not initialized")
        np_x = x.numpy()
        self.encoder_input[0] = np_x
        self.encoder_input_dict['x'] = np_x
        out = self.encoder.inference(inputs=self.encoder_input)
        if out is None:
            return None
        self.update_encoder_input(out, 'rknn')
        return out[0]

    def run_decoder(self, decoder_input: np.ndarray) -> Optional[np.ndarray]:
        if self.decoder is None:
            raise RuntimeError("Decoder not initialized")
        out = self.decoder.inference(inputs=[decoder_input])
        if out is None:
            return None
        return out[0]

    def run_joiner(self, encoder_out: np.ndarray, decoder_out: np.ndarray) -> Optional[np.ndarray]:
        if self.joiner is None:
            raise RuntimeError("Joiner not initialized")
        out = self.joiner.inference(inputs=[encoder_out, decoder_out])
        if out is None:
            return None
        return out[0]

def read_vocab(tokens_file: str) -> dict:
    with open(tokens_file, 'r') as f:
        vocab = {}
        for line in f:
            parts = line.strip().split(None, 1)
            if len(parts) == 0:
                continue
            if len(parts) == 1:
                key = parts[0]
                value = ""
            else:
                value, key = parts[0], parts[1]
            vocab[key] = value
    return vocab

def set_model(args) -> BaseModel:
    # 仅支持 .rknn 三个文件同时存在的情况
    if args.encoder_model_path.endswith(".rknn") and args.decoder_model_path.endswith(".rknn") and args.joiner_model_path.endswith(".rknn"):
        model = RKNNModel(args.encoder_model_path, args.decoder_model_path, args.joiner_model_path,
                          target=args.target, device_id=args.device_id)
        return model
    else:
        print("Unsupported model files. Expect all three files to be .rknn")
        sys.exit(2)

def run_model(model: BaseModel, audio_data: torch.Tensor, sample_rate: int) -> Tuple[List[int], list]:
    # Set kaldifeat config
    opts = kaldifeat.FbankOptions()
    opts.frame_opts.samp_freq = 16000 # sample_rate=16000
    opts.mel_opts.num_bins = 80
    opts.mel_opts.high_freq = -400
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = False
    fbank = kaldifeat.OnlineFbank(opts)

    # Inference
    num_processed_frames = 0
    segment = 103
    offset = 96
    context_size = 2
    hyp: Optional[List[int]] = None
    decoder_out = None

    fbank.accept_waveform(sampling_rate=sample_rate, waveform=audio_data)
    num_frames = fbank.num_frames_ready
    timestamp: List[int] = []
    frame_offset = 0

    while num_frames - num_processed_frames > 0:
        if (num_frames - num_processed_frames) < segment:
            tail_padding_len = (segment - (num_frames - num_processed_frames)) / 100.0
            tail_padding = torch.zeros(int(tail_padding_len * sample_rate), dtype=torch.float32)
            fbank.accept_waveform(sampling_rate=sample_rate, waveform=tail_padding)
        frames = []
        for i in range(segment):
            frames.append(fbank.get_frame(num_processed_frames + i))

        frames = torch.cat(frames, dim=0)
        frames = frames.unsqueeze(0)
        hyp, decoder_out, timestamp, frame_offset = model.run_greedy_search(frames, context_size, decoder_out, hyp, num_processed_frames, timestamp, frame_offset)
        num_processed_frames += offset

    if hyp is None:
        return [], timestamp

    return hyp[context_size:], timestamp

def post_process(hyp, vocab, timestamp):
    text = ""
    for i in hyp:
        text += vocab.get(str(i), "")
    text = text.replace("▁", " ").strip()

    frame_shift_ms = 10
    subsampling_factor = 4
    frame_shift_s = frame_shift_ms / 1000.0 * subsampling_factor
    real_timestamp = [round(frame_shift_s*t, 2) for t in timestamp]
    return text, real_timestamp

def ensure_sample_rate(waveform, original_sample_rate, desired_sample_rate=16000):
    if original_sample_rate != desired_sample_rate:
        print("resample_audio: {} HZ -> {} HZ".format(original_sample_rate, desired_sample_rate))
        desired_length = int(round(float(len(waveform)) / original_sample_rate * desired_sample_rate))
        waveform = scipy.signal.resample(waveform, desired_length)
    return waveform, desired_sample_rate

def ensure_channels(waveform, original_channels, desired_channels=1):
    if original_channels != desired_channels:
        print("convert_channels: {} -> {}".format(original_channels, desired_channels))
        waveform = np.mean(waveform, axis=1)
    return waveform, desired_channels

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Zipformer Python Demo")
    # basic params
    parser.add_argument("--encoder_model_path", type=str, required=True, help="encoder model path, could be .onnx or .rknn file")
    parser.add_argument("--decoder_model_path", type=str, required=True, help="decoder model path, could be .onnx or .rknn file")
    parser.add_argument("--joiner_model_path", type=str, required=True, help="joiner model path, could be .onnx or .rknn file")
    parser.add_argument("--target", type=str, default="rk3588", help="target RKNPU platform")
    parser.add_argument("--device_id", type=str, default=None, help="device id")
    parser.add_argument("--vocab", type=str, default="../model/vocab.txt", help="vocab/tokens file")
    parser.add_argument("--audio", type=str, default="../data/test.wav", help="audio file to run")
    args = parser.parse_args()

    # Set inputs
    vocab = read_vocab(args.vocab)
    audio_data, sample_rate = sf.read(args.audio)
    channels = audio_data.ndim
    audio_data, channels = ensure_channels(audio_data, channels)
    audio_data, sample_rate = ensure_sample_rate(audio_data, sample_rate)
    audio_data = torch.tensor(audio_data, dtype=torch.float32)

    # Set model
    model = set_model(args)
    model.init_encoder_input()

    # Run model
    hyp, timestamp = run_model(model, audio_data, sample_rate)

    # Post process
    text, timestamp = post_process(hyp, vocab, timestamp)
    print("\nTimestamp (s):", timestamp)
    print("\nZipformer output:", text)

    # Release model
    model.release_model()