import numpy as np
import librosa
import soundfile as sf
from rich.progress import track
import argparse
import os

def clipping(speech_sample, min_quantile: float = 0.0, max_quantile: float = 0.9):
    """Apply the clipping distortion to the input signal.

    Args:
        speech_sample (np.ndarray): a single speech sample (1, Time)
        min_quantile (float): lower bound on the quantile of samples to be clipped
        max_quantile (float): upper bound on the quantile of samples to be clipped

    Returns:
        ret (np.ndarray): clipped speech sample (1, Time)
    """
    q = np.array([min_quantile, max_quantile])
    min_, max_ = np.quantile(speech_sample, q, axis=-1, keepdims=False)
    # per-channel clipping
    ret = np.stack(
        [
            np.clip(speech_sample[i], min_[i], max_[i])
            for i in range(speech_sample.shape[0])
        ],
        axis=0,
    )
    return ret


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean_wav_dir', default='voice_bank/train')
    parser.add_argument('--output_dir', default='voice_bank/clip_train')
    args = parser.parse_args()

    input_lists = os.listdir(args.clean_wav_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    for index in track(input_lists):
        clean_signal, sr = librosa.load(os.path.join(args.clean_wav_dir, index), sr=48000)
        clean_signal = clean_signal[None,:]
        clipped_signal = clipping(clean_signal, min_quantile=0.1, max_quantile=0.9)
        sf.write(os.path.join(args.output_dir, index), clipped_signal[0], sr)
