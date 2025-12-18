import librosa
import sys
import os 
import glob
import random
from scipy import signal
import numpy as np
import soundfile as sf
from rich.progress import track

def get_all_audio_files(root_dir, exts=(".wav", ".flac")):
    audio_files = []
    for ext in exts:
        audio_files.extend(glob.glob(os.path.join(root_dir, "**", f"*{ext}"), recursive=True))
    return sorted(audio_files)

wav_dir = 'data/train'
rir_dir = 'data/RIR_files/train'
noise_dir = 'data/DEMAND/train'
output = 'data/noisy_train'
os.makedirs(output, exist_ok=True)
wav_index = get_all_audio_files(wav_dir)
rir_index = os.listdir(rir_dir)
noise_index = get_all_audio_files(noise_dir)


for index in track(wav_index):
    clean_wav, sr = librosa.load(index, sr=None)
    # add reverb
    clean_power = np.sum(clean_wav ** 2) / len(clean_wav)
    rir = random.choice(rir_index)
    rir_wav, _ = librosa.load(os.path.join(rir_dir, rir), sr=None)
    reverb_wav = signal.convolve(clean_wav, rir_wav, mode="full")[:len(clean_wav)]
    reverb_power = np.sum(reverb_wav ** 2) / len(reverb_wav)
    reverb_wav = np.sqrt(clean_power / max(reverb_power, 1e-10)) * reverb_wav
    # reverb_wav = clean_wav

    # add noise
    noise = random.choice(noise_index)
    noise_audio, _ = librosa.load(noise, sr=None)
    snr = random.uniform(0,5)
    if len(noise_audio) < len(reverb_wav):
        repeat_times = int(np.ceil(len(reverb_wav) / len(noise_audio)))
        noise_audio = np.tile(noise_audio, repeat_times)
    rand_start = random.randrange(0, len(noise_audio) - len(reverb_wav) + 1)
    noise_audio = noise_audio[rand_start: rand_start + len(reverb_wav)]
    reverb_power = np.sum(reverb_wav ** 2) / len(reverb_wav)
    noise_power = np.sum(noise_audio ** 2) / len(noise_audio)
    scale = (np.sqrt(reverb_power) / np.sqrt(max(noise_power, 1e-10))) * (10 ** (-snr / 20))
    noisy_audio = reverb_wav + scale * noise_audio

    name = os.path.basename(index)
    out_rel = os.path.splitext(name)[0] + '.wav'

    sf.write(os.path.join(output, out_rel), noisy_audio, sr, subtype='PCM_16')
