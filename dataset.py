import os
import random
import torch
import torch.utils.data
import librosa
from librosa.filters import mel as librosa_mel_fn
import numpy as np

def same_len(clean, rir):
    min_len = len(clean)
    clean = clean[:min_len]
    if len(rir) < min_len:
        rir = np.pad(rir, (0,(min_len - len(rir))), mode='constant', constant_values=0)
    rir = rir[:min_len]
    return clean, rir 

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output

def mel_spectrogram(y, n_fft=400, num_mels=80, sampling_rate=8000, hop_size=100, win_size=400, fmin=0, fmax=None, center=True):
    mel = librosa_mel_fn(sr=sampling_rate,n_fft=n_fft,n_mels=num_mels,fmin=fmin,fmax=fmax)
    mel_basis = torch.from_numpy(mel).float().to(y.device)
    hann_window = torch.hann_window(win_size).to(y.device)
    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window, center=True,return_complex=True)
    spec = torch.view_as_real(spec)
    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))
    spec = torch.matmul(mel_basis, spec)
    spec = spectral_normalize_torch(spec)
    return spec #[batch_size,n_fft/2+1,frames]

def mag_pha_stft(y, n_fft, hop_size, win_size, compress_factor=1.0, center=True):

    hann_window = torch.hann_window(win_size).to(y.device)
    stft_spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window,
                           center=center, pad_mode='reflect', normalized=False, return_complex=True)
    stft_spec = torch.view_as_real(stft_spec)
    mag = torch.sqrt(stft_spec.pow(2).sum(-1)+(1e-9))
    pha = torch.atan2(stft_spec[:, :, :, 1]+(1e-10), stft_spec[:, :, :, 0]+(1e-5))
    # Magnitude Compression
    mag = torch.pow(mag, compress_factor)
    com = torch.stack((mag*torch.cos(pha), mag*torch.sin(pha)), dim=-1)

    return mag, pha, com


def mag_pha_istft(mag, pha, n_fft, hop_size, win_size, compress_factor=1.0, center=True):
    # Magnitude Decompression
    mag = torch.pow(mag, (1.0/compress_factor))
    com = torch.complex(mag*torch.cos(pha), mag*torch.sin(pha))
    hann_window = torch.hann_window(win_size).to(com.device)
    wav = torch.istft(com, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window, center=center)

    return wav


def get_dataset_filelist(input_clean_wav_list, input_noisy_wav_list):
    clean_files=[]
    filelist=os.listdir(input_clean_wav_list)
    for files in filelist:

        src=os.path.join(input_clean_wav_list,files)
        clean_files.append(src)

    noisy_files=[]
    filelist=os.listdir(input_noisy_wav_list)
    for files in filelist:
        src=os.path.join(input_noisy_wav_list,files)
        noisy_files.append(src)

    return clean_files, noisy_files

# def get_dataset_filelist(input_clean_file, input_noise_file):
#     with open(input_clean_file, 'r', encoding='utf-8') as fi:
#         clean_indexes = [x.split(' ')[-1] for x in fi.read().split('\n') if len(x) > 0]

#     with open(input_noise_file, 'r', encoding='utf-8') as fi:
#         noise_indexes = [x.split(' ')[-1] for x in fi.read().split('\n') if len(x) > 0]

#     return clean_indexes, noise_indexes

class Dataset(torch.utils.data.Dataset):
    def __init__(self, clean_indexes, noise_indexes, segment_size, split=True, shuffle=True, n_cache_reuse=1, device=None):
        self.clean_audio_indexes = clean_indexes
        self.noise_audio_indexes = noise_indexes
        self.fs = 44100
        random.seed(1234)
        if shuffle:
            random.shuffle(self.clean_audio_indexes)
        self.split = split
        self.segment_size = segment_size
        self.cached_clean_wav = None
        self.cached_noisy_wav = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.device = device

    def __getitem__(self, index):
        clean_filename = self.clean_audio_indexes[index]
        noise_filename = self.noise_audio_indexes[index]
        if self._cache_ref_count == 0:
            clean_audio, sr_c = librosa.load(clean_filename, sr=None)
            noise_audio, sr_n = librosa.load(noise_filename, sr=None)
            clean_audio, noise_audio = same_len(clean_audio, noise_audio)
            self.cached_clean_wav = clean_audio
            self.cached_noisy_wav = noise_audio
            self._cache_ref_count = self.n_cache_reuse
        else:
            clean_audio = self.cached_clean_wav
            noisy_audio = self.cached_noisy_wav
            self._cache_ref_count -= 1
        
        # clean_audio, noisy_audio = torch.FloatTensor(clean_audio), torch.FloatTensor(noise_audio)
        # norm_factor = torch.sqrt(len(noisy_audio) / torch.sum(noisy_audio ** 2.0))
        # clean_audio = (clean_audio * norm_factor).unsqueeze(0)
        # noisy_audio = (noisy_audio * norm_factor).unsqueeze(0)
        clean_audio = torch.tensor(clean_audio).unsqueeze(0)
        noisy_audio = torch.tensor(noise_audio).unsqueeze(0)

        assert clean_audio.size(1) == noisy_audio.size(1)

        if self.split:
            if clean_audio.size(1) >= self.segment_size:
                max_audio_start = clean_audio.size(1) - self.segment_size
                audio_start = random.randint(0, max_audio_start)
                clean_audio = clean_audio[:, audio_start: audio_start+self.segment_size]
                noisy_audio = noisy_audio[:, audio_start: audio_start+self.segment_size]
            else:
                clean_audio = torch.nn.functional.pad(clean_audio, (0, self.segment_size - clean_audio.size(1)), 'constant')
                noisy_audio = torch.nn.functional.pad(noisy_audio, (0, self.segment_size - noisy_audio.size(1)), 'constant')

        return (clean_audio.squeeze(), noisy_audio.squeeze())

    def __len__(self):
        return len(self.clean_audio_indexes)