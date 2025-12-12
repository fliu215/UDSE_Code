from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import dac
import sys
import argparse
import json
import torch
import torchaudio
import torch.nn.functional as F
from utils import AttrDict
from models import infer_UDSE
import soundfile as sf
import librosa
import numpy as np
from audiotools import AudioSignal
from rich.progress import track

h = None
device = None

def same_len(clean, rir):
    min_len = len(clean)
    clean = clean[:min_len]
    if len(rir) < min_len:
        rir = np.pad(rir, (0,(min_len - len(rir))), mode='constant', constant_values=0)
    rir = rir[:min_len]
    return clean, rir 

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict

def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]

def get_all_audio_files(root_dir, exts=(".wav", ".flac")):
    """
    递归获取 root_dir 下所有音频文件
    """
    audio_files = []
    for ext in exts:
        audio_files.extend(glob.glob(os.path.join(root_dir, "**", f"*{ext}"), recursive=True))
    return sorted(audio_files)

def inference(a):
    generator = infer_UDSE().to(device)
    dac_model = dac.DAC.load('/home/aiyang/.cache/descript/dac/weights_44khz_8kbps_0.0.1.pth')
    dac_model.to(device)

    state_dict = load_checkpoint(h.checkpoint_file, device)
    generator.load_state_dict(state_dict['generator'])

    test_indexes = get_all_audio_files(a.test_noise_wav)
    os.makedirs(a.output_dir, exist_ok=True)

    generator.eval()
    torch.cuda.empty_cache()

    with torch.no_grad():
        for index in track(test_indexes):
            noisy_wav, sr = librosa.load(index, sr=None)
            mix = noisy_wav
            signal = AudioSignal(index)
            signal.to(dac_model.device)
            
            dac_in = dac_model.preprocess(signal.audio_data, h.sampling_rate)
            _, _, latents, _, _ = dac_model.encode(dac_in)
            dac_noisy = latents.permute(0,2,1)
            B, T, _ = dac_noisy.size()
            initial_embed = torch.rand((B, T, 1024)).to(device) 
            token_g = generator(initial_embed, dac_noisy, dac_model) 
            z, _, _ = dac_model.quantizer.from_codes(token_g)
            audio_g = dac_model.decode(z)
            audio_g = audio_g.cpu().numpy()
            mix, audio_g = same_len(mix.squeeze(), audio_g.squeeze())

            name = os.path.basename(index)
            output_file = os.path.join(a.output_dir, name)
            sf.write(output_file, audio_g.squeeze(), sr, 'PCM_16')

def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_noise_wav', default='/train/aiyang/data/voice_bank/noisy_testset_wav_44.1k')
    parser.add_argument('--output_dir', default='/train/aiyang/data/genhancer/UDSE/')
    a = parser.parse_args()

    config_file = 'config.json'
    with open(config_file) as f:
        data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)
    global device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    

    inference(a)


if __name__ == '__main__':
    main()


