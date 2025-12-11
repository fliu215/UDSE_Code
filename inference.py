from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import dac
import sys
import argparse
import json
import torch
import torchaudio
# import fairseq
import torch.nn.functional as F
from utils import AttrDict
from models_embedding_final_cut import infer_Parallel
import soundfile as sf
import librosa
import numpy as np
from audiotools import AudioSignal
from rich.progress import track
import matplotlib.pyplot as plt

h = None
device = None

def log(t, eps = 1e-20):
    return torch.log(t + eps)

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1):
    return ((t / temperature) + gumbel_noise(t)).argmax(dim = dim)

def same_len(clean, rir):
    min_len = len(clean)
    clean = clean[:min_len]
    if len(rir) < min_len:
        # num_repeat = min_len // len(rir) + 1
        # rir = np.tile(rir, num_repeat)[:min_len]
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

def get_dataset_filelist(input_training_wav_list):
    training_files=[]
    filelist=os.listdir(input_training_wav_list)
    for files in filelist:

        src=os.path.join(input_training_wav_list,files)
        training_files.append(src)
    return training_files

def inference(a):
    device = torch.device('cuda:3')
    generator = infer_Parallel().to(device)
    dac_model_path = dac.utils.download(model_type="44khz")
    dac_model = dac.DAC.load(dac_model_path)
    dac_model.to(device)
    # cp_path = '/home/aiyang/Genhancer/fairseq/libri960_big.pt'
    # ssl_model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
    # ssl_model = ssl_model[0]
    # ssl_model.remove_pretraining_modules()
    # ssl_model.to(device)

    state_dict = load_checkpoint(h.checkpoint_file, device)
    generator.load_state_dict(state_dict['generator'])

    test_indexes = get_dataset_filelist(a.test_noise_wav)
    os.makedirs(a.output_dir, exist_ok=True)

    generator.eval()
    torch.cuda.empty_cache()

    with torch.no_grad():
        for index in track(test_indexes):
            # torch.cuda.empty_cache()
            noisy_wav, sr = librosa.load(index, sr=None)
            signal = AudioSignal(index)
            signal.to(dac_model.device)
            # wav = torch.tensor(noisy_wav).unsqueeze(0)
            # wav = torchaudio.transforms.Resample(orig_freq=44100, new_freq=16000)(wav).to(device)
            # res = ssl_model(wav, mask=False, features_only=True)
            # ssl_feature = res['x']

            mix = noisy_wav

            dac_in = dac_model.preprocess(signal.audio_data, h.sampling_rate)
            z1, c_t, latents, _, _ = dac_model.encode(dac_in)
            dac_noisy = latents.permute(0,2,1)
            B, T, _ = dac_noisy.size()
            # ssl_feature = torch.nn.functional.interpolate(ssl_feature.permute(0,2,1), size=(T,), mode='linear', align_corners=True).permute(0,2,1)
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
    parser.add_argument('--test_noise_wav', default='./mnt230/data/voice_bank/noisy_testset_wav_44.1k')
    parser.add_argument('--output_dir', default='./mnt230/data/genhancer/output-44k-embedding_final_cut/')
    a = parser.parse_args()

    config_file = './Genhancer/config_embedding_final_cut.json'
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


