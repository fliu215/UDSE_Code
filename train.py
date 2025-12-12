import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import dac
import sys
import time
import torchaudio
import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from dataset import Dataset, get_dataset_filelist
from models import UDSE
from utils import AttrDict, build_env, scan_checkpoint, load_checkpoint, save_checkpoint
from warmup import WarmupConstantSchedule

torch.backends.cudnn.benchmark = True


def train(rank, a, h):
    if h.num_gpus > 1:
        init_process_group(backend=h.dist_config['dist_backend'], init_method=h.dist_config['dist_url'],
                           world_size=h.dist_config['world_size'] * h.num_gpus, rank=rank)

    torch.cuda.manual_seed(h.seed)
    device = torch.device('cuda:{:d}'.format(rank))

    generator = UDSE()
    dac_model = dac.DAC.load('/home/aiyang/.cache/descript/dac/weights_44khz_8kbps_0.0.1.pth')
    dac_model.to(device)


    if rank == 0:
        num_params = 0
        for p in generator.parameters():
            num_params += p.numel()
        print('Total Parameters: {:.3f}M'.format(num_params/1e6))
        os.makedirs(h.checkpoint_path, exist_ok=True)
        os.makedirs(os.path.join(h.checkpoint_path, 'logs'), exist_ok=True)
        print("checkpoints directory : ", h.checkpoint_path)

    if os.path.isdir(h.checkpoint_path):
        cp_g = scan_checkpoint(h.checkpoint_path, 'g_')
        cp_do = scan_checkpoint(h.checkpoint_path, 'do_')
    
    steps = 0
    if cp_g is None or cp_do is None:
        state_dict_do = None
        last_epoch = -1
    else:
        state_dict_g = load_checkpoint(cp_g, 'cpu')
        state_dict_do = load_checkpoint(cp_do, 'cpu')
        generator.load_state_dict(state_dict_g['generator'])
        steps = state_dict_do['steps'] + 1
        last_epoch = state_dict_do['epoch']

    generator = generator.to(device)

    if h.num_gpus > 1:
        generator = DistributedDataParallel(generator, device_ids=[rank]).to(device)

    optim_g = torch.optim.AdamW(generator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    warmup_scheduler = WarmupConstantSchedule(optim_g, warmup_steps=h.warmup_steps)
    scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(optim_g, T_max=h.training_epochs, eta_min=0.00001, last_epoch=last_epoch)
    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do['optim_g'])

    training_clean_indexes, training_noise_indexes = get_dataset_filelist(h.input_train_clean_list, h.input_train_noise_list)
    validation_clean_indexes, validation_noise_indexes = get_dataset_filelist(h.input_validation_clean_list, h.input_validation_noise_list)

    trainset = Dataset(training_clean_indexes, training_noise_indexes, h.segment_size,
                       split=True, n_cache_reuse=0, shuffle=False , device=device)

    train_sampler = DistributedSampler(trainset) if h.num_gpus > 1 else None

    train_loader = DataLoader(trainset, num_workers=h.num_workers, shuffle=False,
                              sampler=train_sampler,
                              batch_size=h.batch_size,
                              pin_memory=True,
                              drop_last=True)
    if rank == 0:
        validset = Dataset(validation_clean_indexes, validation_noise_indexes, h.segment_size,
                           split=False, shuffle=False, n_cache_reuse=0, device=device)
        
        validation_loader = DataLoader(validset, num_workers=1, shuffle=False,
                                       sampler=None,
                                       batch_size=1,
                                       pin_memory=True,
                                       drop_last=True)

        sw = SummaryWriter(os.path.join(h.checkpoint_path, 'logs'))

    generator.train()
    dac_model.eval()

    for epoch in range(max(0, last_epoch), h.training_epochs):
        if rank == 0:
            start = time.time()
            print("Epoch: {}".format(epoch+1))

        if h.num_gpus > 1:
            train_sampler.set_epoch(epoch)

        for i, batch in enumerate(train_loader):
            torch.cuda.empty_cache()
            if rank == 0:
                start_b = time.time()
            clean_audio, mix_audio = batch
            clean_audio = clean_audio.to(device, non_blocking=True)
            mix_audio = mix_audio.to(device, non_blocking=True)

            clean_audio = clean_audio.unsqueeze(1)
            mix_audio = mix_audio.unsqueeze(1)

            clean_in = dac_model.preprocess(clean_audio, h.sampling_rate)
            _, clean_token_dac, _, _, _ = dac_model.encode(clean_in)    # (B,Q,T)
            clean_token = clean_token_dac.permute(0,2,1)
            clean_token_loss = [clean_token[:,:,k].reshape(-1) for k in range(h.num_quantize)]
            
            dac_in = dac_model.preprocess(mix_audio, h.sampling_rate)
            _, _, latents, _, _ = dac_model.encode(dac_in)
            dac_noisy = latents.permute(0,2,1)

            B, T, _ = dac_noisy.size()
            initial_embed = torch.rand((B, T, 1024)).to(device)   
            input_list = [initial_embed]
            for i in range(h.num_quantize-1):
                input_embed, _, _ = dac_model.quantizer.from_codes(clean_token_dac[:,:i+1,:])
                input_list.append(input_embed.permute(0,2,1))
            prob = generator(input_list, dac_noisy)

            # Generator
            optim_g.zero_grad()
            loss_list = []
            for i in range(h.num_quantize):
                loss_c = F.cross_entropy(prob[i].reshape(-1, h.codebook_size) / 0.1, clean_token_loss[i])
                loss_list.append(loss_c)
            loss = sum(loss_list) / h.num_quantize
            loss.backward()
            optim_g.step()

            if rank == 0:
                # STDOUT logging
                if steps % h.stdout_interval == 0:
                    with torch.no_grad():
                        Q1_error = loss_list[0].item()
                        Q2_error = loss_list[1].item()
                        Q3_error = loss_list[2].item()
                        Q4_error = loss_list[3].item()
                    print('Steps : {:d}, Gen Loss: {:4.3f}, Q1 Loss: {:4.3f}, Q2 Loss: {:4.3f}, Q3 Loss: {:4.3f}, Q4 Loss: {:4.3f}, s/b : {:4.3f}'.
                           format(steps, loss, Q1_error, Q2_error, Q3_error, Q4_error, time.time() - start_b))

                # checkpointing
                if steps % h.checkpoint_interval == 0 and steps != 0:
                    checkpoint_path = "{}/g_{:08d}".format(h.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path,
                                    {'generator': (generator.module if h.num_gpus > 1 else generator).state_dict()})
                    checkpoint_path = "{}/do_{:08d}".format(h.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path, 
                                    {
                                     'optim_g': optim_g.state_dict(), 'steps': steps,
                                     'epoch': epoch})

                # Tensorboard summary logging
                if steps % h.summary_interval == 0:
                    sw.add_scalar("Training/Generator Loss", loss, steps)
                    sw.add_scalar("Training/Q1 Loss", Q1_error, steps)
                    sw.add_scalar("Training/Q2 Loss", Q2_error, steps)
                    sw.add_scalar("Training/Q3 Loss", Q3_error, steps)
                    sw.add_scalar("Training/Q4 Loss", Q4_error, steps)

                # Validation
                if steps % h.validation_interval == 0 and steps != 0:
                    generator.eval()
                    val_cross_err_tot = 0
                    with torch.no_grad():
                        for j, batch in enumerate(validation_loader):
                            clean_audio, mix_audio = batch
                            clean_audio = clean_audio.to(device, non_blocking=True)
                            mix_audio = mix_audio.to(device, non_blocking=True)
                
                            clean_audio = clean_audio.unsqueeze(1)
                            mix_audio = mix_audio.unsqueeze(1)
                
                            clean_in = dac_model.preprocess(clean_audio, h.sampling_rate)
                            _, clean_token_dac, _, _, _ = dac_model.encode(clean_in)    # (B,Q,T)
                            clean_token = clean_token_dac.permute(0,2,1)
                            clean_token_loss = [clean_token[:,:,k].reshape(-1) for k in range(h.num_quantize)]


                            dac_in = dac_model.preprocess(mix_audio, h.sampling_rate)
                            _, _, latents, _, _ = dac_model.encode(dac_in)
                            dac_noisy = latents.permute(0,2,1)

                            B, T, _ = dac_noisy.size()
                            initial_embed = torch.rand((B, T, 1024)).to(device)   
                            input_list = [initial_embed]
                            for i in range(h.num_quantize-1):
                                input_embed, _, _ = dac_model.quantizer.from_codes(clean_token_dac[:,:i+1,:])
                                input_list.append(input_embed.permute(0,2,1))
                            prob = generator(input_list, dac_noisy)

                            loss_list = []
                            for i in range(h.num_quantize):
                                loss_c = F.cross_entropy(prob[i].reshape(-1, h.codebook_size) / 0.1, clean_token_loss[i])
                                loss_list.append(loss_c)
                            loss_cross = sum(loss_list) / h.num_quantize
                            val_cross_err_tot += loss_cross.item()
                            

                        val_cross_err = val_cross_err_tot / (j+1)

                        sw.add_scalar("Validation/Generator Loss", val_cross_err, steps)
                    generator.train()
            steps += 1
            if epoch == 0:
                warmup_scheduler.step()
        scheduler_g.step()
        if rank == 0:
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))


def main():
    print('Initializing Training Process..')

    parser = argparse.ArgumentParser()

    parser.add_argument('--group_name', default=None)
    parser.add_argument('--gpu_num', default=0)
    parser.add_argument('--config', default='config.json')

    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    build_env(a.config, 'config.json', h.checkpoint_path)

    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        h.num_gpus = torch.cuda.device_count()
        h.batch_size = int(h.batch_size / h.num_gpus)
        print('Batch size per GPU :', h.batch_size)
    else:
        pass

    if h.num_gpus > 1:
        mp.spawn(train, nprocs=h.num_gpus, args=(a, h,))
    else:
        train(0, a, h)


if __name__ == '__main__':
    main()