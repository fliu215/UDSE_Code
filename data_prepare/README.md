## Denoising(DN)
Use the official [VoiceBank+DEMAND dataset](https://datashare.ed.ac.uk/handle/10283/1942)

## Dereverberation (DR)
Use only the "add reverb" part of the ```add_reverb_noise.py``` file

## Bandwidth Extension (BWE)
First downsample to the target sampling rate, then upsample back to 44.1 kHz

## Declipping (DC)
Use the ```add_clip.py``` file

## Phase Distortion Restoration (PDR)
The original phase is replaced with the predicted phase using the inference code of [NSPP](https://github.com/YangAi520/NSPP)

## Compression Distortion Restoration (CDR)
Compress the raw speech using [APCodec](https://github.com/YangAi520/APCodec) with a single VQ
