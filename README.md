# FB-MelGAN
A pytorch implementation of the FB-MelGAN(https://arxiv.org/pdf/2005.05106.pdf)

## Prepare dataset
* Download dataset for training. This can be any wav files.
* Edit configuration in utils/audio.py
* Process data: python process.py --wav_dir="wavs" --output="data"

## Pre_train & Train & Tensorboard
* python pre_train.py --input="data/train"
* python train.py --input="data/train"
* tensorboard --logdir logdir

## Inference
* python generate.py --input="data/test"

## Reference
* Multi-band MelGAN: Faster Waveform Generation for High-Quality Text-to-Speech(https://arxiv.org/pdf/2005.05106.pdf)
* MelGAN: Generative Adversarial Networks for Conditional Waveform Synthesis(https://arxiv.org/pdf/1910.06711.pdf)
* kan-bayashi/ParallelWaveGAN(https://github.com/kan-bayashi/ParallelWaveGAN)
* Parallel WaveGAN(https://arxiv.org/pdf/1910.11480.pdf)
