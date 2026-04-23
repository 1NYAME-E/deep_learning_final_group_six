# Text-to-Music Generation

A deep learning system that generates music audio from natural language descriptions. Given a text prompt like _"a calm piano melody with soft strings"_, the model generates a corresponding mel spectrogram which is then converted to a playable audio clip.

---

## Overview

This project trains a **conditional denoising diffusion probabilistic model (DDPM)** on the [MusicCaps dataset](https://huggingface.co/datasets/google/MusicCaps) by Google DeepMind. Text prompts are encoded using a pre-trained BERT sentence transformer and injected into a U-Net denoiser via cross-attention, guiding the generation of mel spectrograms from Gaussian noise.

---

## Architecture

```
Text Prompt
    │
    ▼
SentenceTransformer (all-mpnet-base-v2)
    │  768-dim embedding
    ▼
U-Net (UNet2DConditionModel)        ◄── cross-attention conditioning
    │  sample_size=(64, 160)
    │  block_out_channels=(64, 128, 256)
    │  down: CrossAttnDownBlock2D × 2 + DownBlock2D
    │  up:   UpBlock2D + CrossAttnUpBlock2D × 2
    ▼
Generated Mel Spectrogram (64 × 160)
    │
    ▼
Griffin-Lim Inversion (librosa)
    │
    ▼
Audio Waveform (16kHz WAV)
```

---

## Dataset

**[MusicCaps](https://huggingface.co/datasets/google/MusicCaps)** — Google DeepMind (2023)

- 5,521 music clips sourced from YouTube
- Each clip paired with a detailed natural language caption written by professional musicians
- Captions describe genre, instrumentation, tempo, mood, and production style
- Clips are 10 seconds in duration

Audio is downloaded via `yt-dlp`, trimmed with `ffmpeg`, resampled to 16kHz mono WAV, and converted to log-scaled mel spectrograms (64 bins, 5s window, normalised to [0,1]).

---

## Project Structure

```
MusicCaps_Project/
├── audio_data/                  # Downloaded .wav files (stored on Drive)
│   ├── sample_0.wav
│   ├── sample_1.wav
│   └── ...
├── music_caps_embeddings.npy    # Pre-computed BERT embeddings (stored on Drive)
└── best_music_model.pt          # Best model checkpoint (stored on Drive)
```

---

## Setup

### Requirements

```bash
pip install yt-dlp librosa datasets
pip install transformers diffusers
pip install sentence-transformers
apt-get install -y ffmpeg
```

### Google Colab (Recommended)

This project is designed to run on **Google Colab with a T4 GPU**.

1. Open the notebook in Colab
2. Set runtime to **GPU**: Runtime → Change runtime type → T4 GPU
3. Mount Google Drive when prompted
4. Run cells in order (see Cell Execution Order below)

---

## Cell Execution Order

| Cell | Description                  |
| ---- | ---------------------------- |
| 1    | Mount Drive                  |
| 2    | Installs & imports           |
| 3    | Load dataset metadata        |
| 4    | Download audio clips         |
| 5–7  | File checks & visualization  |
| 8    | Generate BERT embeddings     |
| 9    | Build Dataset + DataLoader   |
| 10   | Initialize U-Net + Scheduler |
| 11   | Training loop                |
| 12   | Loss plot                    |
| 13   | Inference                    |

## Training

The model is trained for 50 epochs using:

- **Optimiser:** AdamW, lr=1e-4
- **Loss:** Mean Squared Error on predicted vs actual noise
- **Batch size:** 2
- **Scheduler:** DDPM with 1,000 training timesteps
- **Best checkpoint** saved automatically to Drive when validation loss improves

## Inference

After training, run the inference cell and enter a text prompt when asked:

```
Describe the music you want to generate:
> a slow jazz piano with brushed drums
```

The model will:

1. Encode your prompt to a 768-dim BERT embedding
2. Start from pure Gaussian noise
3. Iteratively denoise over several steps guided by your prompt
4. Display the generated mel spectrogram
5. Convert and play the audio at 16kHz

---

## Limitations

- **Dataset scale:** Only a subset of MusicCaps clips can be downloaded due to YouTube rate limiting. State-of-the-art systems train on orders of magnitude more data.
- **Audio quality:** Griffin-Lim inversion introduces artefacts. A neural vocoder (e.g. HiFi-GAN) would produce higher fidelity audio.
- **Colab storage:** `/content/` is wiped on every disconnect — all data must be stored on Drive.

---

## References

- Agostinelli et al. (2023). _MusicLM: Generating Music From Text._ arXiv:2301.11325
- Ho et al. (2020). _Denoising Diffusion Probabilistic Models._ NeurIPS 2020
- Reimers & Gurevych (2019). _Sentence-BERT._ EMNLP 2019
- HuggingFace Diffusers: https://github.com/huggingface/diffusers
- MusicCaps Dataset: https://huggingface.co/datasets/google/MusicCaps
