#  Multimodal Emotion Analyser

> Recognise human emotions from video clips using audio, visual frames, and spoken text — simultaneously.

---

## What It Does

This project trains a **multimodal emotion recognition model** on the [MELD dataset](https://affective-meld.github.io/) (Multimodal EmotionLines Dataset) — a collection of ~1,400 video clips from the TV show *Friends*, each labelled with one of 7 emotions: `neutral`, `joy`, `surprise`, `anger`, `sadness`, `disgust`, `fear`.

Instead of relying on just one input signal (like text sentiment), the model analyses **all three modalities at once**:

-  **Visual** — facial expressions and body language from video frames
-  **Audio** — tone, pitch, and speech patterns from the audio track
-  **Text** — the actual spoken words (transcript)

These three streams are fused using a **cross-modal transformer**, allowing each modality to influence how the others are interpreted — the way humans naturally process emotion.

---

## How It Works

### Architecture

```
Video (.mp4)
 ├── 8 sampled frames ──► CLIP Vision Encoder ──► (batch, 512)  ─┐
 ├── Audio waveform  ──► Wav2Vec2 Encoder    ──► (batch, 512)  ─┤──► Transformer Fusion ──► Classifier ──► 7 emotions
 └── Transcript text ──► BERT Encoder        ──► (batch, 512)  ─┘
```

1. **CLIP (Visual)** — OpenAI's `clip-vit-base-patch32` encodes 8 uniformly sampled frames; outputs are averaged to get a single video-level visual embedding.
2. **Wav2Vec2 (Audio)** — Facebook's `wav2vec2-base` processes the raw audio waveform; hidden states are mean-pooled over time.
3. **BERT (Text)** — `bert-base-uncased` encodes the utterance transcript; the `[CLS]` token is used as the text embedding.
4. **Fusion** — All three 512-dim embeddings are stacked as a sequence `(batch, 3, 512)` and passed through a 2-layer `TransformerEncoder` with 8 attention heads, allowing cross-modal attention.
5. **Classifier** — The fused output is flattened `(batch, 1536)` and passed through a 2-layer MLP to predict one of 7 emotion classes.

Pretrained encoders are **frozen** — only the projection layers, fusion transformer, and classifier are trained. Class-weighted cross-entropy loss handles the imbalanced emotion distribution.

---

## Tech Stack

| Tool | Why |
|---|---|
| **PyTorch** | Core deep learning framework |
| **HuggingFace Transformers** | CLIP, Wav2Vec2, BERT pretrained models |
| **Decord** | Fast GPU-accelerated video frame extraction |
| **torchaudio + ffmpeg** | Audio extraction and resampling from video |
| **librosa** | Audio analysis utilities |
| **ftfy** | Fixes broken unicode in transcripts |
| **scikit-learn** | Class weight computation for imbalanced data |
| **OpenCV** | Fallback video reading |
| **Kaggle** | Training environment (GPU access + dataset hosting) |

---

## Dataset

**MELD — Multimodal EmotionLines Dataset**

- Source: Dialogues from the TV show *Friends*
- ~13,000 utterances across train / dev / test splits
- Labels: 7 emotion classes + 3 sentiment classes
- Each utterance is a short `.mp4` clip with transcript

Download from Kaggle: [zaber666/meld-dataset](https://www.kaggle.com/datasets/zaber666/meld-dataset)

| Split | Samples |
|---|---|
| Train | ~9,989 |
| Dev | ~1,109 |
| Test | ~2,610 |

---

## Setup & Installation

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (strongly recommended; 16GB+ VRAM for batch size 8)
- `ffmpeg` installed system-wide

```bash
# Ubuntu / Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg
```

### Install dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers decord opencv-python-headless librosa ftfy scikit-learn tqdm
pip install openai-whisper gradio  # optional, for inference UI
```

### Environment setup

```bash
cp .env.example .env
# Fill in your Kaggle credentials and any optional keys
```

### Download the dataset

```bash
# Set up Kaggle API credentials first (from .env)
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_key

kaggle datasets download -d zaber666/meld-dataset
unzip meld-dataset.zip -d data/
```

---

## How to Run

This project is a Jupyter Notebook designed for **Kaggle** (with GPU). To run locally:

```bash
jupyter notebook multimodal-emotion-analyser.ipynb
```

Or on Kaggle:
1. Upload the notebook to [kaggle.com/code](https://kaggle.com/code)
2. Attach the MELD dataset
3. Enable GPU accelerator (P100 or T4)
4. Run all cells top to bottom

### Training flow (in the notebook)

| Step | Cell |
|---|---|
| 1. Explore dataset structure | Cells 1–7 |
| 2. Install dependencies | Cell 8 |
| 3. Define paths and labels | Cells 9–11 |
| 4. Frame + audio extraction | Cells 12–14 |
| 5. Dataset class + DataLoaders | Cells 15–17 |
| 6. Load pretrained models | Cell 18 |
| 7. Build fusion model | Cell 19 |
| 8. Freeze encoders, set optimizer | Cells 20–21 |
| 9. Train (10 epochs) | Cell 22 |
| 10. Preprocess & cache frames | Cell 23 |

Best model is saved to `/kaggle/working/best_model.pt`.

---

## Screenshots

> _Screenshots and training curves will be added here._

---

## API Keys Needed

| Key | Required? | Where to get it |
|---|---|---|
| `KAGGLE_USERNAME` + `KAGGLE_KEY` |  Yes | [kaggle.com/settings](https://www.kaggle.com/settings) → API |
| `HUGGINGFACE_TOKEN` |  Only for gated models | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |
| `WANDB_API_KEY` |  Optional | [wandb.ai/settings](https://wandb.ai/settings) |

The three HuggingFace models used (`clip-vit-base-patch32`, `wav2vec2-base`, `bert-base-uncased`) are all **public** — no HuggingFace token needed unless you switch to a gated model.

---

## Known Issues / TODO

-  Gradio inference UI (referenced in pip install but not yet implemented)
-  Test set evaluation cell missing
-  Training curves / loss plots not yet added
-  `extract_frames_fast` (cv2 version) and `extract_frames` (decord version) both exist — only the decord version is used in training; the cv2 version is used only for preprocessing cache

---

## License

MIT
