# CREATOR.md — Domain Adaptation Framework for Speech Recognition

## 1. Project Overview

**What this project does:**
This project is a complete, end-to-end machine learning framework demonstrating **Domain Adaptation** for Automatic Speech Recognition (ASR). It takes a pre-trained ASR model that expects clean, studio-quality speech, tests its degradation on a new domain (noisy speech), and subsequently fine-tunes the model on a small dataset of the new domain to regain its accuracy.

**The problem it solves:**
It solves the **Domain Mismatch** problem. Pre-trained ASR models like wav2vec2 perform exceptionally well on clean speech. However, when deployed in real-world environments with background noise, varying acoustics, or different accents, their performance degrades significantly.

**The approach used:**
The framework utilizes **Supervised Fine-Tuning** for domain adaptation. It starts by taking clean audio and synthetically augmenting it with Gaussian noise to create a "Target Domain". Then, it freezes the low-level acoustic feature encoder of the pre-trained model and fine-tunes only the higher-level transformer layers on the target domain using CTC (Connectionist Temporal Classification) loss. 

**Key innovations:**
- **Layer Freezing Strategy:** Actively freezes the feature encoder (`model.freeze_feature_encoder()`) to prevent catastrophic forgetting and overfitting on a tiny dataset.
- **Dynamic Padding Collator:** Custom `DataCollatorCTCWithPadding` that dynamically pads audio inputs and masks padding tokens in labels with `-100` so CTC loss ignores them.
- **Laptop-Friendly:** Configured specifically to run on consumer hardware (CPU) within ~30 minutes total by limiting the dataset size (~80 samples) and batch sizes (2).

---

## 2. Project Architecture

### Folder Structure
```text
speech_domain_adaptation/
├── 1_data_preparation.py       # Prepares target domain by adding noise
├── 2_baseline_evaluation.py    # Evaluates pretrained model on source/target
├── 3_fine_tune.py              # Fine-tunes wav2vec2 using HF Trainer
├── 4_final_evaluation.py       # Evaluates adapted model, plots comparisons
├── app.py                      # FastAPI Web interface for testing models
├── test_model.py               # CLI tool to test inference side-by-side
├── utils.py                    # Shared logic: audio I/O, augmentations, metrics
├── requirements.txt            # Python dependencies
├── GUIDE.md / HOW_TO_RUN.md    # Documentation
├── data/                       # Contains dataset
│   ├── source/                 # Clean .wav files
│   ├── target/                 # Noisy .wav files (generated)
│   └── metadata.csv            # Mapping of file_path, transcript, domain
├── adapted_model/              # Fine-tuned model checkpoints (generated)
└── results/                    # Output plots and JSON metrics (generated)
```

### Data Flow
1. **Raw Audio Input:** Clean 16kHz `.wav` files and transcripts are read from `data/source/` and `data/metadata.csv`.
2. **Augmentation:** `1_data_preparation.py` applies `add_noise` from `utils.py` and saves them to `data/target/`.
3. **Evaluation (Pre):** `2_baseline_evaluation.py` passes the audio through the frozen HuggingFace `wav2vec2-base-960h` to get baseline Word Error Rate (WER).
4. **Training:** `3_fine_tune.py` uses HuggingFace `Trainer` to pass the noisy data (80% train split) through the model, optimizing weights with CTC loss.
5. **Evaluation (Post):** `4_final_evaluation.py` passes test split data through the new model in `adapted_model/` and generates visual comparisons in `results/`.
6. **Inference:** The user submits audio via `app.py` or `test_model.py`, which is dynamically decoded by both baseline and adapted models for live comparison.

---

## 3. Models

**Base/Pretrained Model:**
- **Name:** `facebook/wav2vec2-base-960h`
- **Source:** HuggingFace Hub
- **Location in project:** Downloaded dynamically via `Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)` in the scripts. 
  - **What is Dynamic Download?** It means the massive model weights (gigabytes in size) are not physically stored in this project's folder. Instead, the `transformers` library automatically connects to the internet to download the model from the HuggingFace servers the very first time you run the scripts. Once downloaded, it caches the files locally on your machine (usually in `~/.cache/huggingface/hub`). All future runs will instantly load the model from this local cache, saving time and disk space in the project directory.

**Finetuned Model:**
- **Name:** `adapted_wav2vec2-base-960h`
- **Where it is saved:** `./adapted_model/` directory.

**Model Architecture Details:**
- **Type:** Wav2Vec2 for Connectionist Temporal Classification (CTC).
- **Components:** 
  1. A multi-layer convolutional feature encoder (extracts latent representations from raw 1D audio waveforms).
  2. A contextualized Transformer network.
  3. A linear classification head predicting characters from a vocabulary.

**Why this model was chosen:**
Wav2vec2 is a self-supervised model by Facebook AI that has already learned robust representations of human speech from 960 hours of LibriSpeech data. This makes it an ideal, strong baseline for testing domain adaptation strategies.

---

## 4. Dataset

**What dataset is used:**
A small custom dataset of 80 clean speech samples.

**Where the data lives:**
- Clean Audio: `data/source/`
- Noisy Audio: `data/target/`
- Metadata & Transcripts: `data/metadata.csv` (contains `file_path`, `transcript`, `domain`, `duration`, `file_id`).
- Test split tracking: `data/test_split.json`

**Data format and preprocessing steps:**
- **Format:** `.wav` files.
- **Preprocessing:** All audio must be exactly 16,000 Hz (16kHz). If not, it is resampled using `librosa.resample`. Text is upper-cased and stripped of excessive whitespace.
- **Feature Extraction:** Audio is processed into 1D tensors via `Wav2Vec2Processor` (`padding=True`, `return_tensors="pt"`). Transcripts are tokenized into label IDs.

**How domain adaptation data is structured:**
The framework utilizes parallel datasets:
- **Source Domain:** Pure, unadulterated speech.
- **Target Domain:** Synthetic noisy speech generated by applying Gaussian white noise (`noise_level` varying cyclically between 0.005, 0.008, 0.010, 0.012, 0.015) to the source audio. 
Data is split **80% for training** and **20% for testing**.

---

## 5. How to Install & Setup

**Step by step installation:**
1. Clone or download the project folder.
2. Open a terminal and navigate into the `speech_domain_adaptation` directory.
3. Install dependencies using pip:
```bash
pip install -r requirements.txt
```

**Environment Setup & Dependencies:**
- Python 3.8+ or Python 3.10+
- `torch>=1.9.0`, `torchaudio>=0.9.0`: Core Deep Learning backend.
- `transformers>=4.20.0`: HuggingFace library for wav2vec2 and the Trainer API.
- `librosa>=0.9.0`, `soundfile>=0.10.0`: For audio reading, writing, and resampling.
- `jiwer>=2.5.0`: For calculating Word Error Rate (WER) and Character Error Rate (CER).
- `matplotlib>=3.5.0`: For generating performance bar charts and spectrograms.
- `fastapi>=0.110.0`, `uvicorn>=0.29.0`: For the web interface.

**API Keys/Credentials:**
None required. The pre-trained model is downloaded from HuggingFace's public hub.

---

## 6. How to Run

**1. Data Preparation (creates the noisy dataset):**
```bash
python 1_data_preparation.py
```

**2. Baseline Evaluation:**
```bash
python 2_baseline_evaluation.py
```

**3. Training / Finetuning:**
```bash
python 3_fine_tune.py
```

**4. Inference / Final Evaluation:**
```bash
python 4_final_evaluation.py
```

**Run the Web Application Interface:**
```bash
python app.py
```
*(Open http://localhost:8000 in a browser)*

**Run the CLI Test Script:**
```bash
python test_model.py path/to/your/audio.wav "THE EXPECTED TRANSCRIPT"
```

**Example Output (from Evaluation):**
```text
+------------------+------------------+------------------+-----------+
|     Domain       |     Baseline     |     Adapted      |  Change   |
+------------------+------------------+------------------+-----------+
|  Source (Clean)  |       3.83%      |       2.35%      |  -1.48%   |
|  Target (Noisy)  |      32.97%      |      21.61%      | -11.36%   |
+------------------+------------------+------------------+-----------+
```

---

## 7. How to Finetune

**Exactly what is being finetuned:**
Only the context network (Transformer layers) and the linear output layer. The multi-layer convolutional feature encoder is completely frozen.

**Why those specific parts are finetuned:**
The feature encoder learns low-level acoustic properties (similar to edge detectors in CNNs for vision). These represent basic speech sounds that transfer well across domains. Freezing the encoder prevents "catastrophic forgetting" of these base representations, drastically reduces memory usage, and stops the model from immediately overfitting the tiny 80-sample dataset.

**Step by step finetuning process (`3_fine_tune.py`):**
1. **Load data:** Metadata is parsed to fetch only `domain="target"` samples.
2. **Split:** Data is split 80% Train / 20% Eval.
3. **Instantiate Dataset:** The custom `SpeechDataset` tokenizes labels and converts `.wav` to tensors via `Wav2Vec2Processor`.
4. **Freeze:** `model.freeze_feature_encoder()` is called.
5. **Trainer Setup:** A HuggingFace `Trainer` is configured with `DataCollatorCTCWithPadding` and hyperparameters.
6. **Train:** `trainer.train()` executes the optimization loop for 5 epochs.
7. **Save:** Model is exported to `./adapted_model/`.

---

## 8. Hyperparameters

All defined at the top of `3_fine_tune.py`.

- **`NUM_EPOCHS` (5):** Total passes over the training dataset. Small because the dataset is tiny and wav2vec2 learns fast.
- **`BATCH_SIZE` (2):** Extremely small to allow training on CPU/laptops without OOM (Out of Memory) errors.
- **`LEARNING_RATE` (1e-4):** Standard starting LR for fine-tuning Transformers.
- **`WARMUP_STEPS` (100):** Gradually increases the learning rate from 0 to 1e-4 over 100 steps to prevent destabilization early in training.
- **`WEIGHT_DECAY` (0.005):** L2 regularization applied via AdamW optimizer to prevent overfitting.
- **`SAVE_STEPS` (50) & `LOGGING_STEPS` (10):** Frequency of checkpointing and printing loss.

**Most important to tune:** 
`BATCH_SIZE` (increase to 8 or 16 if using a GPU) and `NUM_EPOCHS` (increase if underfitting, decrease if validation loss spikes).
**Recommended ranges:** `LEARNING_RATE`: 5e-5 to 3e-4. `NUM_EPOCHS`: 3 to 15.

---

## 9. Training Details

- **Training Loop:** Orchestrated entirely by HuggingFace's `Trainer` API, utilizing a custom `LossTrackingCallback` to log loss values at steps for visualization.
- **Loss Function:** **CTC (Connectionist Temporal Classification) Loss**. CTC is used because in ASR, the exact temporal alignment between the audio wave and the text transcript is unknown. CTC calculates the probability of all possible alignments and maximizes the likelihood of the correct transcript. Padding tokens in labels are set to `-100` so PyTorch's CTC loss ignores them.
- **Optimizer and Scheduler:** The Trainer uses `AdamW` (Adam with weight decay) by default, accompanied by a linear learning rate scheduler with warmup.
- **Hardware/Time:** Takes roughly ~20-30 minutes on a standard CPU. GPU takes < 5 minutes.
- **Domain Adaptation Technical Implementation:** Implemented purely at the data level (Supervised Fine-tuning). We do not use architectural adapters or adversarial networks; we strictly fine-tune the pre-trained weights on the exact conditions (Gaussian noise) of the target domain.

---

## 10. Evaluation & Metrics

**Metrics Used:**
- **WER (Word Error Rate):** Calculates the percentage of words that are wrong. Formula: `(Substitutions + Deletions + Insertions) / Total Words`. Lower is better.
- **CER (Character Error Rate):** Same as WER, but calculated at the individual character/letter level.

**How to evaluate:**
Run `4_final_evaluation.py`. It runs the test split data through the model using `torch.no_grad()`, decodes the output logits using `argmax()`, and compares the predicted string to the reference using the `jiwer` library.

**Baseline vs Finetuned Performance:**
Before domain adaptation, the model gets ~32.97% WER on the noisy target domain. After 5 epochs of fine-tuning, the WER drops to ~21.61%, marking a ~34.5% relative improvement. The source domain (clean speech) is often slightly improved or maintained (e.g. 3.83% -> 2.35%).

---

## 11. Key Design Decisions

- **Why wav2vec2:** State-of-the-art self-supervised representations. It works flawlessly with just CTC (no external language model required for decent performance).
- **Why CTC Loss:** Standard sequence-to-sequence loss when alignment is unknown. Much faster and easier to train than attention-based encoder-decoder ASR architectures.
- **Why dynamic padding:** Audio lengths vary wildly. Batching requires padding. By padding dynamically at batch-time via `DataCollatorCTCWithPadding`, we avoid aggressively truncating long files or wasting memory padding short files to the max dataset length.
- **Why Gaussian Noise for Adaptation:** A deterministic, easily reproducible degradation that mathematically challenges the model's feature space, making it perfect for demonstrating the concepts of Domain Adaptation.

---

## 12. How to Extend This Project

- **How to add new domains:** 
  Open `utils.py`. The project already includes unused functions like `add_background_noise()`, `add_reverb()`, and `add_speed_perturbation()`. Go to `1_data_preparation.py` and swap out `add_noise()` with one of these functions to adapt the model to e.g., echoing rooms instead of static noise.
- **How to add new languages:**
  Change `MODEL_NAME` to a multilingual wav2vec2 checkpoint like `facebook/wav2vec2-large-xlsr-53`. Swap the clean `.wav` files in `data/source/` for files of the new language and update `data/metadata.csv` transcripts.
- **How to swap the base model:**
  Change `MODEL_NAME = "facebook/wav2vec2-base-960h"` in all scripts to another HuggingFace CTC model (e.g., `patrickvonplaten/wav2vec2-base-100h-with-lm`).

---

## 13. Common Errors & Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError: No module named 'transformers'` | Dependencies not installed. | Run `pip install -r requirements.txt`. |
| `RuntimeError: CUDA out of memory` | Batch size is too high for your GPU. | In `3_fine_tune.py`, change `BATCH_SIZE` from 2 to 1. |
| `OSError: Port 8000 is in use` | `app.py` is already running in the background. | Kill the previous terminal session or change `port=8000` to `port=8001` at the bottom of `app.py`. |
| `ValueError: Could not decode audio` (in app) | Using an unsupported codec like m4a. | Convert audio to 16-bit PCM WAV. The app uses `soundfile` and `wave` which prefer strict WAV formats. |

---

## 14. File by File Explanation

### `1_data_preparation.py`
- **What it does:** Reads clean audio and transcripts, synthesizes the noisy dataset, and updates metadata.
- **Inputs:** `data/source/*.wav`, `data/metadata.csv` (source rows).
- **Outputs:** `data/target/*.wav`, updated `data/metadata.csv`, `results/spectrogram_comparison.png`.
- **Key Functions:** `load_local_source_data()` loads clean files. `process_target_domain()` iteratively applies `add_noise()` to create the target dataset.

### `2_baseline_evaluation.py`
- **What it does:** Benchmarks the un-adapted pre-trained model.
- **Inputs:** `data/metadata.csv` and audio files.
- **Outputs:** `results/baseline_results.json` containing metrics.
- **Key Functions:** `evaluate_domain()` loops over audio, passes it to the `transcribe()` utility, and calculates `jiwer` metrics.

### `3_fine_tune.py`
- **What it does:** Houses the entire Deep Learning training logic.
- **Inputs:** The target domain data and pre-trained HuggingFace model.
- **Outputs:** Model weights in `adapted_model/`, `data/test_split.json`, and `results/training_loss.json`.
- **Key Classes/Functions:** 
  - `SpeechDataset`: PyTorch dataset that tokenizes text and extracts audio values.
  - `DataCollatorCTCWithPadding`: Handles batch padding and applies `-100` to padding tokens.
  - `fine_tune()`: Sets up the `Trainer`, freezes the encoder, and starts the loop.

### `4_final_evaluation.py`
- **What it does:** Tests the newly fine-tuned model and produces visual comparisons.
- **Inputs:** `adapted_model/`, baseline metrics, and test split data.
- **Outputs:** `adapted_results.json`, `wer_comparison.png`, `sample_predictions.txt`.
- **Key Functions:** `create_visualizations()` plots matplotlib bar charts. `save_sample_predictions()` writes a text file comparing specific sentence outputs side-by-side.

### `utils.py`
- **What it does:** Shared library for math, audio, plotting, and inference.
- **Inputs/Outputs:** Takes raw numpy arrays, strings, or model objects and returns processed arrays, floats, or transcription strings.
- **Key Functions:** `load_audio()` uses `librosa`. `calculate_wer()` uses `jiwer.wer`. `transcribe()` is a generalized wrapper for model inference without gradient calculation (`torch.no_grad()`).

### `app.py`
- **What it does:** A FastAPI web application offering a GUI for the project.
- **Inputs:** User audio uploads or microphone streams.
- **Outputs:** JSON responses containing latency, WER metrics, and raw transcription strings.
- **Key Functions:** `@app.post("/transcribe")` handles HTTP uploads, resamples arrays on the fly to 16kHz, runs inference on both models, and returns JSON.

### `test_model.py`
- **What it does:** A quick Command Line Interface tool for arbitrary testing.
- **Inputs:** A specific `.wav` file path via `sys.argv`.
- **Outputs:** Prints standard out comparisons.
- **Key Functions:** `test_with_file()` runs `transcribe` on the same file using both models and prints the delta.
