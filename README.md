# Domain Adaptation Framework for Speech Recognition

A complete framework demonstrating domain adaptation for Automatic Speech Recognition (ASR) using wav2vec2. The project shows how a model trained on clean speech degrades on noisy/accented speech, and how fine-tuning on target domain data significantly improves performance.

## Overview

**Problem:** Pre-trained ASR models perform well on clean speech but struggle with noisy, accented, or domain-specific audio — this is the *domain mismatch* problem.

**Solution:** Fine-tune the pre-trained model on a small amount of target domain data to adapt it to the new acoustic conditions.

**Pipeline:**
1. **Data Preparation** — Create source (clean) and target (noisy) domain datasets
2. **Baseline Evaluation** — Measure pre-trained model performance on both domains
3. **Fine-tuning** — Adapt the model to the target domain
4. **Final Evaluation** — Compare before vs. after adaptation

## Technologies Used

| Technology | Purpose |
|---|---|
| **PyTorch** | Deep learning framework |
| **HuggingFace Transformers** | Pre-trained wav2vec2 model and Trainer |
| **librosa** | Audio processing and feature extraction |
| **jiwer** | WER and CER computation |
| **matplotlib** | Visualization and plotting |
| **torchaudio** | Audio I/O and transformations |

## Project Structure

```
speech_domain_adaptation/
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── utils.py                    # Helper functions (audio, metrics, plots)
├── 1_data_preparation.py       # Step 1: Prepare source & target data
├── 2_baseline_evaluation.py    # Step 2: Evaluate pre-trained model
├── 3_fine_tune.py              # Step 3: Fine-tune on target domain
├── 4_final_evaluation.py       # Step 4: Evaluate adapted model & compare
├── data/
│   ├── source/                 # Clean speech audio files (.wav)
│   ├── target/                 # Noisy speech audio files (.wav)
│   └── metadata.csv            # File paths, transcripts, domain labels
├── adapted_model/              # Fine-tuned model checkpoint
└── results/
    ├── baseline_results.json   # Pre-adaptation metrics
    ├── adapted_results.json    # Post-adaptation metrics
    ├── wer_comparison.png      # WER bar chart (before vs after)
    ├── training_loss.png       # Training loss curve
    ├── spectrogram_comparison.png  # Clean vs noisy spectrograms
    └── sample_predictions.txt  # Example transcriptions
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

```bash
# Clone or download the project
cd speech_domain_adaptation

# Install dependencies
pip install -r requirements.txt
```

## How to Run

Run each script in order. Each step is independent and can be re-run.

### Step 1: Prepare Data
```bash
python 1_data_preparation.py
```
- Loads clean source audio from `data/source/`
- Uses transcripts from existing `data/metadata.csv` source rows
- Creates noisy target audio by adding Gaussian noise only
- Saves metadata CSV

### Step 2: Baseline Evaluation
```bash
python 2_baseline_evaluation.py
```
- Loads pre-trained wav2vec2-base-960h model
- Evaluates on source (clean) and target (noisy) domains
- Saves baseline WER and CER metrics

### Step 3: Fine-tune
```bash
python 3_fine_tune.py
```
- Fine-tunes wav2vec2 on 80% of target domain data
- Trains for 5 epochs with batch size 2 (laptop-friendly)
- Saves adapted model and training loss plot

### Step 4: Final Evaluation
```bash
python 4_final_evaluation.py
```
- Evaluates the fine-tuned model on both domains
- Generates comparison charts and sample predictions
- Prints comprehensive results summary

## Expected Results

| Domain | Baseline WER | Adapted WER | Improvement |
|---|---|---|---|
| Source (Clean) | ~5-10% | ~5-15% | Slight change |
| Target (Noisy) | ~30-60% | ~15-35% | Significant drop |

**Key observation:** The adapted model shows substantial WER reduction on noisy speech while maintaining reasonable performance on clean speech.

## Training Configuration

| Parameter | Value |
|---|---|
| Base model | wav2vec2-base-960h |
| Epochs | 5 |
| Batch size | 2 |
| Learning rate | 1e-4 |
| Warmup steps | 100 |
| Feature encoder | Frozen |
| Dataset size | ~80 samples |

## Hardware Requirements

- **Minimum:** Any modern laptop with 8GB RAM
- **GPU:** Not required (CPU training works, takes ~10-20 minutes)
- **Storage:** ~2GB for model downloads + ~500MB for data
- **Total runtime:** Under 30 minutes on CPU

## Noise Augmentation Pipeline

The target domain is created by applying these augmentations to clean speech:

1. **Gaussian noise** — White noise at varying intensities (σ = 0.005–0.015)

## Key Concepts

- **Domain Adaptation:** Technique to adapt a model from a source domain to a different target domain
- **wav2vec2:** Self-supervised pre-trained speech model by Facebook AI
- **CTC Loss:** Connectionist Temporal Classification — loss function for sequence-to-sequence alignment
- **WER:** Word Error Rate — standard ASR evaluation metric
- **CER:** Character Error Rate — character-level evaluation metric
- **Feature Encoder Freezing:** Keeping lower layers fixed to prevent catastrophic forgetting
