# Domain Adaptation Framework for Speech Recognition

## A Complete Beginner's Guide

---

## What Is This Project?

Imagine you have a smart assistant (like Siri or Alexa) that can understand what you say. It was trained to understand **clean, studio-quality speech**. But when you use it in a **noisy coffee shop** or with a **different accent**, it starts making mistakes. That's the **domain mismatch problem**.

This project demonstrates how to **fix** that problem using a technique called **Domain Adaptation**.

### The Simple Version:

```
1. We take a speech-to-text AI model (wav2vec2 by Facebook)
2. We test it on noisy audio  -->  it performs BADLY  (32.97% errors)
3. We fine-tune it on noisy audio  -->  it LEARNS from it
4. We test again  -->  it performs MUCH BETTER  (21.61% errors)
5. We show the improvement with graphs
```

---

## Key Concepts Explained (For Beginners)

### What is ASR (Automatic Speech Recognition)?
ASR is the technology that converts spoken words into text. When you say "Hello" to your phone and it types "Hello" -- that's ASR.

### What is wav2vec2?
wav2vec2 is a pre-trained AI model created by Facebook (Meta). Think of it as a student who has already learned English in a quiet classroom. It's really good at understanding clean speech, but struggles in noisy environments.

### What is Domain Adaptation?
Domain = the type of data (clean speech vs noisy speech).
Adaptation = teaching the model to handle a new type.
So **domain adaptation** = teaching a clean-speech model to handle noisy speech.

### What is Fine-Tuning?
The model already knows 95% of what it needs. Fine-tuning means we give it a small amount of new data (noisy speech) and let it adjust itself slightly. It's like a student doing extra practice problems before an exam.

### What is WER (Word Error Rate)?
WER measures how many words the model gets wrong.
- WER = 0% means PERFECT (every word correct)
- WER = 100% means TERRIBLE (every word wrong)
- WER = 5% means great (only 5 out of 100 words wrong)

### What is CER (Character Error Rate)?
Same as WER but at the character/letter level instead of word level.

### What is CTC Loss?
CTC (Connectionist Temporal Classification) is the math formula used during training. It helps the model learn to align audio with text even when we don't know exactly which part of the audio corresponds to which letter.

### What Does "Freezing Layers" Mean?
The model has two parts:
1. **Feature Encoder** (lower layers) -- extracts basic audio features (like recognizing sounds)
2. **Transformer** (upper layers) -- converts sounds into words

We **freeze** the feature encoder (don't change it) and only train the transformer. This prevents the model from forgetting what it already knows (called "catastrophic forgetting").

---

## Project Structure (What Each File Does)

```
speech_domain_adaptation/
|
|-- requirements.txt            # List of Python packages needed
|-- README.md                   # This guide
|-- utils.py                    # Helper functions used by all scripts
|
|-- 1_data_preparation.py       # STEP 1: Load local audio + create noisy versions
|-- 2_baseline_evaluation.py    # STEP 2: Test the original model (before training)
|-- 3_fine_tune.py              # STEP 3: Train the model on noisy data
|-- 4_final_evaluation.py       # STEP 4: Test the improved model + generate graphs
|-- test_model.py               # BONUS: Test with any audio file
|
|-- data/
|   |-- source/                 # Clean speech audio files (.wav)
|   |-- target/                 # Noisy speech audio files (.wav)
|   +-- metadata.csv            # Info about each audio file
|
|-- adapted_model/              # The fine-tuned model (saved after training)
|
+-- results/
    |-- baseline_results.json       # Numbers from step 2
    |-- adapted_results.json        # Numbers from step 4
    |-- wer_comparison.png          # Bar chart: before vs after
    |-- training_loss.png           # Graph: loss going down during training
    |-- spectrogram_comparison.png  # Visual: clean audio vs noisy audio
    +-- sample_predictions.txt      # Example transcriptions
```

### File-by-File Explanation:

#### `utils.py` - Helper Functions
Contains reusable functions that other scripts use:
- `load_audio()` -- loads a .wav file
- `add_noise()` -- adds static/white noise to audio
- `add_reverb()` -- adds echo effect
- `calculate_wer()` -- computes Word Error Rate
- `plot_spectrogram()` -- draws a visual picture of audio
- `transcribe()` -- feeds audio to the model and gets text back

#### `1_data_preparation.py` - Data Preparation
- Loads up to 80 clean source speech files from `data/source/`
- Reads source transcripts from existing `data/metadata.csv`
- Creates noisy versions by adding Gaussian noise only
- Saves them as "target domain" audio
- Creates a CSV file listing all audio files and their transcripts

#### `2_baseline_evaluation.py` - Baseline Evaluation
- Downloads the pre-trained wav2vec2 model from HuggingFace
- Feeds every audio file to the model and gets predictions
- Compares predictions to actual transcripts
- Calculates WER and CER for both clean and noisy audio
- Shows that clean audio = low errors, noisy audio = high errors

#### `3_fine_tune.py` - Fine-Tuning (Training)
- Loads the pre-trained model
- Splits noisy data: 80% for training, 20% for testing
- Trains the model for 5 epochs (5 passes through the data)
- The model gradually learns to handle noise
- Saves the improved model to `adapted_model/`

#### `4_final_evaluation.py` - Final Evaluation
- Loads the fine-tuned model
- Tests it on both clean and noisy audio
- Compares with the baseline results
- Generates all the graphs and charts
- Prints a final summary table

#### `test_model.py` - Test With Any Audio
- Lets you test both models side-by-side on any audio file
- Shows which model gives better results

---

## Installation (Step by Step)

### Step 1: Make Sure Python is Installed
Open a terminal/command prompt and type:
```bash
python --version
```
You should see something like `Python 3.10.x` or higher. If not, download Python from https://python.org

### Step 2: Navigate to the Project Folder
```bash
cd path/to/speech_domain_adaptation
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```
This installs all the Python packages the project needs. It may take a few minutes.

If the above fails, try installing them one by one:
```bash
pip install torch torchaudio transformers
pip install librosa soundfile jiwer matplotlib
pip install numpy pandas scikit-learn tqdm accelerate
```

---

## How to Run the Full Pipeline

Run these 4 commands IN ORDER. Each one must finish before running the next:

```bash
# Step 1: Prepare the data (uses local source audio, creates noisy versions)
# Takes ~2-5 minutes
python 1_data_preparation.py

# Step 2: Test the original model (before any training)
# Takes ~5-10 minutes
python 2_baseline_evaluation.py

# Step 3: Fine-tune the model (the actual training)
# Takes ~20-30 minutes on CPU
python 3_fine_tune.py

# Step 4: Test the improved model and generate graphs
# Takes ~5-10 minutes
python 4_final_evaluation.py
```

**Total time: approximately 30-50 minutes on a laptop (CPU only)**

---

## What the Output Looks Like

### After Step 2 (Baseline):
```
Source WER (clean audio):  3.83%   <-- model is GOOD on clean speech
Target WER (noisy audio): 32.97%  <-- model is BAD on noisy speech
Gap: 29.14%                        <-- this is the problem we want to fix
```

### After Step 3 (Training):
You'll see the loss decreasing:
```
Step  10 | Loss: 275.09
Step  50 | Loss: 244.69
Step 100 | Loss: 158.14
Step 140 | Loss:  74.05    <-- loss going DOWN = model is LEARNING
Step 160 | Loss: 100.62
```

### After Step 4 (Final Results):
```
+------------------+------------------+------------------+-----------+
|     Domain       |     Baseline     |     Adapted      |  Change   |
+------------------+------------------+------------------+-----------+
|  Source (Clean)  |       3.83%      |       2.35%      |  -1.48%   |
|  Target (Noisy)  |      32.97%      |      21.61%      | -11.36%   |
+------------------+------------------+------------------+-----------+

Target domain WER improvement: 34.5%
Domain adaptation SUCCEEDED!
```

---

## How to Demonstrate This Project

### Quick Demo (2 minutes)
If you've already run the full pipeline, just run:
```bash
python test_model.py
```
This will:
- Load both models (original and fine-tuned)
- Test on 5 noisy audio samples
- Show side-by-side comparison

**Example output you'll see:**
```
Sample 1 (noisy audio):
  REFERENCE: CONCORD RETURNED TO ITS PLACE AMIDST THE TENTS
  BASELINE:  CON HORDER SHOT HIS LIGHT OF ITS DETEST        (100% errors!)
  ADAPTED:   CONGOR RETURNED TO HIS PLACE AMIDST THE TENTS  (25% errors)
  --> Adapted model is BETTER!
```

### Demo With Your Own Audio File
Record yourself saying something, save it as a .wav file, and run:
```bash
python test_model.py your_recording.wav "WHAT YOU ACTUALLY SAID"
```

### Full Live Demo (for presentations)
1. Show the spectrogram comparison image: `results/spectrogram_comparison.png`
   - "This is what clean vs noisy audio looks like"

2. Show the baseline results: `results/baseline_results.json`
   - "The model gets 3.83% errors on clean speech but 32.97% on noisy speech"

3. Show the training loss curve: `results/training_loss.png`
   - "During training, the loss goes down - the model is learning"

4. Show the WER comparison chart: `results/wer_comparison.png`
   - "After adaptation, noisy speech errors dropped from 32.97% to 21.61%"

5. Show sample predictions: `results/sample_predictions.txt`
   - "Here are actual examples - look how much better the predictions are"

6. Run live demo: `python test_model.py`
   - "Let's test it in real time"

---

## How to Use the Fine-Tuned Model in Your Own Code

### Basic Usage (copy-paste this):
```python
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import librosa

# Load the fine-tuned model
processor = Wav2Vec2Processor.from_pretrained("adapted_model/")
model = Wav2Vec2ForCTC.from_pretrained("adapted_model/")
model.eval()

# Load any audio file (must be 16kHz)
audio, sr = librosa.load("your_audio.wav", sr=16000)

# Transcribe
inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
with torch.no_grad():
    logits = model(inputs.input_values).logits
predicted_ids = logits.argmax(dim=-1)
text = processor.batch_decode(predicted_ids)[0]

print(f"Transcription: {text}")
```

### Using in a Flask Web App:
```python
from flask import Flask, request, jsonify
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch, librosa, io, soundfile as sf

app = Flask(__name__)

# Load model once at startup
processor = Wav2Vec2Processor.from_pretrained("adapted_model/")
model = Wav2Vec2ForCTC.from_pretrained("adapted_model/")
model.eval()

@app.route("/transcribe", methods=["POST"])
def transcribe():
    audio_file = request.files["audio"]
    audio, sr = sf.read(io.BytesIO(audio_file.read()))
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        logits = model(inputs.input_values).logits
    text = processor.batch_decode(logits.argmax(dim=-1))[0]
    return jsonify({"transcription": text})

if __name__ == "__main__":
    app.run(debug=True)
```

---

## Understanding the Results

### What the Graphs Show:

**`wer_comparison.png`** (Bar Chart)
- Red bars = Baseline (before training)
- Green bars = Adapted (after training)
- The green bar for "Target (Noisy)" should be much shorter than the red bar
- This visually proves domain adaptation worked

**`training_loss.png`** (Line Chart)
- X-axis = training steps
- Y-axis = loss (error) value
- The line should go DOWN from left to right
- This proves the model is learning during training

**`spectrogram_comparison.png`** (Two Heatmaps)
- Left = clean audio, Right = noisy audio
- Spectrograms show frequency (pitch) over time
- The noisy one has more "fuzz" and random patterns
- This shows why the model struggles with noisy audio

---

## Our Actual Results

| Metric | Before Training | After Training | Improvement |
|--------|----------------|----------------|-------------|
| Source WER (Clean) | 3.83% | 2.35% | -1.48% better |
| Target WER (Noisy) | 32.97% | 21.61% | -11.36% better |
| Source CER (Clean) | 1.16% | 0.69% | -0.47% better |
| Target CER (Noisy) | 19.57% | 9.34% | -10.23% better |

**Key takeaway:** The fine-tuned model reduced noisy speech errors by **34.5%** while also slightly improving on clean speech.

---

## Troubleshooting Common Issues

### "ModuleNotFoundError: No module named 'xxx'"
Solution: Install the missing package:
```bash
pip install xxx
```

### "Out of memory" during training
Solution: Reduce batch size in `3_fine_tune.py`:
```python
BATCH_SIZE = 1  # Change from 2 to 1
```

### Training takes too long
Solution: Reduce epochs in `3_fine_tune.py`:
```python
NUM_EPOCHS = 3  # Change from 5 to 3
```

### "CUDA not available" message
This is normal! The project works on CPU. GPU just makes it faster.

### Data preparation fails before training
Make sure you already have clean `.wav` files inside `data/source/` and that `data/metadata.csv` contains matching rows where `domain=source`.

---

## Technologies Used

| Technology | Version | What It Does |
|---|---|---|
| Python | 3.8+ | Programming language |
| PyTorch | 2.0+ | Deep learning framework (runs the neural network) |
| HuggingFace Transformers | 4.20+ | Provides the wav2vec2 model and training tools |
| librosa | 0.9+ | Audio processing (loading, resampling, spectrograms) |
| jiwer | 2.5+ | Calculates WER and CER metrics |
| matplotlib | 3.5+ | Creates the graphs and charts |
| soundfile | 0.10+ | Reads and writes .wav audio files |
| accelerate | 1.1+ | Helps HuggingFace Trainer work with PyTorch |

---

## Hardware Requirements

- **Minimum RAM:** 8 GB
- **GPU:** Not required (CPU works fine)
- **Storage:** ~2 GB for models + ~100 MB for data
- **OS:** Windows, macOS, or Linux
- **Total runtime:** ~30-50 minutes on CPU

---

## The Science Behind It (For Reports)

### Why Does Domain Mismatch Happen?
Pre-trained ASR models learn statistical patterns from their training data. When the test data has different characteristics (noise, accent, microphone quality), these patterns don't match, causing errors.

### Why Does Fine-Tuning Work?
Fine-tuning adjusts the model's internal weights (parameters) using a small amount of target domain data. The model learns to:
1. Ignore noise patterns
2. Focus on speech features even when corrupted
3. Map noisy audio patterns to correct text

### Why Freeze the Feature Encoder?
- The feature encoder learns low-level audio representations (like edges in image recognition)
- These representations are general-purpose and transfer well across domains
- Training them on a small dataset could cause overfitting (memorizing instead of learning)
- Freezing these layers keeps the stable foundation while only adapting the higher-level understanding

### CTC Loss Explained Simply
CTC lets the model output a prediction at every time step without needing exact alignment. For example, if the audio says "cat", the model might output "c-c-a-a-t-t" and CTC automatically collapses repeated characters and removes blanks to get "cat".
