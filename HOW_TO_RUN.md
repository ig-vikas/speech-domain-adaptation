# How to Run - Speech Domain Adaptation

## Prerequisites

- Python 3.10+ installed
- All dependencies installed:
```bash
pip install -r requirements.txt
```

---

## Step-by-Step: Run the Full Pipeline

Run each script in order from the `speech_domain_adaptation` folder.

### Step 1: Prepare the Data
```bash
python 1_data_preparation.py
```
- Loads clean source audio from `data/source/`
- Uses source transcripts from existing `data/metadata.csv`
- Creates noisy (target) audio files in `data/` using Gaussian noise only
- Generates spectrograms in `results/`
- Takes ~2-5 minutes

### Step 2: Evaluate the Baseline Model
```bash
python 2_baseline_evaluation.py
```
- Tests the pretrained wav2vec2 model on both clean and noisy audio
- Saves WER/CER results to `results/baseline_results.json`
- Takes ~5-10 minutes on CPU

### Step 3: Fine-Tune the Model
```bash
python 3_fine_tune.py
```
- Fine-tunes wav2vec2 on the noisy target domain data
- Saves the adapted model to `adapted_model/`
- Saves training loss plot to `results/`
- Takes ~25-30 minutes on CPU

### Step 4: Evaluate the Adapted Model
```bash
python 4_final_evaluation.py
```
- Tests the fine-tuned model on both domains
- Generates comparison graphs in `results/`
- Saves final results to `results/adapted_results.json`
- Takes ~5-10 minutes on CPU

---

## Run the Web Demo

After completing all 4 steps above:

```bash
python app.py
```

Then open **http://localhost:8000** in your browser.

### What You Can Do on the Web Page

1. **Upload Audio** - Click the upload area or drag & drop a .wav/.mp3/.flac file
2. **Record from Mic** - Click "Record from Mic", speak, then click "Stop Recording"
3. **Transcribe** - Click the "Transcribe" button to compare both models
4. **Test Samples** - Click any sample button in section 3 to instantly test dataset audio
5. **View Results** - Scroll down to see training graphs and project summary

To stop the server, press **Ctrl+C** in the terminal.

---

## Run the Test Script

To quickly compare both models from the command line (no web browser needed):

```bash
python test_model.py
```

Or test with your own audio file:
```bash
python test_model.py your_audio.wav "THE EXPECTED TRANSCRIPT"
```

---

## Folder Structure After Running

```
speech_domain_adaptation/
├── data/
│   ├── source/          <- Clean audio files
│   ├── target/          <- Noisy audio files
│   └── metadata.csv     <- File paths and transcripts
├── adapted_model/       <- Fine-tuned model weights
├── results/
│   ├── baseline_results.json
│   ├── adapted_results.json
│   ├── wer_comparison.png
│   ├── training_loss.png
│   └── spectrogram_comparison.png
├── templates/
│   └── index.html       <- Web frontend
├── 1_data_preparation.py
├── 2_baseline_evaluation.py
├── 3_fine_tune.py
├── 4_final_evaluation.py
├── app.py               <- FastAPI web server
├── test_model.py        <- CLI test script
├── utils.py             <- Helper functions
└── requirements.txt
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| Port 8000 already in use | Close the other terminal running `app.py`, or change the port in `app.py` |
| Slow training | This is normal on CPU (~25-30 min). Use a GPU machine for faster training |
| Storage warning | The pipeline uses ~2-3 GB total. Ensure you have enough free disk space |
| Mic not working on webpage | Allow microphone permission when the browser asks |
