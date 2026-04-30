"""
test_model.py - Test the fine-tuned model on any audio file
==============================================================
Usage:
  python test_model.py                          # test with a sample from the dataset
  python test_model.py path/to/your/audio.wav   # test with your own audio file

Compares output from both the baseline and adapted model side-by-side.
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from utils import load_audio, calculate_wer, calculate_cer, get_device


def load_models():
    """Load both baseline and adapted models."""
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    import torch

    device = get_device()

    # Load baseline (pretrained)
    print("\n  Loading baseline model (facebook/wav2vec2-base-960h)...")
    baseline_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    baseline_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    baseline_model.to(device).eval()

    # Load adapted (fine-tuned)
    adapted_dir = os.path.join(PROJECT_DIR, "adapted_model")
    print(f"  Loading adapted model ({adapted_dir})...")
    adapted_processor = Wav2Vec2Processor.from_pretrained(adapted_dir)
    adapted_model = Wav2Vec2ForCTC.from_pretrained(adapted_dir)
    adapted_model.to(device).eval()

    print("  Both models loaded!\n")
    return baseline_model, baseline_processor, adapted_model, adapted_processor, device


def transcribe(model, processor, audio, device):
    """Transcribe audio array."""
    import torch
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = logits.argmax(dim=-1)
    return processor.batch_decode(predicted_ids)[0].strip()


def test_with_file(audio_path, reference=None):
    """Test both models on a single audio file."""
    baseline_model, baseline_proc, adapted_model, adapted_proc, device = load_models()

    audio, sr = load_audio(audio_path, sr=16000)
    duration = len(audio) / sr

    print("=" * 60)
    print(f"  Audio: {os.path.basename(audio_path)}")
    print(f"  Duration: {duration:.1f}s")
    print("=" * 60)

    # Baseline prediction
    baseline_text = transcribe(baseline_model, baseline_proc, audio, device)
    print(f"\n  BASELINE (pretrained):  {baseline_text}")

    # Adapted prediction
    adapted_text = transcribe(adapted_model, adapted_proc, audio, device)
    print(f"  ADAPTED  (fine-tuned):  {adapted_text}")

    # If reference transcript is available, show WER
    if reference:
        ref = reference.upper().strip()
        b_wer = calculate_wer(ref, baseline_text)
        a_wer = calculate_wer(ref, adapted_text)
        print(f"\n  REFERENCE:              {ref}")
        print(f"\n  Baseline WER: {b_wer:.2%}")
        print(f"  Adapted  WER: {a_wer:.2%}")
        if a_wer < b_wer:
            print(f"  --> Adapted model is BETTER by {b_wer - a_wer:.2%}")
        elif a_wer > b_wer:
            print(f"  --> Baseline model is better by {a_wer - b_wer:.2%}")
        else:
            print(f"  --> Both models perform equally")

    print("\n" + "=" * 60)


def test_with_samples():
    """Test with samples from the dataset."""
    import csv

    metadata_path = os.path.join(PROJECT_DIR, "data", "metadata.csv")
    if not os.path.exists(metadata_path):
        print("  [ERROR] No metadata.csv found. Run 1_data_preparation.py first.")
        return

    baseline_model, baseline_proc, adapted_model, adapted_proc, device = load_models()

    # Read some target domain samples
    samples = []
    with open(metadata_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["domain"] == "target":
                samples.append(row)

    # Test on 5 samples
    num_test = min(5, len(samples))
    print("=" * 60)
    print(f"  Testing on {num_test} noisy (target domain) samples")
    print("=" * 60)

    total_b_wer, total_a_wer = 0, 0

    for i in range(num_test):
        s = samples[i]
        audio, sr = load_audio(s["file_path"], sr=16000)
        ref = s["transcript"].upper().strip()

        b_text = transcribe(baseline_model, baseline_proc, audio, device)
        a_text = transcribe(adapted_model, adapted_proc, audio, device)

        b_wer = calculate_wer(ref, b_text)
        a_wer = calculate_wer(ref, a_text)
        total_b_wer += b_wer
        total_a_wer += a_wer

        print(f"\n  --- Sample {i+1} ({os.path.basename(s['file_path'])}) ---")
        print(f"  REF:      {ref[:80]}...")
        print(f"  BASELINE: {b_text[:80]}...")
        print(f"  ADAPTED:  {a_text[:80]}...")
        print(f"  WER:  Baseline={b_wer:.2%}  Adapted={a_wer:.2%}  {'<-- IMPROVED' if a_wer < b_wer else ''}")

    avg_b = total_b_wer / num_test
    avg_a = total_a_wer / num_test
    print(f"\n{'=' * 60}")
    print(f"  AVERAGE WER:  Baseline={avg_b:.2%}  Adapted={avg_a:.2%}")
    print(f"  Improvement:  {avg_b - avg_a:.2%} absolute, {(avg_b - avg_a)/max(avg_b, 0.001)*100:.1f}% relative")
    print(f"{'=' * 60}\n")


def main():
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
        if not os.path.exists(audio_path):
            print(f"  [ERROR] File not found: {audio_path}")
            sys.exit(1)

        # Optional: pass reference text as second argument
        reference = sys.argv[2] if len(sys.argv) > 2 else None
        test_with_file(audio_path, reference)
    else:
        test_with_samples()


if __name__ == "__main__":
    main()
