"""
2_baseline_evaluation.py - Baseline Evaluation (Before Domain Adaptation)
==========================================================================
This script:
  1. Loads the pretrained wav2vec2-base-960h model from HuggingFace
  2. Evaluates it on source domain (clean speech) → should get LOW WER
  3. Evaluates it on target domain (noisy speech)  → should get HIGH WER
  4. Saves results to results/baseline_results.json

Usage: python 2_baseline_evaluation.py
"""

import os
import sys
import json
import csv
import time
import warnings
warnings.filterwarnings("ignore")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    transcribe, calculate_wer, calculate_cer,
    get_device, print_separator, ensure_dir
)

# ============================================================
# Configuration
# ============================================================

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
METADATA_PATH = os.path.join(DATA_DIR, "metadata.csv")

MODEL_NAME = "facebook/wav2vec2-base-960h"


def load_metadata():
    """Load the metadata CSV file."""
    print_separator("Loading Metadata")

    if not os.path.exists(METADATA_PATH):
        print(f"  [ERROR] Metadata file not found: {METADATA_PATH}")
        print("  [INFO] Please run 1_data_preparation.py first!")
        sys.exit(1)

    records = {"source": [], "target": []}

    with open(METADATA_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            domain = row["domain"]
            records[domain].append(row)

    print(f"  [OK] Loaded {len(records['source'])} source samples")
    print(f"  [OK] Loaded {len(records['target'])} target samples")

    return records


def load_pretrained_model():
    """Load the pretrained wav2vec2-base-960h model and processor."""
    print_separator("Loading Pretrained Model")

    try:
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        print(f"  [INFO] Loading model: {MODEL_NAME}")
        print("  [INFO] This may take a few minutes on first run...\n")

        processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
        model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)

        device = get_device()
        model = model.to(device)
        model.eval()

        print(f"\n  [OK] Model loaded: {MODEL_NAME}")
        print(f"  [OK] Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        return model, processor, device

    except Exception as e:
        print(f"  [ERROR] Failed to load model: {e}")
        print("  [INFO] Make sure transformers and torch are installed.")
        sys.exit(1)


def evaluate_domain(model, processor, records, domain_name, device):
    """
    Evaluate the model on a set of audio samples.

    Returns:
        dict with WER, CER, and per-sample results
    """
    print_separator(f"Evaluating on {domain_name} Domain")

    results = []
    total_wer = 0.0
    total_cer = 0.0
    num_samples = len(records)

    start_time = time.time()

    for i, record in enumerate(records):
        file_path = record["file_path"]
        reference = record["transcript"].upper().strip()

        # Transcribe
        hypothesis = transcribe(model, processor, file_path, device)

        # Calculate metrics
        sample_wer = calculate_wer(reference, hypothesis)
        sample_cer = calculate_cer(reference, hypothesis)

        total_wer += sample_wer
        total_cer += sample_cer

        results.append({
            "file_path": file_path,
            "reference": reference,
            "hypothesis": hypothesis,
            "wer": round(sample_wer, 4),
            "cer": round(sample_cer, 4)
        })

        # Progress update every 10 samples
        if (i + 1) % 10 == 0 or (i + 1) == num_samples:
            elapsed = time.time() - start_time
            avg_wer_so_far = total_wer / (i + 1)
            print(f"  [PROGRESS] {i + 1}/{num_samples} samples | "
                  f"Avg WER: {avg_wer_so_far:.2%} | "
                  f"Time: {elapsed:.1f}s")

    elapsed_total = time.time() - start_time
    avg_wer = total_wer / num_samples
    avg_cer = total_cer / num_samples

    print(f"\n  {'='*50}")
    print(f"  {domain_name} DOMAIN RESULTS")
    print(f"  {'='*50}")
    print(f"  Average WER:  {avg_wer:.2%} ({avg_wer:.4f})")
    print(f"  Average CER:  {avg_cer:.2%} ({avg_cer:.4f})")
    print(f"  Samples:      {num_samples}")
    print(f"  Time:         {elapsed_total:.1f}s")
    print(f"  {'='*50}")

    # Print some example predictions
    print(f"\n  Sample Predictions ({domain_name}):")
    for j, r in enumerate(results[:3]):
        print(f"\n  --- Sample {j + 1} ---")
        print(f"  REF: {r['reference'][:80]}...")
        print(f"  HYP: {r['hypothesis'][:80]}...")
        print(f"  WER: {r['wer']:.2%}")

    return {
        "domain": domain_name.lower(),
        "avg_wer": round(avg_wer, 4),
        "avg_cer": round(avg_cer, 4),
        "num_samples": num_samples,
        "eval_time_seconds": round(elapsed_total, 1),
        "per_sample_results": results
    }


def save_results(source_results, target_results):
    """Save evaluation results to JSON."""
    print_separator("Saving Results")

    ensure_dir(RESULTS_DIR)

    results = {
        "model": MODEL_NAME,
        "evaluation_type": "baseline",
        "source_domain": {
            "avg_wer": source_results["avg_wer"],
            "avg_cer": source_results["avg_cer"],
            "num_samples": source_results["num_samples"],
            "eval_time_seconds": source_results["eval_time_seconds"]
        },
        "target_domain": {
            "avg_wer": target_results["avg_wer"],
            "avg_cer": target_results["avg_cer"],
            "num_samples": target_results["num_samples"],
            "eval_time_seconds": target_results["eval_time_seconds"]
        },
        "per_sample_results": {
            "source": source_results["per_sample_results"],
            "target": target_results["per_sample_results"]
        }
    }

    results_path = os.path.join(RESULTS_DIR, "baseline_results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"  [OK] Results saved to {results_path}")

    return results


def print_comparison(source_results, target_results):
    """Print side-by-side comparison of source vs target performance."""
    print_separator("BASELINE EVALUATION COMPARISON")

    print(f"""
    {'='*60}
    PRETRAINED MODEL PERFORMANCE (Before Domain Adaptation)
    {'='*60}
    Model: {MODEL_NAME}

    +-------------------+-------------+--------------+
    |     Metric        |   Source    |    Target    |
    |                   |  (Clean)   |   (Noisy)    |
    +-------------------+-------------+--------------+
    |  Word Error Rate  |  {source_results['avg_wer']:>8.2%}   |  {target_results['avg_wer']:>8.2%}    |
    |  Char Error Rate  |  {source_results['avg_cer']:>8.2%}   |  {target_results['avg_cer']:>8.2%}    |
    |  Num Samples      |  {source_results['num_samples']:>8d}   |  {target_results['num_samples']:>8d}    |
    +-------------------+-------------+--------------+

    KEY OBSERVATION:
    - Source domain (clean speech): WER = {source_results['avg_wer']:.2%}  (should be LOW)
    - Target domain (noisy speech): WER = {target_results['avg_wer']:.2%}  (should be HIGH)
    - Performance gap: {abs(target_results['avg_wer'] - source_results['avg_wer']):.2%} WER difference

    This gap demonstrates the DOMAIN MISMATCH problem!
    The model performs well on clean speech but struggles with noisy speech.
    Fine-tuning on target domain data should reduce this gap.
    {'='*60}
    """)


# ============================================================
# Main Entry Point
# ============================================================

def main():
    print("\n" + "=" * 70)
    print("  DOMAIN ADAPTATION - BASELINE EVALUATION")
    print("  Evaluating pretrained wav2vec2 on source and target domains")
    print("=" * 70 + "\n")

    # Step 1: Load metadata
    records = load_metadata()

    # Step 2: Load pretrained model
    model, processor, device = load_pretrained_model()

    # Step 3: Evaluate on source domain (clean speech)
    source_results = evaluate_domain(
        model, processor, records["source"], "SOURCE", device
    )

    # Step 4: Evaluate on target domain (noisy speech)
    target_results = evaluate_domain(
        model, processor, records["target"], "TARGET", device
    )

    # Step 5: Save results
    save_results(source_results, target_results)

    # Step 6: Print comparison
    print_comparison(source_results, target_results)

    print("\n  [DONE] Baseline evaluation complete!")
    print("  [NEXT] Run: python 3_fine_tune.py\n")


if __name__ == "__main__":
    main()
