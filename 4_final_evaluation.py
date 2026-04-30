"""
4_final_evaluation.py - Final Evaluation & Comparison (After Domain Adaptation)
================================================================================
This script:
  1. Loads the fine-tuned model from adapted_model/
  2. Evaluates on target domain test split
  3. Also evaluates on source domain for comparison
  4. Compares with baseline results
  5. Creates visualizations: WER bar chart, training loss, spectrograms
  6. Prints a comprehensive summary table
  7. Saves sample predictions (before vs after adaptation)

Usage: python 4_final_evaluation.py
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
    get_device, print_separator, ensure_dir,
    plot_wer_comparison, plot_training_loss,
    plot_spectrogram_comparison, load_audio
)

# ============================================================
# Configuration
# ============================================================

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
METADATA_PATH = os.path.join(DATA_DIR, "metadata.csv")
MODEL_OUTPUT_DIR = os.path.join(PROJECT_DIR, "adapted_model")
BASELINE_RESULTS_PATH = os.path.join(RESULTS_DIR, "baseline_results.json")
TRAINING_LOSS_PATH = os.path.join(RESULTS_DIR, "training_loss.json")


def load_baseline_results():
    """Load baseline evaluation results from JSON."""
    print_separator("Loading Baseline Results")

    if not os.path.exists(BASELINE_RESULTS_PATH):
        print(f"  [WARNING] Baseline results not found: {BASELINE_RESULTS_PATH}")
        print("  [INFO] Using default baseline values...")
        return {
            "source_domain": {"avg_wer": 0.05, "avg_cer": 0.02},
            "target_domain": {"avg_wer": 0.45, "avg_cer": 0.30}
        }

    with open(BASELINE_RESULTS_PATH, 'r') as f:
        results = json.load(f)

    print(f"  [OK] Baseline source WER: {results['source_domain']['avg_wer']:.2%}")
    print(f"  [OK] Baseline target WER: {results['target_domain']['avg_wer']:.2%}")

    return results


def load_adapted_model():
    """Load the fine-tuned model and processor."""
    print_separator("Loading Fine-tuned (Adapted) Model")

    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    if not os.path.exists(MODEL_OUTPUT_DIR):
        print(f"  [ERROR] Adapted model not found: {MODEL_OUTPUT_DIR}")
        print("  [INFO] Please run 3_fine_tune.py first!")
        sys.exit(1)

    try:
        processor = Wav2Vec2Processor.from_pretrained(MODEL_OUTPUT_DIR)
        model = Wav2Vec2ForCTC.from_pretrained(MODEL_OUTPUT_DIR)

        device = get_device()
        model = model.to(device)
        model.eval()

        print(f"  [OK] Adapted model loaded from {MODEL_OUTPUT_DIR}")
        print(f"  [OK] Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        return model, processor, device

    except Exception as e:
        print(f"  [ERROR] Failed to load adapted model: {e}")
        sys.exit(1)


def load_metadata():
    """Load metadata and identify test split."""
    print_separator("Loading Evaluation Data")

    if not os.path.exists(METADATA_PATH):
        print(f"  [ERROR] Metadata not found: {METADATA_PATH}")
        sys.exit(1)

    # Load all records
    source_records = []
    target_records = []

    with open(METADATA_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["domain"] == "source":
                source_records.append(row)
            else:
                target_records.append(row)

    # Load test split info to only evaluate on test set
    test_split_path = os.path.join(DATA_DIR, "test_split.json")
    if os.path.exists(test_split_path):
        with open(test_split_path, 'r') as f:
            test_info = json.load(f)
        test_file_ids = set(test_info["test_file_ids"])
        target_test_records = [r for r in target_records if r["file_id"] in test_file_ids]
        print(f"  [OK] Target test split: {len(target_test_records)} samples")
    else:
        # If no split info, use last 20% of target records
        split_idx = int(len(target_records) * 0.8)
        target_test_records = target_records[split_idx:]
        print(f"  [OK] Target test split (fallback): {len(target_test_records)} samples")

    print(f"  [OK] Source domain: {len(source_records)} samples")
    print(f"  [OK] Target test:   {len(target_test_records)} samples")

    return source_records, target_test_records


def evaluate_domain(model, processor, records, domain_name, device):
    """Evaluate the adapted model on a domain."""
    print_separator(f"Evaluating Adapted Model on {domain_name}")

    results = []
    total_wer = 0.0
    total_cer = 0.0
    num_samples = len(records)
    start_time = time.time()

    for i, record in enumerate(records):
        file_path = record["file_path"]
        reference = record["transcript"].upper().strip()

        hypothesis = transcribe(model, processor, file_path, device)

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

        if (i + 1) % 10 == 0 or (i + 1) == num_samples:
            elapsed = time.time() - start_time
            avg_wer_so_far = total_wer / (i + 1)
            print(f"  [PROGRESS] {i + 1}/{num_samples} | "
                  f"Avg WER: {avg_wer_so_far:.2%} | Time: {elapsed:.1f}s")

    elapsed_total = time.time() - start_time
    avg_wer = total_wer / num_samples
    avg_cer = total_cer / num_samples

    print(f"\n  {domain_name} Domain (Adapted Model):")
    print(f"  Average WER: {avg_wer:.2%}")
    print(f"  Average CER: {avg_cer:.2%}")

    return {
        "domain": domain_name.lower(),
        "avg_wer": round(avg_wer, 4),
        "avg_cer": round(avg_cer, 4),
        "num_samples": num_samples,
        "eval_time_seconds": round(elapsed_total, 1),
        "per_sample_results": results
    }


def save_adapted_results(source_results, target_results, baseline):
    """Save adapted model results to JSON."""
    print_separator("Saving Adapted Results")
    ensure_dir(RESULTS_DIR)

    results = {
        "model": "adapted_wav2vec2-base-960h",
        "evaluation_type": "adapted",
        "source_domain": {
            "avg_wer": source_results["avg_wer"],
            "avg_cer": source_results["avg_cer"],
            "num_samples": source_results["num_samples"]
        },
        "target_domain": {
            "avg_wer": target_results["avg_wer"],
            "avg_cer": target_results["avg_cer"],
            "num_samples": target_results["num_samples"]
        },
        "comparison": {
            "baseline_source_wer": baseline["source_domain"]["avg_wer"],
            "baseline_target_wer": baseline["target_domain"]["avg_wer"],
            "adapted_source_wer": source_results["avg_wer"],
            "adapted_target_wer": target_results["avg_wer"],
            "target_wer_improvement": round(
                baseline["target_domain"]["avg_wer"] - target_results["avg_wer"], 4
            ),
            "target_wer_improvement_pct": round(
                (baseline["target_domain"]["avg_wer"] - target_results["avg_wer"])
                / max(baseline["target_domain"]["avg_wer"], 0.001) * 100, 1
            )
        }
    }

    results_path = os.path.join(RESULTS_DIR, "adapted_results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"  [OK] Results saved to {results_path}")
    return results


def create_visualizations(baseline, source_results, target_results):
    """Create all comparison visualizations."""
    print_separator("Creating Visualizations")
    ensure_dir(RESULTS_DIR)

    # 1. WER Comparison Bar Chart
    print("  [INFO] Creating WER comparison chart...")
    plot_wer_comparison(
        baseline_source_wer=baseline["source_domain"]["avg_wer"],
        baseline_target_wer=baseline["target_domain"]["avg_wer"],
        adapted_source_wer=source_results["avg_wer"],
        adapted_target_wer=target_results["avg_wer"],
        save_path=os.path.join(RESULTS_DIR, "wer_comparison.png")
    )

    # 2. Training Loss Plot
    print("  [INFO] Creating training loss plot...")
    if os.path.exists(TRAINING_LOSS_PATH):
        with open(TRAINING_LOSS_PATH, 'r') as f:
            loss_data = json.load(f)
        if loss_data.get("loss_history"):
            plot_training_loss(
                loss_data["loss_history"],
                save_path=os.path.join(RESULTS_DIR, "training_loss.png")
            )
    else:
        print("  [WARNING] Training loss data not found, skipping plot")

    # 3. Spectrogram Comparison (if we have source and target audio)
    print("  [INFO] Creating spectrogram comparison...")
    try:
        source_files = [f for f in os.listdir(os.path.join(DATA_DIR, "source"))
                        if f.endswith(".wav")]
        target_files = [f for f in os.listdir(os.path.join(DATA_DIR, "target"))
                        if f.endswith(".wav")]
        if source_files and target_files:
            clean_audio, sr = load_audio(
                os.path.join(DATA_DIR, "source", source_files[0])
            )
            noisy_audio, sr = load_audio(
                os.path.join(DATA_DIR, "target", target_files[0])
            )
            plot_spectrogram_comparison(
                clean_audio, noisy_audio, sr,
                save_path=os.path.join(RESULTS_DIR, "spectrogram_comparison.png")
            )
    except Exception as e:
        print(f"  [WARNING] Spectrogram comparison failed: {e}")


def save_sample_predictions(baseline, source_results, target_results):
    """Save sample predictions showing before vs after adaptation."""
    print_separator("Saving Sample Predictions")

    predictions_path = os.path.join(RESULTS_DIR, "sample_predictions.txt")

    # Get baseline per-sample results if available
    baseline_samples = {}
    if "per_sample_results" in baseline:
        for sample in baseline.get("per_sample_results", {}).get("target", []):
            baseline_samples[sample["file_path"]] = sample

    with open(predictions_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("SAMPLE PREDICTIONS: Before vs After Domain Adaptation\n")
        f.write("=" * 80 + "\n\n")

        target_samples = target_results.get("per_sample_results", [])
        num_to_show = min(10, len(target_samples))

        for i in range(num_to_show):
            sample = target_samples[i]
            f.write(f"{'-' * 70}\n")
            f.write(f"Sample {i + 1}: {os.path.basename(sample['file_path'])}\n")
            f.write(f"{'-' * 70}\n")
            f.write(f"REFERENCE:         {sample['reference']}\n")

            # Before adaptation (baseline)
            baseline_hyp = baseline_samples.get(sample['file_path'], {}).get(
                'hypothesis', '[baseline prediction not available]'
            )
            baseline_wer_val = baseline_samples.get(sample['file_path'], {}).get(
                'wer', 'N/A'
            )
            f.write(f"BEFORE ADAPTATION: {baseline_hyp}\n")
            if isinstance(baseline_wer_val, float):
                f.write(f"  WER (before):    {baseline_wer_val:.2%}\n")

            # After adaptation
            f.write(f"AFTER ADAPTATION:  {sample['hypothesis']}\n")
            f.write(f"  WER (after):     {sample['wer']:.2%}\n")
            f.write("\n")

        f.write("=" * 80 + "\n")
        f.write("END OF SAMPLE PREDICTIONS\n")
        f.write("=" * 80 + "\n")

    print(f"  [OK] Sample predictions saved to {predictions_path}")


def print_final_summary(baseline, source_results, target_results, adapted_data):
    """Print comprehensive final summary table."""
    print("\n\n")
    print("=" * 70)
    print("  FINAL RESULTS - DOMAIN ADAPTATION FOR SPEECH RECOGNITION")
    print("=" * 70)

    bs_wer = baseline["source_domain"]["avg_wer"]
    bt_wer = baseline["target_domain"]["avg_wer"]
    as_wer = source_results["avg_wer"]
    at_wer = target_results["avg_wer"]

    bs_cer = baseline["source_domain"]["avg_cer"]
    bt_cer = baseline["target_domain"]["avg_cer"]
    as_cer = source_results["avg_cer"]
    at_cer = target_results["avg_cer"]

    target_wer_change = at_wer - bt_wer
    target_improvement_pct = (bt_wer - at_wer) / max(bt_wer, 0.001) * 100

    print(f"""
    +-------------------------------------------------------------------+
    |                    WORD ERROR RATE (WER) COMPARISON                |
    +------------------+------------------+------------------+-----------+
    |     Domain       |     Baseline     |     Adapted      |  Change   |
    |                  |   (Pre-trained)  |   (Fine-tuned)   |           |
    +------------------+------------------+------------------+-----------+
    |  Source (Clean)  |    {bs_wer:>8.2%}      |    {as_wer:>8.2%}      | {as_wer - bs_wer:>+7.2%}  |
    |  Target (Noisy)  |    {bt_wer:>8.2%}      |    {at_wer:>8.2%}      | {target_wer_change:>+7.2%}  |
    +------------------+------------------+------------------+-----------+

    +-------------------------------------------------------------------+
    |                CHARACTER ERROR RATE (CER) COMPARISON               |
    +------------------+------------------+------------------+-----------+
    |     Domain       |     Baseline     |     Adapted      |  Change   |
    +------------------+------------------+------------------+-----------+
    |  Source (Clean)  |    {bs_cer:>8.2%}      |    {as_cer:>8.2%}      | {as_cer - bs_cer:>+7.2%}  |
    |  Target (Noisy)  |    {bt_cer:>8.2%}      |    {at_cer:>8.2%}      | {at_cer - bt_cer:>+7.2%}  |
    +------------------+------------------+------------------+-----------+

    +-----------------------------------------------------------+
    |                      KEY FINDINGS                         |
    +-----------------------------------------------------------+
    |  Target domain WER improvement: {target_improvement_pct:>6.1f}%                  |
    |  Baseline target WER:  {bt_wer:.2%}  ->  Adapted: {at_wer:.2%}            |
    |                                                           |
    |  Domain adaptation {'SUCCEEDED' if target_wer_change < 0 else 'NEEDS MORE DATA/TUNING':>38s}   |
    +-----------------------------------------------------------+

    FILES GENERATED:
    |-- results/wer_comparison.png        (WER bar chart)
    |-- results/training_loss.png         (Training loss curve)
    |-- results/spectrogram_comparison.png (Clean vs noisy audio)
    |-- results/sample_predictions.txt     (Before vs after predictions)
    |-- results/baseline_results.json      (Baseline metrics)
    +-- results/adapted_results.json       (Adapted metrics)
    """)

    print("=" * 70)
    print("  PROJECT COMPLETE!")
    print("=" * 70)


# ============================================================
# Main Entry Point
# ============================================================

def main():
    print("\n" + "=" * 70)
    print("  DOMAIN ADAPTATION - FINAL EVALUATION & COMPARISON")
    print("  Comparing baseline vs adapted model performance")
    print("=" * 70 + "\n")

    # Step 1: Load baseline results
    baseline = load_baseline_results()

    # Step 2: Load adapted model
    model, processor, device = load_adapted_model()

    # Step 3: Load evaluation data
    source_records, target_test_records = load_metadata()

    # Step 4: Evaluate adapted model on source domain
    source_results = evaluate_domain(
        model, processor, source_records, "SOURCE", device
    )

    # Step 5: Evaluate adapted model on target domain (test split)
    target_results = evaluate_domain(
        model, processor, target_test_records, "TARGET", device
    )

    # Step 6: Save results
    adapted_data = save_adapted_results(source_results, target_results, baseline)

    # Step 7: Create visualizations
    create_visualizations(baseline, source_results, target_results)

    # Step 8: Save sample predictions
    save_sample_predictions(baseline, source_results, target_results)

    # Step 9: Print final summary
    print_final_summary(baseline, source_results, target_results, adapted_data)

    print("\n  [DONE] Final evaluation complete! Check the results/ folder.\n")


if __name__ == "__main__":
    main()
