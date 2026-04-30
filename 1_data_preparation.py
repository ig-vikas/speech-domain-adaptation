"""
1_data_preparation.py - Data Preparation for Domain Adaptation
================================================================
This script:
    1. Loads clean speech from local data/source/
    2. Reads transcripts from existing data/metadata.csv (source rows)
    3. Creates target domain audio by adding Gaussian noise only
    4. Saves noisy audio to data/target/
    5. Rewrites metadata CSV with source and target records

Usage: python 1_data_preparation.py
"""

import os
import sys
import csv
import warnings
warnings.filterwarnings("ignore")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    add_noise, load_audio, save_audio, get_audio_duration,
    print_separator, ensure_dir, plot_spectrogram_comparison
)

# ============================================================
# Configuration
# ============================================================

NUM_SAMPLES = 80          # Number of samples to use
SAMPLE_RATE = 16000       # wav2vec2 expects 16kHz
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
SOURCE_DIR = os.path.join(DATA_DIR, "source")
TARGET_DIR = os.path.join(DATA_DIR, "target")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
METADATA_PATH = os.path.join(DATA_DIR, "metadata.csv")


def load_local_source_data():
    """Load source-domain audio and transcripts from local files only."""
    print_separator("STEP 1: Loading Local Source Audio")

    if not os.path.isdir(SOURCE_DIR):
        print(f"  [ERROR] Source directory not found: {SOURCE_DIR}")
        print("  [INFO] Add clean .wav files to data/source and rerun.")
        sys.exit(1)

    if not os.path.exists(METADATA_PATH):
        print(f"  [ERROR] Metadata file not found: {METADATA_PATH}")
        print("  [INFO] This local-only pipeline needs existing source transcripts.")
        sys.exit(1)

    transcript_map = {}
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("domain") == "source":
                file_id = row.get("file_id", "").strip()
                transcript = row.get("transcript", "").upper().strip()
                if file_id and transcript:
                    transcript_map[file_id] = transcript

    if not transcript_map:
        print("  [ERROR] No source transcripts found in metadata.csv")
        print("  [INFO] Ensure metadata has rows where domain=source.")
        sys.exit(1)

    wav_files = sorted(
        [name for name in os.listdir(SOURCE_DIR) if name.lower().endswith(".wav")]
    )
    if not wav_files:
        print(f"  [ERROR] No .wav files found in {SOURCE_DIR}")
        sys.exit(1)

    selected_wav_files = wav_files[:NUM_SAMPLES]
    print(f"  [INFO] Found {len(wav_files)} source audio files")
    print(f"  [INFO] Using {len(selected_wav_files)} files for this run\n")

    samples = []
    source_records = []
    total_duration = 0.0
    skipped_without_transcript = 0

    for i, wav_name in enumerate(selected_wav_files):
        file_id = os.path.splitext(wav_name)[0]
        transcript = transcript_map.get(file_id, "")
        if not transcript:
            skipped_without_transcript += 1
            continue

        file_path = os.path.join(SOURCE_DIR, wav_name)
        audio, _ = load_audio(file_path, sr=SAMPLE_RATE)
        duration = get_audio_duration(audio, SAMPLE_RATE)
        total_duration += duration

        samples.append({
            "audio": audio,
            "text": transcript,
            "source_file_id": file_id,
        })
        source_records.append({
            "file_path": file_path,
            "transcript": transcript,
            "domain": "source",
            "duration": round(duration, 2),
            "file_id": file_id,
        })

        if (i + 1) % 20 == 0:
            print(f"  [PROGRESS] Loaded {i + 1}/{len(selected_wav_files)} source files")

    if skipped_without_transcript > 0:
        print(
            f"  [WARNING] Skipped {skipped_without_transcript} files without source transcripts"
        )

    if not samples:
        print("  [ERROR] No usable source samples were loaded")
        print("  [INFO] Confirm source file names match source file_id values in metadata.csv.")
        sys.exit(1)

    print(f"\n  [OK] Loaded {len(source_records)} source samples from local files")
    print(f"  [OK] Total duration: {total_duration:.1f}s ({total_duration/60:.1f} min)")
    return samples, source_records


def process_target_domain(samples):
    """Create target domain by adding Gaussian noise only."""
    print_separator("STEP 2: Processing Target Domain (Noise Only)")
    ensure_dir(TARGET_DIR)

    target_records = []
    total_duration = 0.0
    noise_levels = [0.005, 0.008, 0.010, 0.012, 0.015]
    clean_sample_audio = None
    noisy_sample_audio = None

    for i, sample in enumerate(samples):
        audio = sample["audio"].copy()
        if i == 0:
            clean_sample_audio = audio.copy()

        noise_level = noise_levels[i % len(noise_levels)]
        noisy_audio = add_noise(audio, noise_level=noise_level)

        if i == 0:
            noisy_sample_audio = noisy_audio.copy()

        source_file_id = sample.get("source_file_id", f"source_{i:04d}")
        if source_file_id.startswith("source_"):
            file_id = "target_" + source_file_id[len("source_"):]
        else:
            file_id = f"target_{i:04d}"
        file_path = os.path.join(TARGET_DIR, f"{file_id}.wav")
        save_audio(noisy_audio, file_path, SAMPLE_RATE)
        duration = get_audio_duration(noisy_audio, SAMPLE_RATE)
        total_duration += duration
        target_records.append({
            "file_path": file_path, "transcript": sample["text"],
            "domain": "target", "duration": round(duration, 2), "file_id": file_id
        })
        if (i + 1) % 20 == 0:
            print(f"  [PROGRESS] Processed {i + 1}/{len(samples)} target samples")

    print(f"\n  [OK] Target domain: {len(target_records)} files")
    print(f"  [OK] Total duration: {total_duration:.1f}s ({total_duration/60:.1f} min)")

    if clean_sample_audio is not None and noisy_sample_audio is not None:
        print("\n  [INFO] Generating spectrogram comparison...")
        ensure_dir(RESULTS_DIR)
        plot_spectrogram_comparison(
            clean_sample_audio, noisy_sample_audio, SAMPLE_RATE,
            save_path=os.path.join(RESULTS_DIR, "spectrogram_comparison.png")
        )
    return target_records


def save_metadata(source_records, target_records):
    """Save metadata CSV."""
    print_separator("STEP 3: Saving Metadata")
    all_records = source_records + target_records
    with open(METADATA_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["file_path", "transcript", "domain", "duration", "file_id"])
        writer.writeheader()
        writer.writerows(all_records)
    print(f"  [OK] Metadata saved to {METADATA_PATH}")
    print(f"  [OK] Total records: {len(all_records)}")


def print_summary(source_records, target_records):
    """Print a summary of the prepared data."""
    print_separator("DATA PREPARATION SUMMARY")
    src_dur = sum(r["duration"] for r in source_records)
    tgt_dur = sum(r["duration"] for r in target_records)
    print(f"""
    ==================================================
    SOURCE DOMAIN (Clean Speech)
    ==================================================
    Files:       {len(source_records)}
    Duration:    {src_dur:.1f}s ({src_dur/60:.1f} min)
    Directory:   {SOURCE_DIR}

    ==================================================
    TARGET DOMAIN (Noisy Speech)
    ==================================================
    Files:       {len(target_records)}
    Duration:    {tgt_dur:.1f}s ({tgt_dur/60:.1f} min)
    Directory:   {TARGET_DIR}

    ==================================================
    TOTAL
    ==================================================
    Total files: {len(source_records) + len(target_records)}
    Total time:  {(src_dur + tgt_dur):.1f}s ({(src_dur + tgt_dur)/60:.1f} min)
    Metadata:    {METADATA_PATH}
    ==================================================
    """)


def main():
    print("\n" + "=" * 70)
    print("  DOMAIN ADAPTATION - DATA PREPARATION")
    print("  Preparing target data from local source audio (noise only)")
    print("=" * 70 + "\n")

    samples, source_records = load_local_source_data()
    target_records = process_target_domain(samples)
    save_metadata(source_records, target_records)
    print_summary(source_records, target_records)

    print("\n  [DONE] Data preparation complete!")
    print("  [NEXT] Run: python 2_baseline_evaluation.py\n")


if __name__ == "__main__":
    main()
