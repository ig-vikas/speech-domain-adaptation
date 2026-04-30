"""
3_fine_tune.py - Fine-tune wav2vec2 on Target Domain (Noisy Speech)
====================================================================
This script:
  1. Loads pretrained wav2vec2-base-960h
  2. Prepares target domain data (80% train, 20% test split)
  3. Creates a Dataset class for audio processing
  4. Fine-tunes using HuggingFace Trainer on the noisy target domain
  5. Saves the adapted model to adapted_model/
  6. Plots training loss

Usage: python 3_fine_tune.py
"""

import os
import sys
import csv
import json
import torch
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    load_audio, get_device, print_separator, ensure_dir,
    plot_training_loss
)

# ============================================================
# Configuration
# ============================================================

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
METADATA_PATH = os.path.join(DATA_DIR, "metadata.csv")
MODEL_OUTPUT_DIR = os.path.join(PROJECT_DIR, "adapted_model")

MODEL_NAME = "facebook/wav2vec2-base-960h"
SAMPLE_RATE = 16000

# Training hyperparameters (small for laptop-friendly training)
NUM_EPOCHS = 5
BATCH_SIZE = 2
LEARNING_RATE = 1e-4
WARMUP_STEPS = 100
SAVE_STEPS = 50
LOGGING_STEPS = 10
TRAIN_SPLIT = 0.8   # 80% train, 20% test


# ============================================================
# Dataset Class
# ============================================================

class SpeechDataset(torch.utils.data.Dataset):
    """
    Custom Dataset for loading audio files and preparing them for wav2vec2 fine-tuning.
    """

    def __init__(self, records, processor, sample_rate=16000):
        """
        Args:
            records: list of dicts with 'file_path' and 'transcript'
            processor: Wav2Vec2Processor
            sample_rate: target sample rate
        """
        self.records = records
        self.processor = processor
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        file_path = record["file_path"]
        transcript = record["transcript"].upper().strip()

        # Load audio
        audio, sr = load_audio(file_path, sr=self.sample_rate)

        # Process audio through wav2vec2 processor
        input_values = self.processor(
            audio, sampling_rate=self.sample_rate, return_tensors="pt", padding=False
        ).input_values[0]

        # Encode transcript to label IDs using the tokenizer
        labels = self.processor.tokenizer(transcript, return_tensors="pt").input_ids[0]

        return {
            "input_values": input_values,
            "labels": labels
        }


# ============================================================
# Data Collator
# ============================================================

class DataCollatorCTCWithPadding:
    """
    Data collator that dynamically pads the inputs and labels for CTC training.
    """

    def __init__(self, processor, padding=True):
        self.processor = processor
        self.padding = padding

    def __call__(self, features):
        # Separate input_values and labels
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        # Pad audio inputs
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt"
        )

        # Pad labels using the tokenizer
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=self.padding,
            return_tensors="pt"
        )

        # Replace padding tokens in labels with -100 so they are ignored by CTC loss
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        batch["labels"] = labels
        return batch


# ============================================================
# Custom Trainer Callback for Loss Tracking
# ============================================================

class LossTrackingCallback:
    """Callback to track training loss at each logging step."""

    def __init__(self):
        self.losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.losses.append((state.global_step, logs["loss"]))
            print(f"    Step {state.global_step:>4d} | Loss: {logs['loss']:.4f}")


# ============================================================
# Functions
# ============================================================

def load_target_metadata():
    """Load target domain metadata and split into train/test."""
    print_separator("Loading Target Domain Data")

    if not os.path.exists(METADATA_PATH):
        print(f"  [ERROR] Metadata not found: {METADATA_PATH}")
        print("  [INFO] Please run 1_data_preparation.py first!")
        sys.exit(1)

    target_records = []
    with open(METADATA_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["domain"] == "target":
                target_records.append(row)

    print(f"  [OK] Loaded {len(target_records)} target domain samples")

    # Shuffle and split
    np.random.seed(42)
    indices = np.random.permutation(len(target_records))
    split_idx = int(len(target_records) * TRAIN_SPLIT)

    train_records = [target_records[i] for i in indices[:split_idx]]
    test_records = [target_records[i] for i in indices[split_idx:]]

    print(f"  [OK] Train set: {len(train_records)} samples ({TRAIN_SPLIT:.0%})")
    print(f"  [OK] Test set:  {len(test_records)} samples ({1 - TRAIN_SPLIT:.0%})")

    # Save test split indices for later evaluation
    test_split_path = os.path.join(DATA_DIR, "test_split.json")
    test_file_ids = [r["file_id"] for r in test_records]
    with open(test_split_path, 'w') as f:
        json.dump({"test_file_ids": test_file_ids}, f, indent=2)
    print(f"  [OK] Test split saved to {test_split_path}")

    return train_records, test_records


def load_model_and_processor():
    """Load the pretrained wav2vec2 model for fine-tuning."""
    print_separator("Loading Pretrained Model for Fine-tuning")

    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    print(f"  [INFO] Loading model: {MODEL_NAME}")
    print("  [INFO] This may take a minute...\n")

    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)

    # Freeze the feature extractor (only fine-tune the transformer layers)
    # This is important for small datasets - prevents overfitting
    model.freeze_feature_encoder()
    print("  [OK] Feature encoder frozen (only transformer layers will be trained)")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [OK] Total parameters:     {total_params:>12,}")
    print(f"  [OK] Trainable parameters: {trainable_params:>12,}")
    print(f"  [OK] Frozen parameters:    {total_params - trainable_params:>12,}")

    return model, processor


def fine_tune(model, processor, train_records, test_records, device):
    """Run fine-tuning using HuggingFace Trainer."""
    print_separator("Starting Fine-tuning")

    from transformers import TrainingArguments, Trainer

    # Create datasets
    print("  [INFO] Preparing datasets...")
    train_dataset = SpeechDataset(train_records, processor, SAMPLE_RATE)
    eval_dataset = SpeechDataset(test_records, processor, SAMPLE_RATE)

    print(f"  [OK] Train dataset: {len(train_dataset)} samples")
    print(f"  [OK] Eval dataset:  {len(eval_dataset)} samples")

    # Create data collator
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(MODEL_OUTPUT_DIR, "checkpoints"),
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        eval_strategy="epoch",
        num_train_epochs=NUM_EPOCHS,
        fp16=torch.cuda.is_available(),  # Use FP16 only if GPU available
        gradient_checkpointing=True,
        save_steps=SAVE_STEPS,
        logging_steps=LOGGING_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=0.005,
        warmup_steps=WARMUP_STEPS,
        save_total_limit=2,
        push_to_hub=False,
        report_to="none",  # Disable wandb/tensorboard
        load_best_model_at_end=False,
        dataloader_num_workers=0,  # Avoid multiprocessing issues on Windows
        remove_unused_columns=False,
    )

    print(f"\n  Training Configuration:")
    print(f"  {'-'*40}")
    print(f"  Epochs:         {NUM_EPOCHS}")
    print(f"  Batch size:     {BATCH_SIZE}")
    print(f"  Learning rate:  {LEARNING_RATE}")
    print(f"  Warmup steps:   {WARMUP_STEPS}")
    print(f"  Device:         {device}")
    print(f"  {'-'*40}\n")

    # Create loss tracking callback
    from transformers import TrainerCallback

    class LossLogger(TrainerCallback):
        def __init__(self):
            self.losses = []

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs:
                self.losses.append((state.global_step, logs["loss"]))
                print(f"    Step {state.global_step:>4d} | Loss: {logs['loss']:.4f}")

    loss_logger = LossLogger()

    # Create Trainer
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
        callbacks=[loss_logger],
    )

    # Start training
    print("  [INFO] Starting training... (this may take several minutes)\n")
    print(f"  {'='*50}")
    print(f"  TRAINING LOG")
    print(f"  {'='*50}")

    train_result = trainer.train()

    print(f"\n  {'='*50}")
    print(f"  TRAINING COMPLETE")
    print(f"  {'='*50}")
    print(f"  Total training time: {train_result.metrics.get('train_runtime', 0):.1f}s")
    print(f"  Final loss: {train_result.metrics.get('train_loss', 'N/A')}")

    return trainer, loss_logger.losses, train_result


def save_model(trainer, processor, loss_history):
    """Save the fine-tuned model, processor, and training history."""
    print_separator("Saving Fine-tuned Model")

    ensure_dir(MODEL_OUTPUT_DIR)

    # Save model and processor
    trainer.save_model(MODEL_OUTPUT_DIR)
    processor.save_pretrained(MODEL_OUTPUT_DIR)
    print(f"  [OK] Model saved to {MODEL_OUTPUT_DIR}")

    # Save training loss history
    loss_path = os.path.join(RESULTS_DIR, "training_loss.json")
    ensure_dir(RESULTS_DIR)
    with open(loss_path, 'w') as f:
        json.dump({"loss_history": loss_history}, f, indent=2)
    print(f"  [OK] Training loss history saved to {loss_path}")

    # Plot training loss
    if loss_history:
        plot_training_loss(
            loss_history,
            save_path=os.path.join(RESULTS_DIR, "training_loss.png")
        )

    return loss_path


# ============================================================
# Main Entry Point
# ============================================================

def main():
    print("\n" + "=" * 70)
    print("  DOMAIN ADAPTATION - FINE-TUNING")
    print("  Fine-tuning wav2vec2 on target domain (noisy speech)")
    print("=" * 70 + "\n")

    # Step 1: Load target domain data
    train_records, test_records = load_target_metadata()

    # Step 2: Load pretrained model
    model, processor = load_model_and_processor()

    # Step 3: Get device
    device = get_device()
    model = model.to(device)

    # Step 4: Fine-tune
    trainer, loss_history, train_result = fine_tune(
        model, processor, train_records, test_records, device
    )

    # Step 5: Save model and results
    save_model(trainer, processor, loss_history)

    # Print summary
    print_separator("FINE-TUNING SUMMARY")
    print(f"""
    {'='*50}
    TRAINING COMPLETE
    {'='*50}
    Model:           {MODEL_NAME}
    Epochs:          {NUM_EPOCHS}
    Batch size:      {BATCH_SIZE}
    Learning rate:   {LEARNING_RATE}
    Train samples:   {len(train_records)}
    Test samples:    {len(test_records)}
    Final loss:      {train_result.metrics.get('train_loss', 'N/A')}
    Training time:   {train_result.metrics.get('train_runtime', 0):.1f}s
    Model saved to:  {MODEL_OUTPUT_DIR}
    {'='*50}
    """)

    print("\n  [DONE] Fine-tuning complete!")
    print("  [NEXT] Run: python 4_final_evaluation.py\n")


if __name__ == "__main__":
    main()
