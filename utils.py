"""
utils.py - Helper functions for Domain Adaptation Framework
Contains audio processing, evaluation metrics, visualization, and inference utilities.
"""

import os
import numpy as np
import librosa
import soundfile as sf
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
from jiwer import wer, cer
import torch
import warnings
warnings.filterwarnings("ignore")


# ============================================================
# Audio Loading and Resampling
# ============================================================

def load_audio(path, sr=16000):
    """
    Load an audio file and resample to the target sample rate.

    Args:
        path: Path to the audio file (.wav, .flac, etc.)
        sr: Target sample rate (default 16000 for wav2vec2)

    Returns:
        numpy array of audio samples, sample rate
    """
    try:
        audio, orig_sr = librosa.load(path, sr=sr)
        return audio, sr
    except Exception as e:
        print(f"[ERROR] Failed to load audio from {path}: {e}")
        # Return 1 second of silence as fallback
        return np.zeros(sr, dtype=np.float32), sr


# ============================================================
# Noise and Augmentation Functions
# ============================================================

def add_noise(audio, noise_level=0.005):
    """
    Add Gaussian (white) noise to audio.

    Args:
        audio: numpy array of audio samples
        noise_level: standard deviation of noise (higher = more noise)

    Returns:
        Noisy audio as numpy array
    """
    noise = np.random.normal(0, noise_level, len(audio)).astype(np.float32)
    noisy_audio = audio + noise
    # Clip to valid range
    noisy_audio = np.clip(noisy_audio, -1.0, 1.0)
    return noisy_audio


def add_background_noise(audio, sr=16000, noise_level=0.01):
    """
    Add low-frequency rumble (simulating background/ambient noise).

    Args:
        audio: numpy array of audio samples
        sr: sample rate
        noise_level: intensity of background noise

    Returns:
        Audio with background noise added
    """
    duration = len(audio) / sr
    t = np.linspace(0, duration, len(audio), dtype=np.float32)

    # Mix of low-frequency sinusoids to simulate ambient hum
    bg_noise = (
        noise_level * 0.5 * np.sin(2 * np.pi * 50 * t) +   # 50 Hz hum
        noise_level * 0.3 * np.sin(2 * np.pi * 120 * t) +  # 120 Hz hum
        noise_level * 0.2 * np.random.normal(0, 1, len(audio))  # broadband
    ).astype(np.float32)

    noisy_audio = audio + bg_noise
    noisy_audio = np.clip(noisy_audio, -1.0, 1.0)
    return noisy_audio


def add_reverb(audio, sr=16000):
    """
    Add simple synthetic reverb effect by convolving with an exponential decay impulse response.

    Args:
        audio: numpy array of audio samples
        sr: sample rate

    Returns:
        Audio with reverb applied
    """
    try:
        # Create a simple impulse response (exponential decay)
        reverb_duration = 0.3  # seconds
        reverb_len = int(sr * reverb_duration)
        impulse_response = np.zeros(reverb_len, dtype=np.float32)
        impulse_response[0] = 1.0  # direct sound

        # Add decaying reflections
        for i in range(1, reverb_len):
            decay = np.exp(-4.0 * i / reverb_len)
            impulse_response[i] = decay * np.random.normal(0, 0.02)

        impulse_response = impulse_response.astype(np.float32)

        # Convolve audio with impulse response
        reverbed = np.convolve(audio, impulse_response, mode='same')

        # Normalize to prevent clipping
        max_val = np.max(np.abs(reverbed))
        if max_val > 0:
            reverbed = reverbed * (np.max(np.abs(audio)) / max_val)

        return reverbed.astype(np.float32)
    except Exception as e:
        print(f"[WARNING] Reverb failed: {e}, returning original audio")
        return audio


def add_speed_perturbation(audio, sr=16000, speed_factor=None):
    """
    Apply speed perturbation (makes speech faster or slower, changing pitch).

    Args:
        audio: numpy array of audio samples
        sr: sample rate
        speed_factor: speed multiplier (default: random between 0.9-1.1)

    Returns:
        Speed-perturbed audio, new effective sample rate
    """
    try:
        if speed_factor is None:
            speed_factor = np.random.uniform(0.9, 1.1)

        # Time-stretch without pitch preservation = speed change
        stretched = librosa.effects.time_stretch(audio, rate=speed_factor)
        return stretched.astype(np.float32), sr
    except Exception as e:
        print(f"[WARNING] Speed perturbation failed: {e}, returning original")
        return audio, sr


# ============================================================
# Evaluation Metrics
# ============================================================

def calculate_wer(reference, hypothesis):
    """
    Calculate Word Error Rate between reference and hypothesis.

    Args:
        reference: ground truth transcription string
        hypothesis: predicted transcription string

    Returns:
        WER as a float (0.0 = perfect, 1.0 = 100% error)
    """
    try:
        if not reference.strip():
            return 1.0 if hypothesis.strip() else 0.0
        if not hypothesis.strip():
            return 1.0
        return wer(reference.lower(), hypothesis.lower())
    except Exception as e:
        print(f"[WARNING] WER calculation error: {e}")
        return 1.0


def calculate_cer(reference, hypothesis):
    """
    Calculate Character Error Rate between reference and hypothesis.

    Args:
        reference: ground truth transcription string
        hypothesis: predicted transcription string

    Returns:
        CER as a float (0.0 = perfect, 1.0 = 100% error)
    """
    try:
        if not reference.strip():
            return 1.0 if hypothesis.strip() else 0.0
        if not hypothesis.strip():
            return 1.0
        return cer(reference.lower(), hypothesis.lower())
    except Exception as e:
        print(f"[WARNING] CER calculation error: {e}")
        return 1.0


# ============================================================
# Visualization Functions
# ============================================================

def plot_spectrogram(audio, sr, title="Spectrogram", save_path=None):
    """
    Plot and save a mel spectrogram of the audio.

    Args:
        audio: numpy array of audio samples
        sr: sample rate
        title: plot title
        save_path: path to save the image (None = don't save)
    """
    try:
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))

        # Compute mel spectrogram
        S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=8000)
        S_dB = librosa.power_to_db(S, ref=np.max)

        img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel',
                                        sr=sr, fmax=8000, ax=ax)
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        ax.set_title(title, fontsize=14)

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  [SAVED] Spectrogram -> {save_path}")

        plt.close(fig)
    except Exception as e:
        print(f"[WARNING] Spectrogram plot failed: {e}")


def plot_wer_comparison(baseline_source_wer, baseline_target_wer,
                        adapted_source_wer, adapted_target_wer,
                        save_path="results/wer_comparison.png"):
    """
    Create a grouped bar chart comparing WER before and after adaptation.
    """
    try:
        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(2)
        width = 0.3

        baseline_values = [baseline_source_wer * 100, baseline_target_wer * 100]
        adapted_values = [adapted_source_wer * 100, adapted_target_wer * 100]

        bars1 = ax.bar(x - width/2, baseline_values, width, label='Baseline (Pre-trained)',
                        color='#e74c3c', alpha=0.85, edgecolor='black')
        bars2 = ax.bar(x + width/2, adapted_values, width, label='Adapted (Fine-tuned)',
                        color='#2ecc71', alpha=0.85, edgecolor='black')

        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')

        ax.set_xlabel('Domain', fontsize=13)
        ax.set_ylabel('Word Error Rate (%)', fontsize=13)
        ax.set_title('Domain Adaptation: WER Comparison', fontsize=15, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(['Source (Clean)', 'Target (Noisy)'], fontsize=12)
        ax.legend(fontsize=11)
        ax.set_ylim(0, max(baseline_values + adapted_values) * 1.25)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  [SAVED] WER comparison chart -> {save_path}")
    except Exception as e:
        print(f"[WARNING] WER comparison plot failed: {e}")


def plot_training_loss(loss_history, save_path="results/training_loss.png"):
    """
    Plot training loss over steps.

    Args:
        loss_history: list of (step, loss) tuples or list of loss values
        save_path: path to save the plot
    """
    try:
        fig, ax = plt.subplots(figsize=(10, 5))

        if isinstance(loss_history[0], (list, tuple)):
            steps = [x[0] for x in loss_history]
            losses = [x[1] for x in loss_history]
        else:
            steps = list(range(1, len(loss_history) + 1))
            losses = loss_history

        ax.plot(steps, losses, 'b-o', linewidth=2, markersize=4, alpha=0.8,
                color='#3498db', label='Training Loss')
        ax.fill_between(steps, losses, alpha=0.1, color='#3498db')

        ax.set_xlabel('Training Step', fontsize=13)
        ax.set_ylabel('Loss', fontsize=13)
        ax.set_title('Fine-tuning Training Loss', fontsize=15, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  [SAVED] Training loss plot -> {save_path}")
    except Exception as e:
        print(f"[WARNING] Training loss plot failed: {e}")


def plot_spectrogram_comparison(clean_audio, noisy_audio, sr,
                                 save_path="results/spectrogram_comparison.png"):
    """
    Plot side-by-side spectrograms of clean vs noisy audio.
    """
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))

        # Clean spectrogram
        S_clean = librosa.feature.melspectrogram(y=clean_audio, sr=sr, n_mels=128, fmax=8000)
        S_clean_dB = librosa.power_to_db(S_clean, ref=np.max)
        img1 = librosa.display.specshow(S_clean_dB, x_axis='time', y_axis='mel',
                                         sr=sr, fmax=8000, ax=axes[0])
        axes[0].set_title('Clean Audio (Source Domain)', fontsize=13)
        fig.colorbar(img1, ax=axes[0], format='%+2.0f dB')

        # Noisy spectrogram
        S_noisy = librosa.feature.melspectrogram(y=noisy_audio, sr=sr, n_mels=128, fmax=8000)
        S_noisy_dB = librosa.power_to_db(S_noisy, ref=np.max)
        img2 = librosa.display.specshow(S_noisy_dB, x_axis='time', y_axis='mel',
                                         sr=sr, fmax=8000, ax=axes[1])
        axes[1].set_title('Noisy Audio (Target Domain)', fontsize=13)
        fig.colorbar(img2, ax=axes[1], format='%+2.0f dB')

        plt.suptitle('Spectrogram Comparison: Clean vs Noisy', fontsize=15, fontweight='bold')
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  [SAVED] Spectrogram comparison -> {save_path}")
    except Exception as e:
        print(f"[WARNING] Spectrogram comparison plot failed: {e}")


# ============================================================
# Transcription / Inference
# ============================================================

def transcribe(model, processor, audio_path, device="cpu"):
    """
    Transcribe a single audio file using a wav2vec2 model.

    Args:
        model: Wav2Vec2ForCTC model
        processor: Wav2Vec2Processor
        audio_path: path to audio file
        device: 'cpu' or 'cuda'

    Returns:
        Predicted transcription string
    """
    try:
        # Load audio at 16kHz
        audio, sr = load_audio(audio_path, sr=16000)

        # Process audio
        input_values = processor(
            audio, sampling_rate=16000, return_tensors="pt", padding=True
        ).input_values.to(device)

        # Run inference
        with torch.no_grad():
            logits = model(input_values).logits

        # Decode
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]

        return transcription.strip()
    except Exception as e:
        print(f"[ERROR] Transcription failed for {audio_path}: {e}")
        return ""


def transcribe_batch(model, processor, audio_paths, device="cpu", batch_size=4):
    """
    Transcribe a batch of audio files.

    Args:
        model: Wav2Vec2ForCTC model
        processor: Wav2Vec2Processor
        audio_paths: list of paths to audio files
        device: 'cpu' or 'cuda'
        batch_size: number of files to process at once

    Returns:
        List of predicted transcription strings
    """
    all_transcriptions = []

    for i in range(0, len(audio_paths), batch_size):
        batch_paths = audio_paths[i:i + batch_size]
        batch_audio = []

        for path in batch_paths:
            audio, _ = load_audio(path, sr=16000)
            batch_audio.append(audio)

        try:
            inputs = processor(
                batch_audio, sampling_rate=16000,
                return_tensors="pt", padding=True
            ).input_values.to(device)

            with torch.no_grad():
                logits = model(inputs).logits

            predicted_ids = torch.argmax(logits, dim=-1)
            transcriptions = processor.batch_decode(predicted_ids)
            all_transcriptions.extend([t.strip() for t in transcriptions])
        except Exception as e:
            print(f"[WARNING] Batch transcription failed, falling back to single: {e}")
            for path in batch_paths:
                all_transcriptions.append(transcribe(model, processor, path, device))

    return all_transcriptions


# ============================================================
# Utility Functions
# ============================================================

def get_device():
    """Get the best available device (CUDA GPU or CPU)."""
    if torch.cuda.is_available():
        device = "cuda"
        print(f"  [DEVICE] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("  [DEVICE] Using CPU (GPU not available)")
    return device


def ensure_dir(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def save_audio(audio, path, sr=16000):
    """Save audio array to a WAV file."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        sf.write(path, audio, sr)
    except Exception as e:
        print(f"[ERROR] Failed to save audio to {path}: {e}")


def get_audio_duration(audio, sr=16000):
    """Get duration of audio in seconds."""
    return len(audio) / sr


def print_separator(title="", char="=", width=70):
    """Print a formatted separator line."""
    if title:
        padding = (width - len(title) - 2) // 2
        print(f"\n{char * padding} {title} {char * padding}")
    else:
        print(char * width)
