#!/usr/bin/env python3
"""
Enhanced Voice Sample Preparation Script
Converts AAC/MP3/WAV to 16-bit Mono WAV with advanced audio enhancement:
- Noise reduction using spectral gating
- Silence trimming
- Audio normalization with headroom protection
- Support for multiple sample rates (16kHz or 22.05kHz)
"""

import os
import sys
import logging
from pathlib import Path
from typing import Tuple, Optional

import librosa
import soundfile as sf
import numpy as np
try:
    import noisereduce as nr
    HAS_NOISEREDUCE = True
except ImportError:
    HAS_NOISEREDUCE = False
    logging.warning("⚠ noisereduce not installed - noise reduction disabled")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_input_file(input_file: str) -> bool:
    """Validate that input file exists and is readable."""
    if not os.path.exists(input_file):
        logger.error(f"❌ Input file not found: {input_file}")
        return False
    if not os.path.isfile(input_file):
        logger.error(f"❌ Path is not a file: {input_file}")
        return False
    logger.info(f"✓ Input file found: {input_file}")
    return True


def enhance_audio(
    audio: np.ndarray,
    sr: int,
    trim_top_db: int = 30,
    noise_sample_duration: float = 0.5,
    enable_noise_reduction: bool = True
) -> np.ndarray:
    """
    Enhance audio quality with noise reduction and silence trimming.
    
    Args:
        audio: Audio time series
        sr: Sample rate
        trim_top_db: Threshold for silence trimming (dB)
        noise_sample_duration: Duration of noise sample from start (seconds)
        enable_noise_reduction: Whether to apply noise reduction
    
    Returns:
        Enhanced audio array
    """
    try:
        logger.info("📊 Enhancing audio...")
        
        # 1. Trim silence
        logger.info(f"🔪 Trimming silence (threshold: {trim_top_db}dB)")
        audio_trimmed, _ = librosa.effects.trim(audio, top_db=trim_top_db)
        trim_percent = (1 - len(audio_trimmed) / len(audio)) * 100
        logger.info(f"✓ Trimmed {trim_percent:.1f}% silence")
        
        # 2. Noise reduction
        if enable_noise_reduction and HAS_NOISEREDUCE:
            logger.info("🔇 Reducing background noise...")
            # Estimate noise from first sample
            noise_sample_len = int(noise_sample_duration * sr)
            noise_sample = audio_trimmed[:noise_sample_len]
            
            if len(noise_sample) > 0:
                audio_enhanced = nr.reduce_noise(
                    y=audio_trimmed,
                    sr=sr,
                    y_noise=noise_sample,
                    stationary=True,
                    prop_decrease=1.0
                )
                logger.info("✓ Noise reduction applied")
            else:
                logger.warning("⚠ Audio too short for noise estimation, skipping")
                audio_enhanced = audio_trimmed
        else:
            audio_enhanced = audio_trimmed
        
        # 3. Normalize with headroom
        logger.info("📈 Normalizing audio...")
        peak = np.max(np.abs(audio_enhanced))
        if peak > 0:
            audio_normalized = audio_enhanced / peak * 0.95  # 0.95 for 5% headroom
            logger.info(f"✓ Normalized (peak: {peak:.3f} → 0.95)")
        else:
            audio_normalized = audio_enhanced
            logger.warning("⚠ Silent audio detected")
        
        return audio_normalized
    
    except Exception as e:
        logger.error(f"❌ Audio enhancement failed: {e}")
        logger.warning("⚠ Proceeding without enhancement")
        return audio


def prepare_voice_sample(
    input_file: str,
    output_file: str,
    target_sr: int = 16000,
    enhance: bool = True,
    verbose: bool = True
) -> Tuple[bool, Optional[str]]:
    """
    Convert and enhance audio to 16-bit Mono WAV at specified sample rate.
    
    Args:
        input_file: Path to input audio (AAC, MP3, WAV, etc.)
        output_file: Path to output WAV file
        target_sr: Target sample rate (16000 or 22050 Hz)
        enhance: Apply audio enhancement (noise reduction, trimming)
        verbose: Print detailed info
    
    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        # Validate input
        if not validate_input_file(input_file):
            return False, "Input file validation failed"
        
        # Validate sample rate
        if target_sr not in [16000, 22050]:
            logger.warning(f"⚠ Unusual sample rate {target_sr}Hz, proceeding anyway")
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"✓ Created output directory: {output_dir}")
        
        logger.info(f"Loading audio from: {input_file}")
        
        # 1. Load audio (automatically handles .aac if ffmpeg is installed)
        # sr=None preserves original sampling rate for initial load
        audio, sr_original = librosa.load(input_file, sr=None, mono=True)
        logger.info(f"✓ Loaded audio: original_sr={sr_original}Hz, duration={len(audio)/sr_original:.2f}s")
        
        # 2. Enhance audio (noise reduction, trim silence)
        if enhance:
            # First resample to target for enhancement (faster processing)
            if sr_original != target_sr:
                logger.info(f"Resampling to {target_sr}Hz for enhancement...")
                audio_for_enhance = librosa.resample(y=audio, orig_sr=sr_original, target_sr=target_sr)
            else:
                audio_for_enhance = audio
            
            audio_enhanced = enhance_audio(audio_for_enhance, target_sr, enable_noise_reduction=True)
        else:
            # Just resample without enhancement
            if sr_original != target_sr:
                logger.info(f"Resampling from {sr_original}Hz → {target_sr}Hz")
                audio_enhanced = librosa.resample(y=audio, orig_sr=sr_original, target_sr=target_sr)
            else:
                audio_enhanced = audio
                logger.info(f"No resampling needed (already {target_sr}Hz)")
        
        # 3. Final normalization (safety check)
        peak = np.max(np.abs(audio_enhanced))
        if peak > 1.0:
            audio_enhanced = audio_enhanced / peak * 0.95
            logger.info(f"⚠ Clipping prevention: normalized peak from {peak:.3f} → 0.95")
        
        # 4. Save as 16-bit PCM WAV
        logger.info(f"Saving to: {output_file}")
        sf.write(output_file, audio_enhanced, target_sr, subtype='PCM_16')
        
        # 5. Verify output
        file_size = os.path.getsize(output_file) / 1024  # KB
        verify_audio, verify_sr = sf.read(output_file)
        logger.info(f"✓ Saved successfully: {output_file}")
        logger.info(f"  - Format: 16-bit Mono PCM WAV")
        logger.info(f"  - Sample rate: {verify_sr}Hz")
        logger.info(f"  - Duration: {len(verify_audio)/verify_sr:.2f}s")
        logger.info(f"  - File size: {file_size:.1f}KB")
        
        return True, f"Success: {output_file} is ready for cloning."
    
    except FileNotFoundError as e:
        msg = f"❌ File not found: {e}"
        logger.error(msg)
        return False, msg
    except Exception as e:
        msg = f"❌ Error processing audio: {type(e).__name__}: {e}"
        logger.error(msg)
        return False, msg


def batch_prepare_samples(
    input_file: str,
    output_dir: str,
    sample_rates: list = None,
    enhance: bool = True
) -> None:
    """
    Convert and enhance audio to multiple sample rates for flexibility.
    
    Args:
        input_file: Path to input audio file
        output_dir: Directory to save converted files
        sample_rates: List of target sample rates (default: [16000, 22050])
        enhance: Apply audio enhancement (noise reduction, trimming)
    """
    if sample_rates is None:
        sample_rates = [16000, 22050]
    
    os.makedirs(output_dir, exist_ok=True)
    base_name = Path(input_file).stem
    
    logger.info(f"\n{'='*70}")
    logger.info(f"Batch Processing: {base_name}")
    logger.info(f"Enhancement: {'Enabled ✓' if enhance else 'Disabled ✗'}")
    logger.info(f"{'='*70}")
    
    for sr in sample_rates:
        sr_name = f"{sr//1000}k" if sr >= 1000 else f"{sr}Hz"
        output_file = os.path.join(output_dir, f"{base_name}_{sr_name}.wav")
        
        logger.info(f"\n--- Processing {sr}Hz ---")
        success, message = prepare_voice_sample(input_file, output_file, target_sr=sr, enhance=enhance)
        if success:
            logger.info(f"✓ {message}")
        else:
            logger.error(f"✗ {message}")


if __name__ == "__main__":
    # Ask user for input file
    print("\n" + "="*70)
    print("🎤 Voice Sample Resampler")
    print("="*70)
    INPUT_FILE = input("\n📁 Enter the path to the audio file you want to resample:\n> ").strip().strip('"').strip("'")
    
    if not INPUT_FILE:
        logger.error("❌ No file path provided. Exiting.")
        sys.exit(1)
    
    # Output directory
    OUTPUT_DIR = r"D:\Desktop\CS\CS\fyp-voice-clone\phase1_baseline\Real-Time-Voice-Cloning\re_sample"
    
    # Convert to both 16kHz and 22.05kHz for maximum compatibility
    batch_prepare_samples(INPUT_FILE, OUTPUT_DIR, sample_rates=[16000, 22050])
    
    logger.info(f"\n{'='*70}")
    logger.info("✓ All conversions completed!")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info(f"{'='*70}\n")
