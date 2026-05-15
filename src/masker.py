"""
Acoustic masking module. Generates and scales 1/f Pink Noise dynamically 
to 2.5x the ambient RMS of the target audio file.
"""
import os
import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm
import colorednoise as cn
import hashlib
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import PATHS, MODE

def apply_adaptive_masking(audio_path, output_path):
    y, sr = librosa.load(audio_path, sr=None)
    b = int(10 * sr)
    
    if len(y) > 2 * b: 
        ambient = np.concatenate([y[:b], y[-b:]])
        y_core = y[b:-b]
    else: 
        ambient, y_core = y, y
        
    noise_floor_rms = np.sqrt(np.mean(ambient**2))
    
    # Environment-based salt ensures deterministic but secure PRNG initialization
    SALT = os.environ.get('ACOUSTIGUARD_SALT', 'thesis_run')
    rng_seed = int(hashlib.md5((SALT + os.path.basename(audio_path)).encode()).hexdigest(), 16) % (2**32)
    pink_noise = cn.powerlaw_psd_gaussian(1, len(y_core), random_state=rng_seed) 
    
    target_noise_rms = noise_floor_rms * 2.5 
    current_pink_noise_rms = np.sqrt(np.mean(pink_noise**2)) + 1e-8
    scaled_noise = pink_noise * (target_noise_rms / current_pink_noise_rms)
    
    y_masked_core = y_core + scaled_noise
    y_masked = np.concatenate([y[:b], y_masked_core, y[-b:]]) if len(y) > 2 * b else y_masked_core
    
    # Peak normalization implemented as a linear safety scale to prevent overflow.
    peak = np.max(np.abs(y_masked))
    if peak > 1.0: 
        y_masked /= peak
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sf.write(output_path, y_masked, sr)

if __name__ == "__main__":
    if "home" in MODE:
        src_dir = PATHS["home"]["raw"]
        dst_dir = PATHS["home_masked"]["raw"]
    else:
        src_dir = PATHS["classroom"]["raw"]
        dst_dir = PATHS["masked"]["raw"]
    print("[INFO] Initializing adaptive 1/f Pink Noise application sequence.")
    for f in tqdm([f for f in os.listdir(src_dir) if f.endswith('.wav')]):
        apply_adaptive_masking(os.path.join(src_dir, f), os.path.join(dst_dir, f))