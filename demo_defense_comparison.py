"""
AcoustiGuard: Visual Defense Proof (Wayland/PipeWire Optimized)
Records one key, predicts, applies 2.5x Pink Noise, and saves a side-by-side PNG.
"""
import os
import time
import json
import torch
import numpy as np
import librosa
import sounddevice as sd
import matplotlib
matplotlib.use('Agg') # Prevents GUI crashes in Wayland compositors
import matplotlib.pyplot as plt
import colorednoise as cn
import torchvision.models as models
from torchvision import transforms

# --- MATHEMATICALLY PROVEN PARAMETERS ---
SR = 48000                 
N_FFT = 1024               
HOP_LENGTH = 256           
N_MELS = 128               
WINDOW_SAMPLES = 14400     
NOISE_MULTIPLIER = 2.5     
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_assets():
    print("[INFO] Loading EfficientNet-V2-S Weights...")
    map_path = "models/class_mapping.json"
    if not os.path.exists(map_path):
        print("[FATAL] Run 'python src/generate_mapping.py' first!")
        exit()
    with open(map_path) as f:
        mapping = json.load(f)
    classes = [k for k, v in sorted(mapping.items(), key=lambda x: x[1])]
        
    model = models.efficientnet_v2_s()
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(classes))
    
    try:
        # Security: weights_only=True prevents arbitrary code execution during deserialization
        state = torch.load("models/efficientnet_classroom.pth", map_location=DEVICE, weights_only=True)
        model.load_state_dict(state)
    except FileNotFoundError:
        print("[FATAL] 'efficientnet_classroom.pth' not found!")
        exit()
        
    model = model.to(DEVICE).eval()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return model, classes, normalize

def audio_to_tensor(audio_array, normalize):
    """Matches the 48kHz / 128-Mel Float32 pipeline exactly."""
    mel = librosa.feature.melspectrogram(y=audio_array, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=1.0, top_db=80)
    
    tensor = torch.tensor(mel_db, dtype=torch.float32)
    tensor = (tensor + 80.0) / 80.0
    tensor = torch.clamp(tensor, 0.0, 1.0)
    
    # Safe reshaping to avoid contiguous memory errors
    tensor = tensor.unsqueeze(0).unsqueeze(0)
    tensor = torch.nn.functional.interpolate(tensor, size=(224, 224), mode='bilinear', align_corners=False)
    tensor = tensor.squeeze(0).repeat(3, 1, 1)
    
    return mel_db, normalize(tensor).unsqueeze(0).to(DEVICE)

def run_demo():
    model, classes, normalize = load_assets()
    
    print("\n" + "="*50 + "\n ACOUSTIGUARD: STAGE DEMO \n" + "="*50)
    
    # --- MANUAL TRIGGER ---
    input("\n[READY] Press ENTER, then immediately strike your target key... ")
    print("\n[LIVE] <<< RECORDING FOR 3 SECONDS >>>")
    
    try:
        audio = sd.rec(int(3.0 * SR), samplerate=SR, channels=1, dtype='float32')
        sd.wait()
        audio = audio.flatten()
    except Exception:
        print("[WARN] Mic failed. Loading fallback 'demo_fallback.wav'...")
        try:
            audio, _ = librosa.load("demo_fallback.wav", sr=SR)
        except FileNotFoundError:
            print("[FATAL] Fallback audio not found.")
            return

    ambient_rms = np.sqrt(np.mean(audio[:int(0.5*SR)]**2) + 1e-12)
    onsets = librosa.onset.onset_detect(y=audio, sr=SR, delta=0.2, backtrack=True, units='samples')
    
    if len(onsets) == 0:
        print("[FAIL] No key detected. Hit it louder or adjust mic.")
        return

    onset = onsets[0]
    start = max(0, onset - int(0.05 * SR))
    end = min(len(audio), onset + int(0.25 * SR))
    keystroke = audio[start:end]
    
    if len(keystroke) < WINDOW_SAMPLES:
        keystroke = np.pad(keystroke, (0, WINDOW_SAMPLES - len(keystroke)))

    # 1. Baseline Attack
    disp_clean, tens_clean = audio_to_tensor(keystroke, normalize)
    with torch.no_grad():
        out_clean = torch.nn.functional.softmax(model(tens_clean), dim=1)[0]
        prob_c, idx_c = torch.max(out_clean, 0)

    # 2. Apply Defense (2.5x RMS Pink Noise)
    pink = cn.powerlaw_psd_gaussian(1, len(keystroke))
    pink_rms = np.sqrt(np.mean(pink**2) + 1e-12)
    masked = keystroke + (pink * (ambient_rms * NOISE_MULTIPLIER / pink_rms))
    
    # Strict Linear Scaling
    peak_masked = np.max(np.abs(masked))
    if peak_masked > 1.0: masked /= peak_masked

    disp_mask, tens_mask = audio_to_tensor(masked, normalize)
    with torch.no_grad():
        out_mask = torch.nn.functional.softmax(model(tens_mask), dim=1)[0]
        prob_m, idx_m = torch.max(out_mask, 0)

    # 3. Results
    print(f"\n[BASELINE] Pred: {classes[idx_c.item()].upper()} ({prob_c.item()*100:.1f}%)")
    print(f"[DEFENDED] Pred: {classes[idx_m.item()].upper()} ({prob_m.item()*100:.1f}%)")

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.imshow(disp_clean, aspect='auto', origin='lower', cmap='magma')
    ax1.set_title(f"BASELINE: {classes[idx_c.item()].upper()} ({prob_c.item()*100:.1f}%)")
    ax2.imshow(disp_mask, aspect='auto', origin='lower', cmap='magma')
    ax2.set_title(f"ACOUSTIGUARD: {classes[idx_m.item()].upper()} ({prob_m.item()*100:.1f}%)")
    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/defense_proof.png")
    plt.close()
    print("\n[SUCCESS] Figure saved as defense_proof.png")

if __name__ == "__main__":
    run_demo()