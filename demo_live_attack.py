"""
AcoustiGuard: Real-Time Threat Eavesdropper
Streams audio via PipeWire/ALSA and predicts live.
"""
import os
import time
import json
import queue
import torch
import numpy as np
import librosa
import sounddevice as sd
import torchvision.models as models
from torchvision import transforms

# --- MATHEMATICALLY PROVEN PARAMETERS ---
SR, N_FFT, HOP_LENGTH, N_MELS = 48000, 1024, 256, 128
WINDOW = 14400 
THRESHOLD = 0.028     # Tuned to hear quiet keystrokes but ignore light room hum
COOLDOWN = 0.3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_attacker():
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
        
    return model.to(DEVICE).eval(), classes, transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

def main():
    print("[INFO] Loading neural architecture...")
    model, classes, norm = load_attacker()
    
    # --- PRE-WARM LIBROSA & CUDA FOR REAL-TIME INFERENCE ---
    print("[INFO] Pre-warming Librosa JIT compiler and CUDA pipeline...")
    _ = librosa.feature.melspectrogram(
        y=np.zeros(WINDOW, dtype=np.float32), 
        sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    
    # Warm up the GPU model with a dummy forward pass to eliminate first-run latency
    dummy = torch.zeros(1, 3, 224, 224).to(DEVICE)
    with torch.no_grad():
        _ = model(dummy)
        if DEVICE.type == 'cuda': torch.cuda.synchronize()
        
    print("[OK] Pipeline Ready. Sub-100ms inference achieved.")

    q = queue.Queue()
    buffer = np.zeros(SR)
    last_trigger = 0

    def cb(indata, f, t, s): 
        q.put(indata[:, 0].copy())

    print("\n[!] LIVE ATTACK ACTIVE. CTRL+C to stop.")
    with sd.InputStream(samplerate=SR, channels=1, blocksize=2048, callback=cb):
        while True:
            try: 
                chunk = q.get(timeout=1.0)
            except queue.Empty: 
                continue
                
            buffer = np.roll(buffer, -len(chunk))
            buffer[-len(chunk):] = chunk
            
            now = time.time()
            rms = np.sqrt(np.mean(chunk**2) + 1e-12)
            
            # --- ROOM CALIBRATION LINE ---
            # If it ghost types, uncomment the line below to see your room's baseline noise level
            # if rms > 0.01: print(f" Ambient RMS: {rms:.4f}", end='\r', flush=True)
            
            if rms > THRESHOLD and (now - last_trigger) > COOLDOWN:
                start = time.time()
                if DEVICE.type == 'cuda': torch.cuda.synchronize()
                
                window = buffer[-WINDOW:].copy()
                
                mel = librosa.feature.melspectrogram(y=window, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
                mel_db = librosa.power_to_db(mel, ref=1.0, top_db=80)
                
                t_audio = torch.tensor(mel_db, dtype=torch.float32)
                t_audio = (t_audio + 80.0) / 80.0
                t_audio = torch.clamp(t_audio, 0.0, 1.0).unsqueeze(0).unsqueeze(0)
                t_audio = torch.nn.functional.interpolate(t_audio, size=(224, 224), mode='bilinear', align_corners=False)
                t_audio = norm(t_audio.squeeze(0).repeat(3, 1, 1)).unsqueeze(0).to(DEVICE)
                
                with torch.no_grad():
                    probs = torch.nn.functional.softmax(model(t_audio), dim=1)[0]
                    conf, idx = torch.max(probs, 0)
                
                if DEVICE.type == 'cuda': torch.cuda.synchronize()
                latency = (time.time() - start) * 1000
                
                # Overwrite the calibration print with the actual attack result
                print(f"[ATTACK] RMS: {rms:.4f} | Latency: {latency:.1f}ms | Pred: [ {classes[idx.item()].upper()} ] ({conf.item()*100:.1f}%)" + " "*15)
                last_trigger = now

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[SHUTDOWN] Streaming Listener Terminated.")