"""
Audio preprocessing module for onset detection and continuous spectrogram extraction.
"""
import matplotlib
matplotlib.use('Agg')
import os
import librosa
import numpy as np
import warnings
import shutil
from tqdm import tqdm
from config import MODE, PATHS, TIER_MAP

warnings.filterwarnings('ignore', category=UserWarning)

CANON_MAP = {";": "semicolon", "'": "apostrophe", "[": "bracketleft", "]": "bracketright", 
             "\\": "backslash", "`": "backtick", "-": "hyphen", "=": "equals", 
             ",": "comma", ".": "period", "/": "slash"}

def extract_and_split(audio_path, key_label):
    """Extracts keystroke events from a continuous audio stream and partitions into train/test sets."""
    y, sr = librosa.load(audio_path, sr=None)
    b = int(10 * sr)
    y_core = y[b:-b] if len(y) > 2*b else y
    
    onsets_core = librosa.onset.onset_detect(y=y_core, sr=sr, delta=0.2, backtrack=True, units='samples')
    
    if len(onsets_core) == 0: 
        return
        
    onsets = onsets_core + (b if len(y) > 2*b else 0)
    split_idx = int(len(onsets) * 0.8)
    
    def save_specs(onset_list, sub):
        target = os.path.join(PATHS[MODE]["out"], sub, key_label)
        os.makedirs(target, exist_ok=True)
        for i, onset in enumerate(onset_list):
            clip = y[max(0, onset - int(sr*0.05)) : min(len(y), onset + int(sr*0.25))]
            if len(clip) == 0: 
                continue
            
            mel = librosa.feature.melspectrogram(y=clip, sr=sr, n_fft=1024, hop_length=256)
            mel_db = librosa.power_to_db(mel, ref=1.0, top_db=80)
            np.save(os.path.join(target, f"{i}.npy"), mel_db.astype(np.float32))

    save_specs(onsets[:split_idx], "train")
    save_specs(onsets[split_idx:], "test")

if __name__ == "__main__":
    raw_dir = PATHS[MODE]["raw"]
    out_dir = PATHS[MODE]["out"]
    
    if os.path.exists(out_dir): 
        shutil.rmtree(out_dir)
        
    print(f"[INFO] Initializing preprocessing pipeline for mode: {MODE.upper()}")
    files = [f for f in os.listdir(raw_dir) if f.endswith('.wav')]
    valid_keys = set(CANON_MAP.keys()).union(set(CANON_MAP.values())).union(set(sum(TIER_MAP.values(), [])))
    
    for f in tqdm(files, desc="Processing audio files"):
        base = os.path.splitext(f)[0].lower()
        parts = base.split('_')
        found_keys = [p for p in parts if p in valid_keys]
        raw_label = found_keys[0] if found_keys else parts[-1]
        extract_and_split(os.path.join(raw_dir, f), CANON_MAP.get(raw_label, raw_label))