"""
Pre-execution validation script to ensure dataset integrity prior to training.
"""
import os
import sys
from collections import Counter
from config import MODE, PATHS

test_dir = os.path.join(PATHS[MODE]["out"], "test")
warnings = Counter()

print(f"[INFO] Initiating Pre-Flight Dataset Audit for Phase: {MODE.upper()}")
if not os.path.exists(test_dir):
    print(f"[ERROR] Processed test directory ({test_dir}) not found. Execute preprocess_data.py prior to validation.")
    sys.exit(1)

for key_folder in os.listdir(test_dir):
    folder_path = os.path.join(test_dir, key_folder)
    if os.path.isdir(folder_path):
        n_test = len([f for f in os.listdir(folder_path) if f.endswith('.npy')])
        if n_test < 15: 
            warnings[key_folder] = n_test

if not warnings: 
    print("[INFO] Dataset validation passed. All classes meet minimum sample thresholds.")
else:
    for k, v in warnings.items(): 
        print(f"[WARNING] Class '{k}' contains an insufficient test sample count ({v}).")