"""
AcoustiGuard: Class Mapping Generator
Run this once to sync folder names to PyTorch indices.
"""
import os
import json

# Change this to point to your exact training folders
dataset_dir = "data/processed/classroom/train" 

if not os.path.exists(dataset_dir):
    print(f"[ERROR] Path not found: {dataset_dir}")
    print("Please update 'dataset_dir' in the script to point to your key folders.")
else:
    classes = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
    mapping = {cls: i for i, cls in enumerate(classes)}
    
    os.makedirs("models", exist_ok=True)
    with open("models/class_mapping.json", "w") as f:
        json.dump(mapping, f, indent=2)
        
    print(f"[SUCCESS] Created models/class_mapping.json with {len(classes)} keys.")