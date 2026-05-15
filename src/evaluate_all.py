"""
Multi-model comparative evaluation script. Analyzes inference efficacy 
across Baseline, Transformer, and Hybrid architectures.
"""
import torch
import numpy as np
import os
import sys
import json
from torchvision import transforms, models
from sklearn.metrics import f1_score
import timm
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import PATHS

# FORCE EVALUATION ON CLEAN BASELINE DATA
TARGET_MODE = "classroom"
from src.utils import SpectrogramDataset

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_all():
    test_dir = os.path.join(PATHS[TARGET_MODE]["out"], "test")
    print(f"\n[INFO] Executing Multi-Model Evaluation on {TARGET_MODE.upper()} subset")
    
    # 1. EfficientNet Loader (With ImageNet Normalization)
    transform_eff = transforms.Compose([
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    ds_eff = SpectrogramDataset(test_dir, transform=transform_eff)
    loader_eff = torch.utils.data.DataLoader(ds_eff, batch_size=32, shuffle=False, num_workers=2)

    # 2. Sandbox Loader (No Normalization, matching sandbox training)
    ds_sandbox = SpectrogramDataset(test_dir, transform=None)
    loader_sandbox = torch.utils.data.DataLoader(ds_sandbox, batch_size=32, shuffle=False, num_workers=2)
    
    num_classes = len(ds_eff.classes)
    results = {}

    MODELS_TO_TEST = {
        "EfficientNet-V2-S": {"file": "models/efficientnet_classroom.pth", "type": "effnet"}, 
        "ViT-B/16": {"file": "models/sandbox/vit_baseline.pth", "type": "vit"},
        "CoAtNet-0": {"file": "models/sandbox/coatnet_baseline.pth", "type": "coatnet"}
    }

    for name, config in MODELS_TO_TEST.items():
        if not os.path.exists(config["file"]):
            print(f"[WARNING] Skipping evaluation for {name}; weight file unlocated.")
            continue
            
        print(f"[INFO] Processing inferences for {name}")
        
        # Route to the correct architecture
        if config["type"] == "effnet":
            model = models.efficientnet_v2_s()
            model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
            current_loader = loader_eff
        elif config["type"] == "vit":
            model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
            current_loader = loader_sandbox
        elif config["type"] == "coatnet":
            model = timm.create_model('coatnet_0_rw_224', pretrained=False, num_classes=num_classes)
            current_loader = loader_sandbox

        model.load_state_dict(torch.load(config["file"], map_location=device, weights_only=True))
        model = model.to(device)
        
        if torch.cuda.is_available(): 
            model = torch.compile(model, mode="reduce-overhead")
        model.eval()

        y_true, y_pred = [], []
        with torch.inference_mode():
            # Use the mathematically correct loader for the current model
            for inputs, labels in current_loader:
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs.to(device))
                _, preds = torch.max(outputs, 1)
                y_true.extend(labels.numpy())
                y_pred.extend(preds.cpu().numpy())

        score = f1_score(y_true, y_pred, average='macro', zero_division=0)
        results[name] = float(score)

    print("\n" + "="*50 + f"\n Comparative Architecture Efficacy ({TARGET_MODE.upper()})\n" + "="*50)
    for name, score in results.items(): 
        print(f" {name:<20} | Macro F1: {score:.4f} ({score*100:.2f}%)")
    print("="*50 + "\n")
    
    os.makedirs("results", exist_ok=True)
    with open(f"results/results_multimodel_{TARGET_MODE}.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__": 
    evaluate_all()