"""
Model evaluation module. Generates classification metrics, confusion matrices, 
and executes a paired bootstrap simulation for statistical confidence intervals.
"""
import matplotlib
matplotlib.use('Agg')
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import json
import sys
import random
from torchvision import transforms, models
from sklearn.metrics import classification_report, f1_score, confusion_matrix
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import MODE, PATHS, TIER_MAP
from src.utils import SpectrogramDataset

torch.backends.cudnn.benchmark = False

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate():
    test_dir = os.path.join(PATHS[MODE]["out"], "test")
    if "masked" in MODE:
        base_mode = MODE.replace("_masked", "").replace("masked", "classroom")
        model_name = f"models/efficientnet_{base_mode}.pth"
    else:
        model_name = f"models/efficientnet_{MODE}.pth"
    print(f"[INFO] Executing evaluation for mode: {MODE.upper()} using weight state: {model_name}")
    
    transform = transforms.Compose([
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    ds = SpectrogramDataset(test_dir, transform=transform)
    num_classes = len(ds.classes)
    loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False, num_workers=2)

    model = models.efficientnet_v2_s()
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(torch.load(model_name, map_location=device, weights_only=True))
    model = model.to(device).eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs.to(device))
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())

    print("\n" + classification_report(y_true, y_pred, target_names=ds.classes, zero_division=0))
    
    # Generate and save confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, xticklabels=ds.classes, yticklabels=ds.classes)
    plt.title(f"Confusion Matrix - {MODE.upper()} (EfficientNet)")
    os.makedirs("figures", exist_ok=True)
    plt.savefig(f"figures/conf_matrix_{MODE}.png", dpi=300)
    plt.close()

    # Calculate spatial tier metrics
    print("\n[INFO] Spatial Acoustics Tier Report")
    tier_results = {}
    for t_name, t_keys in TIER_MAP.items():
        idx = [i for i, name in enumerate(ds.classes) if name in t_keys]
        if idx:
            score = f1_score(np.array(y_true), np.array(y_pred), labels=idx, average='macro', zero_division=0)
            tier_results[t_name] = float(score)
            print(f"{t_name} Macro F1: {score:.4f}")

    # Bootstrap Confidence Interval computation
    print("\n[INFO] Executing 1000-iteration Paired Bootstrap Confidence Interval calculation")
    y_t, y_p = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    n = len(y_t)
    scores = []
    for _ in range(1000):
        idx = rng.integers(0, n, n)
        scores.append(f1_score(y_t[idx], y_p[idx], average='macro', zero_division=0))
        
    base_score = f1_score(y_true, y_pred, average='macro', zero_division=0)
    print(f"\nFinal Macro F1 Score: {base_score:.4f}")
    print(f"95% Confidence Interval: [{np.percentile(scores, 2.5):.4f}, {np.percentile(scores, 97.5):.4f}]\n" + "-"*50)

    os.makedirs("results", exist_ok=True)
    with open(f"results/results_{MODE}_efficientnet.json", "w") as f:
        json.dump({
            "macro_f1": float(base_score), 
            "ci_low": float(np.percentile(scores, 2.5)), 
            "ci_high": float(np.percentile(scores, 97.5)), 
            "tiers": tier_results
        }, f, indent=2)

if __name__ == "__main__": 
    evaluate()