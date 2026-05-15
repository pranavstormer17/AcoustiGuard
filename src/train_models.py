"""
Primary training script utilizing the EfficientNet-V2-S architecture under deterministic constraints.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
import sys
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import MODE, PATHS
from src.utils import SpectrogramDataset

# Enforce deterministic algorithm execution for reproducible research
torch.backends.cudnn.benchmark = False

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_worker(worker_id):
    """Seed generation for deterministic DataLoader behavior."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def train():
    train_dir = os.path.join(PATHS[MODE]["out"], "train")
    os.makedirs("models", exist_ok=True)
    model_name = f"models/efficientnet_{MODE}.pth"
    print(f"[INFO] Initializing training sequence. Architecture: EfficientNet-V2-S | Hardware: {device} | Mode: {MODE.upper()}") 
    
    transform = transforms.Compose([
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    full_ds = SpectrogramDataset(train_dir, transform=transform)
    num_classes = len(full_ds.classes)
    
    train_idx, val_idx = train_test_split(
        np.arange(len(full_ds)), test_size=0.15, stratify=[s[1] for s in full_ds.samples], random_state=seed
    )
    train_ds = torch.utils.data.Subset(full_ds, train_idx)
    val_ds = torch.utils.data.Subset(full_ds, val_idx)
    
    g = torch.Generator()
    g.manual_seed(seed)
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2, pin_memory=True, worker_init_fn=seed_worker, generator=g)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2, pin_memory=True, worker_init_fn=seed_worker, generator=g)

    # Class distribution weighting
    counts = np.bincount([full_ds.samples[i][1] for i in train_idx], minlength=num_classes)
    counts = np.maximum(counts, 1) 
    weights = torch.FloatTensor(1.0 / counts).to(device)
    weights = weights / weights.mean() 

    model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(15):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                val_loss += criterion(model(inputs), labels).item()
                
        t_loss = train_loss / len(train_loader)
        v_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1:02d}/15 - Training Loss: {t_loss:.4f} | Validation Loss: {v_loss:.4f}")
        
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_name)
        else:
            patience_counter += 1
            if patience_counter >= 3: 
                print("[INFO] Early stopping threshold reached. Terminating epoch iteration.")
                break
        scheduler.step()

if __name__ == "__main__": 
    if MODE != "masked":
        train()
    else:
        print("[ERROR] Attempted execution of training script on masked dataset. Aborting to preserve evaluation integrity.")