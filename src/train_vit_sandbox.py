import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import timm

# Disable Flash SDP to prevent CUDA memory conflicts on Ampere-class GPUs
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)

# SELF-CONTAINED DATASET CLASS (Matches your paper's Float32 pipeline exactly)
class SpectrogramDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        self.filepaths = []
        self.labels = []
        for i, cls_name in enumerate(self.classes):
            cls_dir = os.path.join(data_dir, cls_name)
            for fname in os.listdir(cls_dir):
                if fname.endswith('.npy'):
                    self.filepaths.append(os.path.join(cls_dir, fname))
                    self.labels.append(i)
                    
    def __len__(self):
        return len(self.filepaths)
        
    def __getitem__(self, idx):
        mel_db = np.load(self.filepaths[idx])
        tensor = torch.tensor(mel_db, dtype=torch.float32)
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
            
        # Linear map [-80, 0] dB to [0.0, 1.0]
        tensor = (tensor + 80.0) / 80.0
        tensor = torch.clamp(tensor, 0.0, 1.0)
        
        # Resize to 224x224 for ViT backbone
        tensor = torch.nn.functional.interpolate(tensor.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)
        
        # Repeat to 3 channels
        if tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1)
            
        return tensor, self.labels[idx]

def train_vit():
    print("Initializing ViT-B/16 Sandbox...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_dir = "data/processed/classroom/train"
    train_ds = SpectrogramDataset(train_dir)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4)
    num_classes = len(train_ds.classes)
    
    model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    
    epochs = 15
    print(f"Starting Training on {device} for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        print(f"ViT Epoch [{epoch+1}/{epochs}] Loss: {running_loss/len(train_loader):.4f}")
        
    os.makedirs("models/sandbox", exist_ok=True)
    torch.save(model.state_dict(), "models/sandbox/vit_baseline.pth")
    print("ViT Training Complete. Saved to models/sandbox/vit_baseline.pth")

if __name__ == "__main__":
    train_vit()