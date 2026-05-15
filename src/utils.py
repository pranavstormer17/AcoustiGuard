"""
Utility modules including text processing handlers and custom PyTorch dataset classes.
"""
import os
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset

LABEL_TO_CHAR = {
    "space": " ", "enter": "\n", "backspace": "[B]", "lshift": "[Shift]", "rshift": "[Shift]", 
    "hyphen": "-", "equals": "=", "comma": ",", "period": ".", "slash": "/",
    "bracketleft": "[", "bracketright": "]", "backslash": "\\", "backtick": "`", "semicolon": ";", "apostrophe": "'"
}

def decode_predictions(class_names, predictions):
    """Maps class label strings to their corresponding character representations."""
    return [LABEL_TO_CHAR.get(class_names[p].lower(), class_names[p]) for p in predictions]

def clean_backspaces(char_list):
    """Simulates backspace execution within a list of predicted characters."""
    cleaned = []
    for char in char_list:
        if char == "[B]" and cleaned: 
            cleaned.pop()
        elif char != "[Shift]": 
            cleaned.append(char)
    return "".join(cleaned)

class SpectrogramDataset(Dataset):
    """
    Custom Dataset implementation for loading raw float32 NumPy arrays, 
    preserving absolute dB values and applying deterministic tensor resizing.
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples = []
        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            for file_name in os.listdir(cls_dir):
                if file_name.endswith('.npy'):
                    self.samples.append((os.path.join(cls_dir, file_name), self.class_to_idx[cls_name]))

    def __len__(self): 
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        spec = np.load(path) 
        
        # Linear normalization of dB values to [0, 1] range
        spec = (spec + 80.0) / 80.0 
        spec = np.clip(spec, 0.0, 1.0)
        
        # Tensor conversion and geometric manipulation for pretrained architecture compatibility
        spec_tensor = torch.from_numpy(spec).float().view(1, 1, spec.shape[0], spec.shape[1])
        spec_tensor = F.interpolate(spec_tensor, size=(224, 224), mode='bilinear', align_corners=False)
        spec_tensor = spec_tensor.squeeze(0).repeat(3, 1, 1)
        
        if self.transform: 
            spec_tensor = self.transform(spec_tensor)
            
        return spec_tensor, label