import torch

def check_hardware():
    print("\n[INFO] === Hardware Verification ===")
    cuda_available = torch.cuda.is_available()
    
    if cuda_available:
        print(f"[SUCCESS] CUDA Enabled: True")
        print(f"[SUCCESS] GPU Detected: {torch.cuda.get_device_name(0)}")
        print(f"[SUCCESS] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")
    else:
        print("[FATAL] CUDA Enabled: False")
        print("[FATAL] Running on CPU. Training will be extremely slow and is not recommended.\n")

if __name__ == "__main__":
    check_hardware()