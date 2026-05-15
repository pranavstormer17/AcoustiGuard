# AcoustiGuard: Acoustic Side-Channel Attack Defense

AcoustiGuard is a controlled proof-of-concept for a reactive, hardware-agnostic defense against Acoustic Side-Channel Attacks (ASCA) targeting physical keyboards. This repository demonstrates how dynamically scaled $1/f$ Pink Noise masking can mathematically degrade the feature extraction capabilities of state-of-the-art neural architectures.

This project was built from the ground up, beginning with a custom, single-session dataset. To capture the precise transient acoustics of individual keystrokes, 52 specific physical keys (comprising alphanumeric, modifier, and system keys) were recorded individually. This collection was performed across two distinct acoustic environments: a controlled, quiet 'Home' environment to establish a high-fidelity baseline, and a 'Classroom' environment to simulate real-world ambient noise and acoustic domain shift. The dataset natively incorporates a Zipfian distribution, mirroring the natural frequency of English language keystrokes to ensure the evaluator does not artificially inflate accuracy by guessing high-probability keys.

Unlike standard ASCA literature which frequently normalizes acoustic data into lossy 8-bit `.png` images, AcoustiGuard processes all audio strictly as uncompressed `Float32` tensors. This ensures the neural evaluator is operating on absolute acoustic physics (ranging from -80 dB to 0 dB) rather than visual compression artifacts. During the masking phase, the $1/f$ Pink Noise is mathematically scaled to precisely $2.5\times$ the ambient RMS noise floor before being mixed with the keystroke transient. A linear bounding scale is applied post-mixing to prevent digital clipping while strictly preserving the waveform geometry.

The efficacy of this defense is benchmarked against a pure convolutional architecture (EfficientNet-V2-S), with further exploratory sandbox analysis against hybrid attention (CoAtNet-0) and pure transformer (ViT-B/16) paradigms. 

**Note on Generalizability:** This repository represents a foundational prototype. The current evaluation relies on a single-session dataset recorded on a specific laptop chassis (Lenovo LOQ) and microphone array. While the mathematical collapse of the attacker model is absolute within this constrained environment, significant future work is required to establish true cross-session, multi-device generalizability.

## Project Structure & Outputs
This repository automatically routes all generated artifacts into dedicated directories during execution:
- `figures/`: Confusion matrices (`conf_matrix_*.png`) and the single-strike visual proof (`defense_proof.png`).
- `results/`: Evaluation metrics (`results_*.json`) including F1 score, confidence intervals, and tier scores.
- `logs/`: Execution telemetry (`pipeline_log_*.json` & `.log`) for latency tracking and debugging.
- `models/`: Neural network weight checkpoints (`.pth`), mapping data, and the `sandbox/` directory for comparative model training (ViT, CoAtNet).

---

## Full Pipeline Execution Guide

The following commands document the exact sequence required to reproduce this thesis experiment from start to finish.

**Prerequisites:** 
* **OS:** Strictly optimized for Linux (e.g., CachyOS, Arch, Ubuntu). macOS is not supported.
* **Hardware:** NVIDIA GPU with CUDA acceleration.
* **Network:** A stable internet connection is required for Step 1 (downloading PyTorch/CUDA binaries) and Step 5 (downloading Hugging Face architecture weights).

## External Assets: Dataset & Pre-trained Models
Due to GitHub file size constraints, the heavy acoustic datasets and neural network weights are hosted externally.

* **[Download Raw Acoustic Dataset](https://drive.google.com/file/d/19rVKCxq-1CxT2PDhDqhNO6jS8eLtsfjc/view?usp=drive_link)**: This archive contains the `data/` directory structure. Extract this file directly into the root folder of this repository. It will automatically merge and place the raw audio recordings into `data/raw/home/` and `data/raw/classroom/`.
* **[Download Pre-trained Model Weights](https://drive.google.com/file/d/1HEWiOCCEMl-bKJe0nAwAI2oIbnVNLbR9/view?usp=drive_link)**: This archive contains the `models/` directory structure. Extract this file directly into the root folder of this repository. It will automatically populate the main directory with the `efficientnet.pth` files and the `models/sandbox/` directory with the `vit` and `coatnet` weights.

## Live Demonstration Video
[![AcoustiGuard Demo](https://img.youtube.com/vi/0DB4WZswFkQ/maxresdefault.jpg)](https://www.youtube.com/watch?v=0DB4WZswFkQ)

### Step 1: Environment & Dependencies
First, verify your active shell, create a fresh Python virtual environment, and install the required dependencies.

```bash
# 1. Determine your active shell (Outputs /bin/bash, /usr/bin/zsh, or /usr/bin/fish)
echo $SHELL

# 2. Create the Python virtual environment
python3 -m venv venv

# 3. Activate the environment based on your shell:
# ---> If using standard Linux Bash/Zsh:
source venv/bin/activate
# ---> If using Fish shell:
source venv/bin/activate.fish

# 4. Install all required machine learning and signal processing frameworks
pip install -r requirements.txt

# 5. Verify your GPU is detected and CUDA is enabled
python gpu.py

```

### Step 2: The "Clean Slate" Initialization

*Note for first-time runners: This step is highly recommended to guarantee a sterile environment. It strictly wipes generated arrays and logs without touching the `data/raw/` source audio.*

> **Expected Warning:** If you see terminal messages like `No matches for wildcard` (Fish) or `No such file or directory` (Bash), this simply means your environment is already clean. It can be safely ignored.

```bash
# Wipe processed Float32 arrays
rm -rf data/processed/home/* 2>/dev/null
rm -rf data/processed/classroom/* 2>/dev/null
rm -rf data/processed/home_masked/* 2>/dev/null
rm -rf data/processed/masked/* 2>/dev/null

# Wipe dynamically generated pink noise audio
rm -f data/raw/classroom_masked/*.wav 2>/dev/null
rm -f data/raw/home_masked/*.wav 2>/dev/null

# Wipe legacy models, logs, and figures
rm -f models/*.pth 2>/dev/null
rm -f models/sandbox/*.pth 2>/dev/null
rm -f logs/* 2>/dev/null
rm -f results/* 2>/dev/null
rm -f figures/* 2>/dev/null

echo "Clean slate confirmed. Environment ready for execution."

```

### Step 3: Verify Raw Data Integrity

Verify that the 52 physical key classes are present in the raw audio directories prior to preprocessing.

```bash
echo "=== HOME RAW AUDIO ===" && ls data/raw/home/ | wc -l
echo "=== CLASSROOM RAW AUDIO ===" && ls data/raw/classroom/ | wc -l

```

### Step 4: Core Automated Pipelines

Execute the primary training, masking, and evaluation pipelines for both acoustic environments.

> **Environment Note (Why Both Environments?):** This step fully evaluates our primary Convolutional model (EfficientNet-V2-S) on **both the Classroom and Home environments independently**. We do this by running two parallel pipelines (`run_pipeline` and `run_home_pipeline`). Executing the attack and defense in two distinct acoustic spaces mathematically proves our core thesis: the 1/f Pink Noise defense effectively collapses an attacker's accuracy regardless of ambient background noise or room reverberation.

> **Expected Warning:** During the masked phases, `sanity_check.py` may print: `[WARNING] Class '...' contains an insufficient test sample count`. Because 1/f Pink Noise physically destroys acoustic transients, the librosa onset slicer may occasionally isolate fewer than 15 valid samples for quiet keys (like `[` or `-`). The pipeline will dynamically adjust and continue successfully.

**Option A: If you are using the Fish Shell (Native Scripts)**

```bash
# 1. Execute Classroom Pipeline (Baseline & Masked)
./run_pipeline.fish

# 2. Execute Home Pipeline (Baseline & Masked)
./run_home_pipeline.fish

```

**Option B: If you are using standard Linux Bash/Zsh (Manual Execution)**
*(Because the automation scripts use Fish syntax, Bash users can replicate the pipeline sequentially using native `sed` commands to update the config).*

```bash
# 1. Execute Classroom Pipeline
sed -i 's/^MODE = .*/MODE = "classroom"/' config.py
python -m src.preprocess_data && python sanity_check.py && python -m src.train_models && python -m src.evaluate_models
python -m src.masker
sed -i 's/^MODE = .*/MODE = "masked"/' config.py
python -m src.preprocess_data && python sanity_check.py && python -m src.evaluate_models

# 2. Execute Home Pipeline
sed -i 's/^MODE = .*/MODE = "home"/' config.py
python -m src.preprocess_data && python sanity_check.py && python -m src.train_models && python -m src.evaluate_models
python -m src.masker
sed -i 's/^MODE = .*/MODE = "home_masked"/' config.py
python -m src.preprocess_data && python sanity_check.py && python -m src.evaluate_models

```

### Step 5: Comparative Architecture Sandbox

Train and evaluate the hybrid attention model (CoAtNet-0) and the pure Transformer (ViT-B/16), comparing them against the baseline convolutions.

> **Environment Note (Why Classroom Only?):** This sandbox strictly utilizes the **Classroom Baseline** dataset to evaluate the ViT and CoAtNet architectures. *Why?* To scientifically prove that Transformers struggle with acoustic spatial recognition compared to Convolutions, we must isolate the architecture as the only variable. Testing across different rooms (e.g., training in the classroom and testing in the home) causes "acoustic domain shift," forcing the models to fail due to environmental changes rather than architectural weakness. Restricting this sandbox to a single environment ensures a mathematically fair 1-to-1 baseline comparison.

> **Expected Warning 1:** During training, the `timm` library may warn about `Unauthenticated requests to the HF Hub`. The architecture weights will still download successfully without a token.

> **Expected Warning 2:** During `evaluate_all.py`, PyTorch invokes the Triton compiler which may trigger a `_POSIX_C_SOURCE redefined` C-header conflict. This is a harmless compiler warning and does not affect inference.

> **Expected Warning 3:** During `evaluate_all.py`, PyTorch may print `Not enough SMs to use max_autotune_gemm mode`. This simply means the code is running on a consumer GPU rather than a datacenter GPU, and it will fall back to standard, stable matrix math optimizations.

```bash
python -m src.train_coatnet_sandbox
python -m src.train_vit_sandbox
python -m src.evaluate_all

```

### Step 6: Live Hardware Demonstrations

Execute the live acoustic demonstrations natively over the PipeWire/ALSA backend.

> **Environment Note (Why Classroom Weights?):** These live demonstration scripts are hardcoded to load the **Classroom Baseline** model weights (`efficientnet_classroom.pth`) to perform real-time predictions against the live acoustics of your physical room. *Why?* A public classroom represents the primary, high-risk threat model for acoustic eavesdropping. Using this model as our universal live standard provides the most realistic demonstration of how the attack operates (and how the defense neutralizes it) in a noisy, real-world space.

```bash
# One-time setup: sync the class mapping to the trained model indices
python src/generate_mapping.py

# 1. Visual Defense Proof (Generates defense_proof.png in figures/)
python demo_defense_comparison.py

# 2. Real-Time ASCA Eavesdropping Simulation
python demo_live_attack.py
# (Type target keys on the laptop keyboard, then press Ctrl+C to terminate the stream)

```

### Step 7: NLP Semantic Reconstruction

Demonstrate the threat elevation of using the SymSpell algorithm (Maximum Edit Distance = 2) to rebuild compromised keystrokes into coherent English syntax.

```bash
python -m src.nlp_postprocess

```

### Step 8: Final Results Summary

Print the comprehensive data summary proving the mathematical collapse of the attacker's capabilities.

```bash
python3 -c "
import json
with open('results/results_classroom_efficientnet.json') as f: c = json.load(f)
with open('results/results_masked_efficientnet.json') as f: m = json.load(f)
print('\n==========================================')
print('ACOUSTIGUARD FINAL RESULTS SUMMARY')
print('==========================================')
print('\n--- CLASSROOM BASELINE vs MASKED ---')
print(f'Classroom Baseline F1:  {c[\"macro_f1\"]*100:.2f}%')
print(f'Classroom Masked F1:    {m[\"macro_f1\"]*100:.2f}%')
print(f'Defense Effectiveness:  -{(c[\"macro_f1\"]-m[\"macro_f1\"])*100:.2f} percentage points')
print(f'Attack collapsed to near-random chance: {m[\"macro_f1\"]*100:.2f}%')

with open('results/results_home_efficientnet.json') as f: h = json.load(f)
print('\n--- HOME BASELINE ---')
print(f'Home Baseline F1: {h[\"macro_f1\"]*100:.2f}%')
print(f'95% CI: [{h[\"ci_low\"]*100:.2f}%, {h[\"ci_high\"]*100:.2f}%]')
print('==========================================\n')
"

```
