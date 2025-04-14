# PEFT Methods Comparison

### Requirements
- Ubuntu (minimum 22.04)
- NVIDIA GPU with CUDA 12.x
- Python 3.10+

### Installation
Within project's directory:
1. Set up the virtual environment 
```bash
python -m venv peft_env
source peft_env/bin/activate
```

2. Ensure the proper Nvidia drivers and CUDA installation 
```bash
nvidia-sm
```

3. For CUDA 12.x install Pytorch with: 
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```
4. Install dependencies: 
```bash
pip install transformers peft datasets
``` 

5. Begin your prayers and then verify CUDA/PyTorch 
\`\`\`python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
\`\`\`
