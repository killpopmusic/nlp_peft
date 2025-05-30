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
nvidia-smi
```

3. For CUDA 12.x install Pytorch with: 
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```
4. Install dependencies: 
```bash
pip install transformers peft datasets evaluate scikit-learn matplotlib
``` 

5. Begin your prayers and then verify CUDA/PyTorch 
```bash
python3 cuda_verify.py
```


### Running the project

To run the project: 
```bash
python main.py
```
this will use default arguments, you can modify them by 

```bash
python main.py --method lora/prefix/prompt --epochs i --batch_size j 
```
