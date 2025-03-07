# 🚗 Usage-Based Insurance with Differential Privacy and Homomorphic Encryption

This project implements **Usage-Based Insurance (UBI)** with **Differential Privacy (DP)** and **Homomorphic Encryption (HE)**, using datasets generated from SUMO simulations.

---

## 🚀 Project Setup

### 1️⃣ Clone the Project
First, clone the repository and navigate into the project directory:
```bash
   git clone https://github.com/badr007-01/Usage_Based_insurance_DP_HE_from_SUMODataset.git
   cd Usage_Based_insurance_DP_HE_from_SUMODataset
```

### 2️⃣ Create and Activate the Conda Environment
Create a Python 3.9 environment and activate it:
``` bash 
conda create --name ubi_env python=3.9 -y
conda activate ubi_env
```

### 3️⃣ Install Required Packages
## Install TenSEAL:

``` bash 
   pip install tenseal
``` 

## ✅ Install PyTorch and torchcsprng:
```bash
   pip install torch==1.8.0 torchcsprng==0.2.0 -f https://download.pytorch.org/whl/cu102/torch_stable.html
```
## ✅ Install other dependencies:
If the project includes a requirements.txt file, run:

``` bash
Copy
Edit
pip install -r requirements.txt

#⚠️ If there is no requirements.txt, manually install any additional packages used in the code.
```

## ✅ Install packages from this external repository:
Follow the installation steps provided in that repository for specific DP-related dependencies.

### 💻 Usage Instructions
## Running the Homomorphic Encryption Code:
1. Open Anaconda PowerShell Prompt (or terminal).
2. Activate your environment:
```bash
   conda activate ubi_env
   ```
3. Run the test script:
``` bash
   python test.py
```


###❓ Troubleshooting
## Missing Package Issue?
If you run into missing packages or compatibility problems, reinstall:

```bash
   pip install torch==1.8.0 torchcsprng==0.2.0 -f https://download.pytorch.org/whl/cu102/torch_stable.html
```
More info: torchcsprng installation guide


### Installing a New Python Version on Ubuntu
Follow this guide: https://www.malekal.com/installer-python-3-9-ubuntu/






