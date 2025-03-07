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

### References

In this work \cite{this_work}, we use the approach based on the DP-SGD algorithms from the paper \cite{abadi2016deep}. An implementation is proposed by Meta AI, called Opacus library \cite{yousefpour2021opacus}.
``` bash
@inproceedings{this_work,
  title={PrivFedProfiling: Privacy-Preserving Federated Profiling Model for Enhanced Usage-Based Insurance Assessment},
  author={BADREDDINE CHAH, ANIS BKAKRI, ALEXANDRE LOMBARD, ABDELJALIL ABBAS-TURKI, ALEXANDRE BRUNOUD, YAZAN MUALLA, AND REDA YAICH},
  booktitle={Under Submission, IEEE Access journal},
  year={2025}
}

@inproceedings{abadi2016deep,
  title={Deep learning with differential privacy},
  author={Abadi, Martin and Chu, Andy and Goodfellow, Ian and McMahan, H Brendan and Mironov, Ilya and Talwar, Kunal and Zhang, Li},
  booktitle={Proceedings of the 2016 ACM SIGSAC conference on computer and communications security},
  pages={308--318},
  year={2016}
}

@article{yousefpour2021opacus,
  title={Opacus: User-friendly differential privacy library in PyTorch},
  author={Yousefpour, Ashkan and Shilov, Igor and Sablayrolles, Alexandre and Testuggine, Davide and Prasad, Karthik and Malek, Mani and Nguyen, John and Ghosh, Sayan and Bharadwaj, Akash and Zhao, Jessica and others},
  journal={arXiv preprint arXiv:2109.12298},
  year={2021}
}
```




