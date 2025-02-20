### Installation Steps

1. Install Anaconda

2. Create a Conda environment with Python 3.10.14 or 3.9:
   ```sh
   conda create --name <env_name> python=3.10.14  # or python=3.9
   conda activate <env_name>
   ```

3. Install TenSEAL:
   ```sh
   pip install tenseal
   ```

4. Install required packages from the following repository:
   https://github.com/wenzhu23333/Differential-Privacy-Based-Federated-Learning/tree/f4e0ba82a10f26409a601c65159c83ecfe9cda66

---

### Usage Instructions

To run the homomorphic encryption code:

1. Open **Anaconda PowerShell Prompt**
2. Activate your environment:
   ```sh
   conda activate <env_name>
   ```
3. Run the test script:
   ```sh
   python test.py
   ```

---

### Notes

- The **PAYD1 3.9** environment works with **Visual Studio Code**.
- **Clipping Gradients for Differential Privacy:**
  - To ensure gradients do not become unbounded during training, we apply **gradient clipping**.
  - Let **C** be the target bound for the maximum gradient norm.
  - For each sample in a batch:
    - Compute its gradient.
    - If its norm exceeds **C**, scale it down to **C**.
  - This ensures the model **does not learn too much** from any individual sample, maintaining privacy.

---

### Missing Package Issue?
If you run into missing dependencies, install:
```sh
pip install torchcsprng==0.2.0 torch==1.8.0 -f https://download.pytorch.org/whl/cu102/torch_stable.html
```
More details: https://github.com/pytorch/csprng#installation

---

### Installing a New Python Version on Ubuntu
Follow this guide: https://www.malekal.com/installer-python-3-9-ubuntu/






