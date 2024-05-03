intall anaconda

create an envirennement with python 3.10.14 ou 3.9

install tenseal 

install the packages needed from the https://github.com/wenzhu23333/Differential-Privacy-Based-Federated-Learning/tree/f4e0ba82a10f26409a601c65159c83ecfe9cda66

to use it you need: 

1- open anaconda prompt powershell
2- conda activate <env_name>
3- python test.py the homomorphic code 



the PAYD1 3.9 fonctionnne avec le code de visual studio



 To this end, we use the Gaussian mechanism that takes in two parameters, the noise multiplier and the bound on the gradient norm. But wait… The gradients that arise during training of a deep neural network are potentially unbounded. In fact, for outliers and mislabeled inputs they can be very large indeed. What gives?

If the gradients are not bounded, we’ll make them so ourselves! Let C be the target bound for the maximum gradient norm. For each sample in the batch, we compute its parameter gradient and if its norm is larger than C, we clip the gradient by scaling it down to C. Mission accomplished — all the gradients now are guaranteed to have norm bounded by C, which we naturally call the clipping threshold. Intuitively, this means that we disallow the model from learning more information than a set quantity from any given training sample, no matter how different it is from the rest.


what dont import /
pip install torchcsprng==0.2.0 torch==1.8.0 -f https://download.pytorch.org/whl/cu102/torch_stable.html    ::::: https://github.com/pytorch/csprng#installation

to install new python version / https://www.malekal.com/installer-python-3-9-ubuntu/

