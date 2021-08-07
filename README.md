# Task-Diversity

### Getting started
To avoid any conflict with your existing Python setup, it is suggested to work in a virtual environment with [`virtualenv`](https://docs.python-guide.org/dev/virtualenvs/). To install `virtualenv`:
```bash
pip install --upgrade virtualenv
```
Create a virtual environment, activate it and install the requirements in [`requirements.txt`](requirements.txt).
```bash
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### Requirements
 - Python 3.8
 - PyTorch 1.7.1
 - Torchvision 0.8.2
 - Torchmeta 1.7.0
 - Pillow 7.2.0
 - qpth 0.0.15
 - cvxpy 1.1.13
 - numpy 1.21.0
 - wandb 0.11.2

### Performances

We experiment on many different algorithms in this repository. To this extent, we reproduce the reference paper performances, and compare our results below.

| Model | 1-shot (5-way Acc.) <br>**| 5-shot (5-way Acc.) <br>**| 1 -shot (20-way Acc.) <br>**| 5-shot (20-way Acc.) <br>**|  1-shot (5-way Acc.) <br>°°| 5-shot (5-way Acc.) <br>°°| 1 -shot (10-way Acc.) <br>°°| 5-shot (10-way Acc.) <br>°°|
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MAML ([reference paper](https://arxiv.org/pdf/1703.03400.pdf)) | 98.7% | 99.9% | 95.8% | 98.9% | 48.7% | 63.1% | 31.3% | 46.9%|
| MAML (this repo) | 98.55% | - | - | - | - | - | - | -|
| Matching Networks ([reference paper](https://arxiv.org/pdf/1606.04080.pdf)) | 98.1% | 98.9% | 93.8% | 98.5% | 46.6% | 60.0% | - | - |
| Matching Networks (this repo) | 94.66% | - | - | - | - | - | - | - |
| MetaOptNet ([reference paper](https://arxiv.org/pdf/1904.03758.pdf)) | - | - | - | - | 64.09% | 80.0% | - | - |
| MetaOptNet (this repo) | - | - | - | - | - | - | - | -|
| Protonet ([reference paper](https://arxiv.org/pdf/1703.05175.pdf)) | 98.8% | 99.7% | 96.0% | 98.9% | 	49.42% | 68.20% | 32.9% | 49.3% |
| Protonet (this repo) | 97.7% | - | - | - | - | - | - | -|
| Reptile ([reference paper](https://arxiv.org/pdf/1803.02999.pdf)) | 97.68% | 99.48% | 89.43% | 97.12% | 49.97% | 65.99% | 31.1% | 44.7% |
| Reptile (this repo) | - | - | - | - | - | - | - | -|
| Simple CNAPS ([reference paper](https://arxiv.org/pdf/1906.07697.pdf)) | - | - | - | - | 53.2% | 70.8% | 37.1% | 56.7% |
| Simple CNAPS (this repo) | 94.73% | - | - | - | - | - | - | - |


\*\* denotes the Omniglot dataset.
°° denotes the MiniImagenet dataset.
