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

### Performances

We experiment on many different algorithms in this repository. To this extent, we reproduce the reference paper performances, and compare our results below.

| Model | 1-shot (5-way Acc.) **| 5-shot (5-way Acc.) **| 1 -shot (20-way Acc.) **| 5-shot (20-way Acc.) **|  1-shot (5-way Acc.) °°| 5-shot (5-way Acc.) °°| 1 -shot (20-way Acc.) °°| 5-shot (20-way Acc.) °°|
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Protonet ([reference paper](https://arxiv.org/pdf/1703.05175.pdf)) | 98.8% | 99.7% | 96.0% | 98.9%| - | - | - | -|
| Protonet (this repo) | - | - | - | - | - | - | - | -|
| MAML ([reference paper](https://arxiv.org/pdf/1703.03400.pdf)) | 98.7% | 99.9% | 95.8% | 98.9%| - | - | - | -|
| MAML (this repo) | - | - | - | - | - | - | - | -|


\*\* denotes the Omniglot dataset.
°° denotes the MiniImagenet dataset.
