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

### Task Samplers used in this repository
<p align="center">
<img src="https://raw.githubusercontent.com/RamnathKumar181/Task-Diversity/main/plots/uniform_task_sampler.png?token=AMO7VYQGUSXUNUG4UOYUWS3BAPSEK" width="500"/>
</p>

<p align="center">
<img src="https://raw.githubusercontent.com/RamnathKumar181/Task-Diversity/main/plots/no_diversity_task_sampler.png?token=token=AMO7VYRB6Y54FDJY2UXA5GLBAPTD2" width="500"/>
</p>

<p align="center">
<img src="https://raw.githubusercontent.com/RamnathKumar181/Task-Diversity/main/plots/no_diversity_batch_sampler.png?token=AMO7VYVXBMU3NNPDVWPSVPTBAPTHW" width="500"/>
</p>
