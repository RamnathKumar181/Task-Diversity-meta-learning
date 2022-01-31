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

### Datasets

Omniglot, *mini*ImageNet, and *tiered*ImageNet will be downloaded automatically upon runnning the scripts, with the help of [pytorch-meta](https://github.com/tristandeleu/pytorch-meta). To download meta-dataset, follow the following steps:
* Download ILSVRC2012 (by creating an account [here](https://image-net.org/challenges/LSVRC/2012/index.php) and downloading `ILSVRC2012.tar`) and Cu_birds2012 (downloading from `http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz`) separately.
* Run `sbatch scripts/download_meta_dataset/install_meta_dataset_parallel.sh` to download and prune all datasets in a parallel fashion. Note that, due to memory constraints, we are saving ILSVRC2012 in a seperate directory from the other 9 datasets. While running the scripts to train the model, make sure to copy all the datasets to a single directory.
* That's it, you are good to go! :)

For the few-shot-regression setting, we test on Sinusoid, Sinusoid & Line, and Harmonic dataset. Furthermore, these datasets are toy examples and require no downloads.

### Training & Testing

We have carefully organized our codes under [`scripts`](scripts).

Our model can be trained on all samplers in a parallel fashion as follows:
```bash
sbatch scripts/MAML/Train/train_maml_<dataset>_all_samplers.sh
```
Similarly, our models can be tested on a fixed set of tasks in a parallel fashion as follows:
```bash
sbatch scripts/MAML/Test/test_maml_<dataset>_all_samplers.sh
```

All our codes were run with 1 GPUs, other than CNAPs, which has been run with 2. Furthermore, all our codes have been optimized to be run on less than 30Gb RAM, including our experiments on Meta-Dataset!

### Analysis

To compute results along with statistics, run:
```bash
python -m src.analysis.py <path_to_task_json> -O <path_to_output_json>
```

# Paper Citation

If you find our codes useful, do consider citing our paper:
```
@misc{kumar2022effect,
      title={The Effect of Diversity in Meta-Learning}, 
      author={Ramnath Kumar and Tristan Deleu and Yoshua Bengio},
      year={2022},
      eprint={2201.11775},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

# References

Our repository makes use of various open-source codes. If you find the respective codes useful, do cite their respective papers:

## Models

### Prototypical Network

```
@article{snell2017prototypical,
  title={Prototypical networks for few-shot learning},
  author={Snell, Jake and Swersky, Kevin and Zemel, Richard S},
  journal={arXiv preprint arXiv:1703.05175},
  year={2017}
}
```

### Matching Networks

```
@article{vinyals2016matching,
  title={Matching networks for one shot learning},
  author={Vinyals, Oriol and Blundell, Charles and Lillicrap, Timothy and Wierstra, Daan and others},
  journal={Advances in neural information processing systems},
  volume={29},
  pages={3630--3638},
  year={2016}
}
```

### MAML

```
@inproceedings{finn2017model,
  title={Model-agnostic meta-learning for fast adaptation of deep networks},
  author={Finn, Chelsea and Abbeel, Pieter and Levine, Sergey},
  booktitle={International Conference on Machine Learning},
  pages={1126--1135},
  year={2017},
  organization={PMLR}
}
```

### Reptile

```
@article{nichol2018first,
  title={On first-order meta-learning algorithms},
  author={Nichol, Alex and Achiam, Joshua and Schulman, John},
  journal={arXiv preprint arXiv:1803.02999},
  year={2018}
}
```

### CNAPs

```
@article{requeima2019fast,
  title={Fast and flexible multi-task classification using conditional neural adaptive processes},
  author={Requeima, James and Gordon, Jonathan and Bronskill, John and Nowozin, Sebastian and Turner, Richard E},
  journal={Advances in Neural Information Processing Systems},
  volume={32},
  pages={7959--7970},
  year={2019}
}
```

### MetaOptNet

```
@inproceedings{lee2019meta,
  title={Meta-learning with differentiable convex optimization},
  author={Lee, Kwonjoon and Maji, Subhransu and Ravichandran, Avinash and Soatto, Stefano},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={10657--10665},
  year={2019}
}
```

## Others

### Omniglot

```
@inproceedings{lake2011one,
  title={One shot learning of simple visual concepts},
  author={Lake, Brenden and Salakhutdinov, Ruslan and Gross, Jason and Tenenbaum, Joshua},
  booktitle={Proceedings of the annual meeting of the cognitive science society},
  volume={33},
  number={33},
  year={2011}
}
```

### *mini*ImageNet

```
@article{ravi2016optimization,
  title={Optimization as a model for few-shot learning},
  author={Ravi, Sachin and Larochelle, Hugo},
  year={2016}
}
```

### *tiered*ImageNet

```
@article{ren2018meta,
  title={Meta-learning for semi-supervised few-shot classification},
  author={Ren, Mengye and Triantafillou, Eleni and Ravi, Sachin and Snell, Jake and Swersky, Kevin and Tenenbaum, Joshua B and Larochelle, Hugo and Zemel, Richard S},
  journal={arXiv preprint arXiv:1803.00676},
  year={2018}
}
```

### Meta-Dataset

```
@article{triantafillou2019meta,
  title={Meta-dataset: A dataset of datasets for learning to learn from few examples},
  author={Triantafillou, Eleni and Zhu, Tyler and Dumoulin, Vincent and Lamblin, Pascal and Evci, Utku and Xu, Kelvin and Goroshin, Ross and Gelada, Carles and Swersky, Kevin and Manzagol, Pierre-Antoine and others},
  journal={arXiv preprint arXiv:1903.03096},
  year={2019}
}
```
### Sinusoid

```
@inproceedings{finn2017model,
  title={Model-agnostic meta-learning for fast adaptation of deep networks},
  author={Finn, Chelsea and Abbeel, Pieter and Levine, Sergey},
  booktitle={International Conference on Machine Learning},
  pages={1126--1135},
  year={2017},
  organization={PMLR}
}
```

### Sinudoid & Line

```
@article{finn2018probabilistic,
  title={Probabilistic model-agnostic meta-learning},
  author={Finn, Chelsea and Xu, Kelvin and Levine, Sergey},
  journal={arXiv preprint arXiv:1806.02817},
  year={2018}
}
```

### Harmonic

```
@article{lacoste2018uncertainty,
  title={Uncertainty in multitask transfer learning},
  author={Lacoste, Alexandre and Oreshkin, Boris and Chung, Wonchang and Boquet, Thomas and Rostamzadeh, Negar and Krueger, David},
  journal={arXiv preprint arXiv:1806.07528},
  year={2018}
}
```

### Torchmeta

```
@misc{deleu2019torchmeta,
  title={{Torchmeta: A Meta-Learning library for PyTorch}},
  author={Deleu, Tristan and W\"urfl, Tobias and Samiei, Mandana and Cohen, Joseph Paul and Bengio, Yoshua},
  year={2019},
  url={https://arxiv.org/abs/1909.06576},
  note={Available at: https://github.com/tristandeleu/pytorch-meta}
}
```
