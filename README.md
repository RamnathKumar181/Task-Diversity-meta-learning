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

Omniglot and MiniImagenet will be downloaded automatically upon runnning the scripts, with the help of torch-meta. To download meta_dataset, follow the following steps:
* Download ILSVRC2012 (by creating an account [here](https://image-net.org/challenges/LSVRC/2012/index.php) and downloading `ILSVRC2012.tar`) and Cu_birds2012 (downloading from `http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz`) separately.
* Run `sbatch scripts/download_meta_dataset/install_meta_dataset_parallel.sh` to download and prune all datasets in a parallel fashion. Note that, due to memory constraints, we are saving ILSVRC2012 in a seperate directory from the other 9 datasets. While running the scripts to train the model, make sure to copy all the datasets to a single directory.
* That's it, you are good to go! :)

### Training & Testing

We have carefully organized our codes under ```scripts```.

Our model can be trained on all samplers in a parallel fashion as follows:
```bash
sbatch scripts/MAML/Train/train_maml_<dataset>_all_samplers.sh
```
Similarly, our models can be tested on a fix set of tasks in a parallel fashion as follows:
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

If you find our codes useful, please cite the respective paper:
```
Add arxiv link here
```

# References

Our repository makes use of various open-source codes. If you find the respective codes useful, please cite the respective papers:

## Dataset

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

### *Mini*ImageNet

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
