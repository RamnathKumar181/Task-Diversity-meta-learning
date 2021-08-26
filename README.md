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
* Run `sbatch scripts/download_meta_dataset/install_meta_dataset_parallel.sh` to download and prune all datasets in a parallel fashion. Note that, due to memory constraints, we are saving ILSVRC2012 in a seperate directory from the other 9 datasets. While running the scripts to train the model, make sure to copy all the datasets to a single directory `$SLURM_TMPDIR`.
* That's it, you are good to go! :)

### Performances

We experiment on many different algorithms in this repository. To this extent, we reproduce the reference paper performances, and compare our results below.

| Model | 1-shot (5-way Acc.) <br>**| 1-shot (5-way Acc.) <br>°°|
| --- | --- | --- |
| MAML ([reference paper](https://arxiv.org/pdf/1703.03400.pdf)) | 98.7% | 48.7% |
| MAML (this repo) | 98.55% | - |
| Matching Networks ([reference paper](https://arxiv.org/pdf/1606.04080.pdf)) | 98.1% | 46.6% |
| Matching Networks (this repo) | 94.66% | - |
| MetaOptNet ([reference paper](https://arxiv.org/pdf/1904.03758.pdf)) | - | 64.09%
| MetaOptNet (this repo) | - | - |
| Protonet ([reference paper](https://arxiv.org/pdf/1703.05175.pdf)) | 98.8% | 49.42% |
| Protonet (this repo) | 97.7% | - |
| Reptile ([reference paper](https://arxiv.org/pdf/1803.02999.pdf)) | 97.68% | 49.97% |
| Reptile (this repo) | - | - |
| Simple CNAPS ([reference paper](https://arxiv.org/pdf/1906.07697.pdf)) | - |53.2% |
| Simple CNAPS (this repo) | 94.73% | - |


\*\* denotes the Omniglot dataset.
°° denotes the MiniImagenet dataset.

We had trouble reproducing the results from matching networks using cosine distance since, the convergence seemed to be slow and the final performance dependent on the random initialization. This is similar to what is observed by [other repos](https://github.com/oscarknagg/few-shot). \
Finally, we also notice some discrepancy when it comes to the MetaOptNet. This discrepancy is mainly due to the different setting we run our experiments in. For instance, to test on 5 way-1 shot, we also train the model on 5 way-1 shot mode. However, the official repository for MetaOptNet trains the model on 15 shot, and tests the same model on 5 shot, or 1 shot. We believe this to be the reason for the discrepancy in the performace.
