# Task-Diversity in meta-learning

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

### Analysis

To compute results along with statistics, run:
```bash
python -m src.analysis.py <path_to_task_json> -O <path_to_output_json>
```
