# Fair-DP
This is the codebase accompanying the paper [Disparate Impact in Differential Privacy from Gradient Misalignment](https://arxiv.org/abs/2206.07737). It was accepted for a spotlight presentation in ICLR 2023 and you can check the [open review](https://openreview.net/forum?id=qLOaeRvteqbx).
## Prerequisites

- Install conda, pip
- Python 3.10

```bash
conda create -n FairDP python=3.10
conda activate FairDP
```

- PyTorch 1.11.0

```bash
conda install pytorch=1.11.0 torchvision=0.12.0 numpy=1.22 -c pytorch
```

- functorch 0.1.1

```bash
pip install functorch==0.1.1
```

- opacus 1.1

```bash
conda install -c conda-forge opacus=1.1
```

- matplotlib 3.4.3

```bash
conda install -c conda-forge matplotlib=3.4.3
```

- Other requirements

```bash
conda install pandas tbb regex tqdm tensorboardX=2.2
pip install tensorboard==2.9

```

Scripts to reproduce experiments located at fair-dp/experiment_scripts, results saved to fair-dp/runs.

```
bash ./experiment_scripts/mnist_script.sh
tensorboard --logdir=runs
```

- Download CelebA dataset from https://www.kaggle.com/datasets/jessicali9530/celeba-dataset and save files to
  fair-dp/data/celeba/

- Download Adult dataset from https://archive.ics.uci.edu/ml/datasets/Adult and save files adult.data, adult.test to
  fair-dp/data/adult/

```
bash ./experiment_scripts/adult_script.sh
```

- Download Dutch dataset from https://easy.dans.knaw.nl/ui/datasets/id/easy-dataset:32357. Free registration is required
  on the website. Under the "Data Files" tab download all files. Unzip and save to fair-dp/data/dutch/. Full file path
  required is ./fair-dp/data/dutch/original/org/IPUMS2001.asc

```
bash ./experiment_scripts/dutch_script.sh
```