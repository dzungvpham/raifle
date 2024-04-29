# RAIFLE: Reconstruction Attack on Interaction-based Federated Learning with Adversarial Data Manipulation

This is the repository for the paper RAIFLE: Reconstruction Attack on Interaction-based Federated Learning with Adversarial Data Manipulation. To reproduce our results, please follow the following steps:

### Installation:

Clone the repo, then create a conda environment (inside the repo or wherever) using `conda-spec-file.txt`: `conda create --name raifle --file conda-spec-file.txt`. This will install all the necessary libraries using conda. Make sure to activate the conda environment once you are done: `conda activate raifle`.

### Data:

After downloading the data below, you should unzip them in the `dataset` folder.

- Learning to Rank
  - LETOR: Download MQ2007.rar and MQ2008.rar from [here](https://1drv.ms/f/s!Aqi9ONgj3OqPaynoZZSZVfHPJd0). (Microsoft [link](https://www.microsoft.com/en-us/research/project/letor-learning-rank-information-retrieval/letor-4-0/))
  - MSLR: Download [MSLR 10K](https://1drv.ms/u/s!AtsMfWUz5l8nbOIoJ6Ks0bEMp78) (Microsoft [link](https://www.microsoft.com/en-us/research/project/mslr/))

- Recommendation:
  - MovieLens: Download the 100K version (and others if you want to) from [here](https://grouplens.org/datasets/movielens/)

### Files:

- `attack.py`: Our implementation of the RAIFLE attack.
- `ranker.py`: Our implementation of the FPDGD algorithm and the FNCF algorithm.
- `dataset.py`: Code to load and preprocess dataset.
- `utils.py`: Contains code for metrics, differential privacy, and OLTR click models
- `experiment_ltr.ipynb`: Experiments for Federated Online Learning to Rank with LETOR and MSLR data.
- `experiment_ltr_cv.ipynb`: Experiments for Federated Online Learning to Rank with image-based data. First, follow PyTorch's instructions to set up the ImageNet-1K 2012 dataset. Then, generate the manipulated images before you can run the attack (follow the cells in the notebook). GPU is highly recommended (you may need to adjust the batch size depending on how much GPU memory is available)
- `experiment_rec.ipynb`: Experiments for Federated Recommender Systems.
- `plot.ipynb`: Code for generating plots and numbers
