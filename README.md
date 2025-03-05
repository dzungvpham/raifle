# RAIFLE: Reconstruction Attack on Interaction-based Federated Learning with Adversarial Data Manipulation

This is the repository for our paper [RAIFLE: Reconstruction Attack on Interaction-based Federated Learning with Adversarial Data Manipulation](https://www.ndss-symposium.org/ndss-paper/raifle-reconstruction-attacks-on-interaction-based-federated-learning-with-adversarial-data-manipulation/) (NDSS 2025).
We improve the performance of reconstruction attacks against federated recommendation and learning to rank systems by manipulating the training features of the recommendation/ranking items.
To reproduce our results in the paper, please follow the following steps:

### Requirements:

- Hardware:
  - A commodity machine with at least 16GB of RAM and 30GB of storage.
- Software:
  - A (x86-64) Unix-based OS. (Windows WSL will probably also work, but might need some installation modifications.)
  - conda (such as Miniconda)
- Highly recommended (for the image-based experiment only):
  - A CUDA-capable NVIDIA GPU with at least 8GB of VRAM (preferably 12-16GB) and NVIDIA driver installed.

### Installation:

- Download this repo to your machine (e.g. `git clone https://github.com/dzungvpham/raifle.git`) and navigate to the downloaded folder.
- Install conda if needed (e.g. [Miniconda](https://docs.anaconda.com/miniconda/#quick-command-line-install)).
- Create a conda environment with `environment.yml`: `conda env create -f environment.yml`. This will create an environment named `raifle` and install all necessary packages.
- Activate the environment: `conda activate raifle`.

### Download Data:

Download the following datasets into the `dataset` folder and make sure to put the content in the specified subfolders (case-sensitive):

- Recommendation:
  - MovieLens-100K: Download the 100K version from [here](https://files.grouplens.org/datasets/movielens/ml-100k.zip) and extract into folder `ML-100K`.
  - Steam-200K: Download the Kaggle .csv file from [here](https://www.kaggle.com/datasets/tamber/steam-video-games). You will need to sign up for a Kaggle account. Put the .csv file in folder `STEAM-200K`.
- Learning to Rank:
  - LETOR: Download MQ2007.rar from [here](https://1drv.ms/f/s!Aqi9ONgj3OqPaynoZZSZVfHPJd0). (Microsoft [link](https://www.microsoft.com/en-us/research/project/letor-learning-rank-information-retrieval/letor-4-0/).) Extract to folder `MQ2007`. (Optional: Do the same for MQ2008.rar if you want to test on this dataset)
  - MSLR: Download the zip file [MSLR-WEB10K.zip](https://1drv.ms/u/s!AtsMfWUz5l8nbOIoJ6Ks0bEMp78). (Microsoft [link](https://www.microsoft.com/en-us/research/project/mslr/).) Unzip the content into folder `MSLR-WEB10K`.
- ImageNet:
  - Sign up for an account at [ImageNet](https://image-net.org/index.php) and obtain permissions to download data.
  - Once you have permission, go to [ILSVRC 2012](https://image-net.org/challenges/LSVRC/2012/2012-downloads.php) and download the "Development kit (Task 1 & 2)" and the "Validation images (all tasks)". You do not need to extract, just put the compressed files into folder `ImageNet`.

### Explanation of Code Files:

All of our code are in the `code` folder.

- `attack.py`: Our implementation of the RAIFLE attack.
- `dataset.py`: Code to load and preprocess datasets.
- `ranker.py`: Our implementation of the FPDGD algorithm and the FNCF algorithm.
- `utils.py`: Contains code for metrics, differential privacy, and click models.
- `experiment_rec.ipynb`: Experiments for Federated Recommender Systems.
- `experiment_ltr.ipynb`: Experiments for Federated Online Learning to Rank (FOLTR) with LETOR and MSLR data.
- `experiment_ltr_cv.ipynb`: Experiments for FOLTR with image-based data from ImageNet.
- `raifle_ltr_cv_colab.ipynb`: A Colab notebook modified from `experiment_ltr_cv.ipynb` in case a GPU is not available.
- `plot.ipynb`: Code for generating various plots and tables.

### Reproducing Our Results:

All of our experiment code are in IPython Jupyter notebooks. We recommend using [Visual Studio Code](https://code.visualstudio.com/docs/datascience/jupyter-notebooks) so that you can interactively run/modify our code.

- Main results (Section VI) + Local DP results (Section VII.A):
  - Federated recommendation (Section VI.A):
    - Run all cells in `experiment_rec.ipynb` in order (default dataset is MovieLens-100K).
    - Cell #2 contains instructions on how to change the dataset. For artifact evaluation, we scaled down the number of users attacked to 30 (a full run on MovieLens can take more than 1 day).
    - After cell #3 is done, the results are saved to disk and also printed out. The name column describes the configuration in the format FNCF_eps_{epsilon}\_IMIA_{reg_factor}, where {epsilon} refers to to the local DP epsilon parameter (inf means no privacy). IMIA_0.0 means the IMIA defense is not applied, IMIA_1.0 means the IMIA defense is applied with L1 regularization factor 1.0.
  - Federated OLTR with MQ2007 and MSLR-WEB10K (Section VI.B):
    - Run cell #1, 2, and 3 in `experiment_ltr.ipynb`.
    - Cell #2 contains instructions on how to change the dataset and other configs. For artifact evaluation, the default configuration is a linear ranker + a neural net ranker with 16 hidden units, MQ2007 dataset, and 16 queries per user (a full run on MQ2007 can take more than 1 day, MSLR-WEB10K is much longer).
    - After cell #3 is done, the results are saved to disk and also printed out. The name column describes the config in the format {model_name}\_{click_model_name}\_{num_query}\_query_eps_{epsilon}_{key}, where {model_name} is either 'linear_pdgd' or 'neural_16_pdgd', {click_model_name} is either 'informational' or 'navigational', {num_query} is the number of queries per user (e.g., 16), {epsilon} is the local DP epsilon (inf means no privacy), and {key} is either 0.0 or 1.0, where 0.0 means no manipulation and 1.0 means full manipulation.
  - FOLTR with ImageNet (Section VI.C):
    - If you have a GPU:
      - Run cell #1, 2, 3, and 4 in `experiment_ltr_cv.ipynb`. The default configuration (scaled down for artifact evaluation) is ResNet18 as feature extractor, 30 rounds of simulation, and 5,000 images.
      - Cell #2 contains instructions on how to change the feature extractor.
      - Cell #3 generates the manipulated images. You may need to adjust the batch size depending on how much GPU memory is available, e.g., 128 if 8GB, 256 if 12GB or more.
      - After cell #4 is done, the results are saved to disk and also printed out. The name column describes the config in the format {model_name}\_{num_items}\_items_eps_{epsilon}_{key}, where {num_items} is 512, 1024, or 2048 (assuming ResNet18), {epsilon} is the local DP epsilon (inf means no privacy), and {key} is 'no_adm' (no manipulation) or 'adm_opt' (RAIFLE).
      - If you want to run FGSM, see the instructions in cell #4, it will take quite a bit longer since it optimizes the images. The {key} will be `adm_FGSM_0.1`.
    - If you don't have a GPU: GPU is only necessary for cell #3 to generate the manipulated images. The reconstruction is decently fast without GPU (unless you also want FGSM).
      - You can use Google Colab with the T4 GPU as a free alternative. You will need to download the ImageNet data to Colab (e.g., upload to Google Drive then download from Drive to Colab). Use our notebook `raifle_ltr_cv_colab.ipynb` which has been modified from `experiment_ltr_cv.ipynb` to work with Colab.
- Additional discussion/appendix results:
  - FL Utility vs DP (Table X of Section VII.A, Section VIII.A):
    - Run cell #1, 2, and 4 of `experiment_ltr.ipynb`. Make sure to change the dataset to MSLR-WEB10K to see the results in our paper.
  - RAIFLE vs Secure aggregation + DP (distributed DP) (Table XI of Section VII.B):
    - Run cell #1, 2, and 5 of `experiment_ltr.ipynb`. Make sure to change the dataset to MQ2007 to see the results in our paper.
  - t-SNE visualization:
    - Run cell #1, 2, and 6 of `experiment_ltr.ipynb`. Make sure to change the dataset to MSLR-WEB10K to see the results in our paper.
    - Run cell #1, 2, and 5 of `experiment_ltr_cv.ipynb`.
  - Constrained server capability:
    - Run cell #1, 2, and 3 in `experiment_ltr.ipynb`. Make sure to set `alphas = [0.5, 0.75]` in cell #3, as this controls the % of features that the server can manipulate.
  - Manipulated image quality:
    - Run cell #1, 2, and 6 of `experiment_ltr_cv.ipynb`.

### Citation

Here's the BibTex for our paper:

```
@inproceedings{pham2025raifle,
  title={{RAIFLE}: Reconstruction Attacks on Interaction-based Federated Learning with Adversarial Data Manipulation},
  author={Pham, Dzung and Kulkarni, Shreyas and Houmansadr, Amir},
  booktitle={32nd Annual Network and Distributed System Security Symposium (NDSS 2025)},
  year={2025},
  doi={10.14722/ndss.2025.240363},
  url={https://dx.doi.org/10.14722/ndss.2025.240363},
  address={San Diego, CA, USA},
  publisher={The Internet Society}
}
```
