# bnn_chaos_model
Model and training code for Bayesian neural network for compact planetary instability

# Running

To train the model, download the data into a `/data` folder at the base of this repository.
Run the script `train.sh`, which will train
the model 30 times from different seeds.

To generate the figures, edit the `figures/generate.sh` script
to have `true` instead of `false` for any figure you'd like to generate.
Then, execute that script in the folder.

# Requirements

Version requirements, when necessary, are given in bold.
For others, you can probably install the latest stable release.

Python: 3.7.0

Packages:

- icecream==2.0.0
- scipy==1.4.1
- scikit-learn=0.22.1
- tqdm==4.42.0
- matplotlib==3.3.1
- numpy==1.19.1
- pandas==0.25.3
- **pytorch_lightning==0.9.1rc4**
- **seaborn==0.11.0**
- **torch==1.5.1**
- celluloid==0.2.0
- numba==0.50.1
- **rebound==3.9.0**
- xgboost==1.2.0
- **dask==2.11.0**
- einops==0.3.0
- fire==0.3.1
