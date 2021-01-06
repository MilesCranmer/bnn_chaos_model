# bnn_chaos_model
Model and training code for Bayesian neural network for compact planetary instability

# Running

To train the model, download the data into a `/data` folder at the base of this repository
data from this [Globus link](https://app.globus.org/file-manager?origin_id=ae09b8a8-5040-11eb-a4d1-0a53a3613b81&origin_path=%2F).

Run the script `train.sh`, which will train
the model 30 times from different seeds.

To generate the figures, edit the `figures/generate.sh` script
to have `true` instead of `false` for any figure you'd like to generate.
Then, execute that script in the folder.

# Requirements

Python: 3.7.0

Package version requirements are given in `create_env.sh`.
