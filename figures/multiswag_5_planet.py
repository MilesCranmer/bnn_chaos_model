#!/usr/bin/env python
# coding: utf-8
# %matplotlib inline

import sys
sys.path.append('../')

import os

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'


# import jax
# import jax.numpy as jnp
# from jax.random import PRNGKey, normal
import numpy as np

# +
import numpy as np
import rebound
import matplotlib.pyplot as plt
import matplotlib
import random
import dill
import sys
import pandas as pd
import spock
from spock import FeatureRegressor, FeatureRegressorXGB
from icecream import ic

try:
    plt.style.use('paper')
except:
    pass

spockoutfile = '../spockprobstesttrio.npz'
version = int(sys.argv[1])

from multiswag_5_planet_plot import make_plot

try:
    cleaned = pd.read_csv('cur_plot_dataset_1604437382.10866.csv')#'cur_plot_dataset_1604339344.2705607.csv')
    # make_plot(cleaned, version)
    make_plot(cleaned, version, t20=False)
    exit(0)
except FileNotFoundError:
    ...

stride = 1
nsim_list = np.arange(0, 17500)
# Paper-ready is 5000:
N = 5000
# Paper-ready is 10000
samples = 10000
used_axes = np.linspace(0, 17500-1, N).astype(np.int32)#np.arange(17500//3, 17500, 1750//3)

nsim_list = nsim_list[used_axes]


model = FeatureRegressor(
    cuda=True,
    filebase='../*' + f'v{version:d}' + '*output.pkl'
    # filebase='*' + 'v30' + '*output.pkl'
    #'long_zero_megno_with_angles_power_v14_*_output.pkl'
)

xgbmodel = FeatureRegressorXGB()


# -

# # read initial condition file

# +
infile_delta_2_to_10 = '../data/initial_conditions_delta_2_to_10.npz'
infile_delta_10_to_13 = '../data/initial_conditions_delta_10_to_13.npz'

ic1 = np.load(infile_delta_2_to_10)
ic2 = np.load(infile_delta_10_to_13)

m_star = ic1['m_star'] # mass of star
m_planet = ic1['m_planet'] # mass of planets
rh = (m_planet/3.) ** (1./3.)

Nbody = ic1['Nbody'] # number of planets
year = 2.*np.pi # One year in units where G=1
tf = ic1['tf'] # end time in years

a_init = np.concatenate([ic1['a'], ic2['a']], axis=1) # array containing initial semimajor axis for each delta,planet
f_init = np.concatenate([ic1['f'], ic2['f']], axis=1) # array containing intial longitudinal position for each delta, planet, run
# -

# # create rebound simulation and predict stability for each system in nsim_list

# +
infile_delta_2_to_10 = '../data/initial_conditions_delta_2_to_10.npz'
infile_delta_10_to_13 = '../data/initial_conditions_delta_10_to_13.npz'

outfile_nbody_delta_2_to_10 = '../data/merged_output_files_delta_2_to_10.npz'
outfile_nbody_delta_10_to_13 = '../data/merged_output_files_delta_10_to_13.npz'

## load hill spacing

ic_delta_2_to_10 = np.load(infile_delta_2_to_10)
ic_delta_10_to_13 = np.load(infile_delta_10_to_13)

delta_2_to_10 = ic_delta_2_to_10['delta']
delta_10_to_13 = ic_delta_10_to_13['delta']

delta = np.hstack((delta_2_to_10, delta_10_to_13))
delta=delta[used_axes]

## load rebound simulation first close encounter times

nbody_delta_2_to_10 = np.load(outfile_nbody_delta_2_to_10)
nbody_delta_10_to_13 = np.load(outfile_nbody_delta_10_to_13)

t_exit_delta_2_to_10 = nbody_delta_2_to_10['t_exit']/(0.99)**(3./2)
t_exit_delta_10_to_13 = nbody_delta_10_to_13['t_exit']/(0.99)**(3./2)

t_exit = np.hstack((t_exit_delta_2_to_10, t_exit_delta_10_to_13))
t_exit = t_exit[used_axes]

df = pd.DataFrame(np.array([nsim_list, delta, t_exit]).T, columns=['nsim', 'delta', 't_exit'])
df.head()

# -

# Generate features for the model:

# +
from oldsimsetup import init_sim_parameters

sims = []
sims_for_xgb = []
for nsim in nsim_list:
    # From Dan's fig 5
    sim = rebound.Simulation()
    sim.add(m=m_star)
    sim.G = 4*np.pi**2
    for i in range(Nbody): # add the planets
        sim.add(m=m_planet, a=a_init[i, nsim], f=f_init[i, nsim])


    init_sim_parameters(sim)
    sims.append(sim)
    sims_for_xgb.append(sim.copy())
    
# +


def data_setup_kernel(mass_array, cur_tseries):
    mass_array = np.tile(mass_array[None], (100, 1))[None]

    old_X = np.concatenate((cur_tseries, mass_array), axis=2)

    isnotfinite = lambda _x: ~np.isfinite(_x)

    old_X = np.concatenate((old_X, isnotfinite(old_X[:, :, [3]]).astype(np.float)), axis=2)
    old_X = np.concatenate((old_X, isnotfinite(old_X[:, :, [6]]).astype(np.float)), axis=2)
    old_X = np.concatenate((old_X, isnotfinite(old_X[:, :, [7]]).astype(np.float)), axis=2)

    old_X[..., :] = np.nan_to_num(old_X[..., :], posinf=0.0, neginf=0.0)

    # axis_labels = []
    X = []

    for j in range(old_X.shape[-1]):#: #, label in enumerate(old_axis_labels):
        if j in [11, 12, 13, 17, 18, 19, 23, 24, 25]: #if 'Omega' in label or 'pomega' in label or 'theta' in label:
            X.append(np.cos(old_X[:, :, [j]]))
            X.append(np.sin(old_X[:, :, [j]]))
            # axis_labels.append('cos_'+label)
            # axis_labels.append('sin_'+label)
        else:
            X.append(old_X[:, :, [j]])
            # axis_labels.append(label)
    X = np.concatenate(X, axis=2)
    if X.shape[-1] != 41:
        raise NotImplementedError("Need to change indexes above for angles, replace ssX.")

    return X


# +

from collections import OrderedDict
import sys
sys.path.append('../spock')
from tseries_feature_functions import get_extended_tseries


# +

def get_features_for_sim(sim_i, indices=None):
    sim = sims[sim_i]
    if sim.N_real < 4:
        raise AttributeError("SPOCK Error: SPOCK only works for systems with 3 or more planets") 
    if indices:
        if len(indices) != 3:
            raise AttributeError("SPOCK Error: indices must be a list of 3 particle indices")
        trios = [indices] # always make it into a list of trios to test
    else:
        trios = [[i,i+1,i+2] for i in range(1,sim.N_real-2)] # list of adjacent trios

    kwargs = OrderedDict()
    kwargs['Norbits'] = int(1e4)
    kwargs['Nout'] = 1000
    kwargs['trios'] = trios
    args = list(kwargs.values())
    # These are the .npy.
    tseries, stable = get_extended_tseries(sim, args)

    if not stable:
        return np.ones((3, 1, 100, 41))*4

    tseries = np.array(tseries)
    simt = sim.copy()
    alltime = []
    Xs = []
    for i, trio in enumerate(trios):
        sim = simt.copy()
        # These are the .npy.
        cur_tseries = tseries[None, i, ::10]
        mass_array = np.array([sim.particles[j].m/sim.particles[0].m for j in trio])
        X = data_setup_kernel(mass_array, cur_tseries)
        Xs.append(X)
        
    return Xs

def get_xgb_prediction(sim_i):
    sim = sims_for_xgb[sim_i].copy()
    return xgbmodel.predict(sim)

# +

from functools import partial

from multiprocessing import Pool

pool = Pool(7)

xgb_predictions = np.array(pool.map(
    get_xgb_prediction,
    range(len(sims))
))


X = np.array(pool.map(
    get_features_for_sim,
    range(len(sims))
))[:, :, 0, :, :]
#(sim, trio, time, feature)

# -

allmeg = X[..., model.swag_ensemble[0].megno_location].ravel()

# +
# from plotnine import *

# +
import seaborn as sns

# sns.distplot(
    # x=allmeg[allmeg<5][::10],
    # kde=True, hist=False,
    # label='5-planet')

# plt.legend()
# -

# Computationally heavy bit:

# Calculate samples:

a_init_p = a_init[:, used_axes]

# +
Xp = (model.ssX
      .transform(X.reshape(-1, X.shape[-1]))
      .reshape(X.shape)
)

import torch

Xpp = torch.tensor(Xp).float()

Xflat = Xpp.reshape(-1, X.shape[-2], X.shape[-1])

if model.cuda:
    Xflat = Xflat.cuda()


time = torch.cat([
    torch.cat([model.sample_full_swag(Xpart).detach().cpu() for Xpart in torch.chunk(Xflat, chunks=10)])[None]
    for _ in range(samples)
], dim=0).reshape(samples, X.shape[0], X.shape[1], 2).numpy()
# -

import numpy as jnp


# +

def fast_truncnorm(
        loc, scale, left=jnp.inf, right=jnp.inf,
        d=10000, nsamp=50, seed=0):
    """Fast truncnorm sampling.
    
    Assumes scale and loc have the desired shape of output.
    length is number of elements.
    Select nsamp based on expecting at minimum one sample of a Gaussian
        to fit within your (left, right) range.
    Select d based on memory considerations - need to operate on
        a (d, nsamp) array.
    """
    oldscale = scale
    oldloc = loc
    
    scale = scale.reshape(-1)
    loc = loc.reshape(-1)
    samples = jnp.zeros_like(scale)
    start = 0
    try:
        rng = PRNGKey(seed)
    except:
        rng = 0
        
    for start in range(0, scale.shape[0], d):

        end = start + d
        if end > scale.shape[0]:
            end = scale.shape[0]
        
        cd = end-start
        try:
            rand_out = normal(
                rng,
                (nsamp, cd)
            )
        except:
            rand_out = np.random.normal(size=(nsamp, cd))
        rng += 1

        rand_out = (
            rand_out * scale[None, start:end]
            + loc[None, start:end]
        )
        
        #rand_out is (nsamp, cd)
        if right == jnp.inf:
            mask = (rand_out > left)
        elif left == jnp.inf:
            mask = (rand_out < right)
        else:
            mask = (rand_out > left) & (rand_out < right)
            
        first_good_val = rand_out[
            mask.argmax(0), jnp.arange(cd)
        ]
        
        try:
            samples = jax.ops.index_update(
                samples, np.s_[start:end], first_good_val
            )
        except:
            samples[start:end] = first_good_val
        
    return samples.reshape(*oldscale.shape)

from time import time as ttime
# -





# +

sys.path.append('/mnt/home/mcranmer/local_orbital_physics/miles')

from petit20_survival_time import Tsurv
# -



# +

samps_time = np.array(fast_truncnorm(
        time[..., 0], time[..., 1],
        left=4, d=10000, nsamp=40,
        seed=int((ttime()*1e6) % 1e10)
    ))


#Resample with prior:
stable_past_9 = samps_time >= 9

from scipy.integrate import quad

_prior = lambda logT: (
    3.27086190404742*np.exp(-0.424033970670719 * logT) -
    10.8793430454878*np.exp(-0.200351029031774 * logT**2)
)
normalization = quad(_prior, a=9, b=np.inf)[0]

prior = lambda logT: _prior(logT)/normalization

# Let's generate random samples of that prior:

from scipy.interpolate import interp1d

n_samples = stable_past_9.sum()
bins = n_samples*4
top = 100.
bin_edges = np.linspace(9, top, num=bins)
cum_values = [0] + list(np.cumsum(prior(bin_edges)*(bin_edges[1] - bin_edges[0]))) + [1]
bin_edges = [9.] +list(bin_edges)+[top]
inv_cdf = interp1d(cum_values, bin_edges)
r = np.random.rand(n_samples)
samples = inv_cdf(r)

samps_time[stable_past_9] = samples

#min of sampled mu
# outs = np.min(time, 2)[..., 0].T

#min of samples of sampled mu
outs = np.min(samps_time, 2).T

#weighted samples
# chosen_samp = np.min(time[..., 0], 2)
# outs_std = np.array(
    # [[time[i, j, np.argmin(time[i, j, :, 0]), 0]
     # for j in range(time.shape[1])]
     # for i in range(time.shape[0])])
# outs_avg = np.average(chosen_samp, 0, weights=1/outs_std**2)

log_t_exit = np.log10(t_exit)

import seaborn as sns

cleaned = dict(
    median=[],
    xgb=[],
    average=[],
    l=[],
    u=[],
    ll=[],
    uu=[],
    true=[],
    delta=[],
    p12=[],
    p23=[],
    m1=[],
    m2=[],
    m3=[],
)
for i in range(len(outs)):
    cleaned['true'].append(log_t_exit[i])
    cleaned['delta'].append(delta[i])
    a1 = a_init_p[0, i]
    a2 = a_init_p[1, i]
    a3 = a_init_p[2, i]
    p12 = (a1/a2)**(3./2)
    p23 = (a2/a3)**(3./2)
    m1 = m_planet
    m2 = m_planet
    m3 = m_planet
    cleaned['p12'].append(p12)
    cleaned['p23'].append(p23)
    cleaned['m1'].append(m1)
    cleaned['m2'].append(m2)
    cleaned['m3'].append(m3)
    cleaned['xgb'].append(xgb_predictions[i])
    
    if log_t_exit[i] <= 4.0:
        cleaned['average'].append(log_t_exit[i])
        cleaned['median'].append(log_t_exit[i])
        cleaned['l'].append(log_t_exit[i])
        cleaned['u'].append(log_t_exit[i])
        cleaned['ll'].append(log_t_exit[i])
        cleaned['uu'].append(log_t_exit[i])
    elif outs[i] is not None:
        cleaned['average'].append(np.average(outs[i]))
        cleaned['median'].append(np.median(outs[i]))
        cleaned['l'].append(np.percentile(outs[i], 50+68/2))
        cleaned['u'].append(np.percentile(outs[i], 50-68/2))
        cleaned['ll'].append(np.percentile(outs[i], 50+95/2))
        cleaned['uu'].append(np.percentile(outs[i], 50-95/2))
    else:
        cleaned['average'].append(4.)
        cleaned['median'].append(4.)
        cleaned['l'].append(4.)
        cleaned['u'].append(4.)
        cleaned['ll'].append(4.)
        cleaned['uu'].append(4.)

cleaned = pd.DataFrame(cleaned)

for key in 'average median l u ll uu'.split(' '):
    cleaned.loc[cleaned['true']<=4.0, key] = cleaned.loc[cleaned['true']<=4.0, 'true']

    
import glob

from matplotlib import ticker


# +
cleaned['petit'] = np.log10(pd.Series([Tsurv(
        *list(cleaned[['p12', 'p23']].iloc[i]),
        [m_planet, m_planet, m_planet],
        res=False,
        fudge=1,
        m0=1
    )
    for i in range(len(cleaned))]))

cleaned['petitf'] = np.log10(pd.Series([Tsurv(
        *list(cleaned[['p12', 'p23']].iloc[i]),
        [m_planet, m_planet, m_planet],
        res=False,
        fudge=2,
        m0=1
    )
     for i in range(len(cleaned))]))

# +
# petit = Tsurv(p12, p23, [m1, m2, m3])
# petit = np.nan_to_num(np.log10(Tsurv), posinf=1e9, neginf=0.0)
# cleaned['petit'].append(petit)
# petit = Tsurv(p12, p23, [m1, m2, m3], fudge=2)
# petit = np.nan_to_num(np.log10(Tsurv), posinf=1e9, neginf=0.0)
# cleaned['petitf'].append(petit)
# -






import time
cleaned.to_csv(f'cur_plot_dataset_{time.time()}.csv')
make_plot(cleaned, version)
