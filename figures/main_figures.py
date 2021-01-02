#!/usr/bin/env python
# coding: utf-8
import glob
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
plt.style.use('science')

import spock_reg_model
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateLogger, ModelCheckpoint

import torch
import numpy as np
from scipy.stats import truncnorm

import time
from tqdm.notebook import tqdm
from icecream import ic

import fit_trunc_dist

from custom_cmap import custom_cmap

import sys

# +
# manual_argv = "--version 50 --total_steps 300000 --swa_steps 50000 --angles --no_mmr --no_nan --no_eplusminus --seed -1 --plot".split(' ')

# +
# del sys.argv[1:]
# sys.argv.extend(manual_argv)
# -

from parse_swag_args import parse
args, checkpoint_filename = parse(glob=True)

swag_ensemble = [
    spock_reg_model.load_swag(fname).cuda()
    for fname in glob.glob('*' + checkpoint_filename + '*output.pkl') #
]

if args.plot_random:
    checkpoint_filename += '_random'


import matplotlib.pyplot as plt
plt.switch_backend('agg')

from tqdm.notebook import tqdm

colorstr = """*** Primary color:

   shade 0 = #A0457E = rgb(160, 69,126) = rgba(160, 69,126,1) = rgb0(0.627,0.271,0.494)
   shade 1 = #CD9CBB = rgb(205,156,187) = rgba(205,156,187,1) = rgb0(0.804,0.612,0.733)
   shade 2 = #BC74A1 = rgb(188,116,161) = rgba(188,116,161,1) = rgb0(0.737,0.455,0.631)
   shade 3 = #892665 = rgb(137, 38,101) = rgba(137, 38,101,1) = rgb0(0.537,0.149,0.396)
   shade 4 = #74104F = rgb(116, 16, 79) = rgba(116, 16, 79,1) = rgb0(0.455,0.063,0.31)

*** Secondary color (1):

   shade 0 = #CDA459 = rgb(205,164, 89) = rgba(205,164, 89,1) = rgb0(0.804,0.643,0.349)
   shade 1 = #FFE9C2 = rgb(255,233,194) = rgba(255,233,194,1) = rgb0(1,0.914,0.761)
   shade 2 = #F1D195 = rgb(241,209,149) = rgba(241,209,149,1) = rgb0(0.945,0.82,0.584)
   shade 3 = #B08431 = rgb(176,132, 49) = rgba(176,132, 49,1) = rgb0(0.69,0.518,0.192)
   shade 4 = #956814 = rgb(149,104, 20) = rgba(149,104, 20,1) = rgb0(0.584,0.408,0.078)

*** Secondary color (2):

   shade 0 = #425B89 = rgb( 66, 91,137) = rgba( 66, 91,137,1) = rgb0(0.259,0.357,0.537)
   shade 1 = #8C9AB3 = rgb(140,154,179) = rgba(140,154,179,1) = rgb0(0.549,0.604,0.702)
   shade 2 = #697DA0 = rgb(105,125,160) = rgba(105,125,160,1) = rgb0(0.412,0.49,0.627)
   shade 3 = #294475 = rgb( 41, 68,117) = rgba( 41, 68,117,1) = rgb0(0.161,0.267,0.459)
   shade 4 = #163163 = rgb( 22, 49, 99) = rgba( 22, 49, 99,1) = rgb0(0.086,0.192,0.388)

*** Complement color:

   shade 0 = #A0C153 = rgb(160,193, 83) = rgba(160,193, 83,1) = rgb0(0.627,0.757,0.325)
   shade 1 = #E0F2B7 = rgb(224,242,183) = rgba(224,242,183,1) = rgb0(0.878,0.949,0.718)
   shade 2 = #C9E38C = rgb(201,227,140) = rgba(201,227,140,1) = rgb0(0.788,0.89,0.549)
   shade 3 = #82A62E = rgb(130,166, 46) = rgba(130,166, 46,1) = rgb0(0.51,0.651,0.18)
   shade 4 = #688C13 = rgb(104,140, 19) = rgba(104,140, 19,1) = rgb0(0.408,0.549,0.075)"""

colors = []
shade = 0
for l in colorstr.replace(' ', '').split('\n'):
    elem = l.split('=')
    if len(elem) != 5: continue
    if shade == 0:
        new_color = []
    rgb = lambda x, y, z: np.array([x, y, z]).astype(np.float32)
    
    new_color.append(eval(elem[2]))
    
    shade += 1
    if shade == 5:
        colors.append(np.array(new_color))
        shade = 0
colors = np.array(colors)/255.0

if len(swag_ensemble) == 0:
    raise ValueError(checkpoint_filename + " not found!")

swag_ensemble[0].make_dataloaders()
if args.plot_random:
    assert swag_ensemble[0].ssX is not None
    from copy import deepcopy as copy
    tmp_ssX = copy(swag_ensemble[0].ssX)
    # print(tmp_ssX.mean_)
    if args.train_all:
        swag_ensemble[0].make_dataloaders(
            ssX=swag_ensemble[0].ssX,
            train=True,
            plot_random=True)
    else:
        swag_ensemble[0].make_dataloaders(
            ssX=swag_ensemble[0].ssX,
            train=False,
            plot_random=True) #train=False means we show the whole dataset (assuming we don't train on it!)

    # print(swag_ensemble[0].ssX.mean_)
    assert np.all(tmp_ssX.mean_ == swag_ensemble[0].ssX.mean_)

val_dataloader = swag_ensemble[0]._val_dataloader

def sample_full_swag(X_sample):
    """Pick a random model from the ensemble and sample from it
    within each model, it samples from its weights."""
    
    swag_i = np.random.randint(0, len(swag_ensemble))
    swag_model = swag_ensemble[swag_i]
    swag_model.eval()
    swag_model.w_avg = swag_model.w_avg.cuda()
    swag_model.w2_avg = swag_model.w2_avg.cuda()
    swag_model.pre_D = swag_model.pre_D.cuda()
    swag_model.cuda()
    out = swag_model.forward_swag(X_sample, scale=0.5)
    return out

truths = []
preds = []
raw_preds = []

nc = 0
losses = 0.0
do_sample = True
for X_sample, y_sample in tqdm(val_dataloader):
    X_sample = X_sample.cuda()
    y_sample = y_sample.cuda()
    nc += len(y_sample)
    truths.append(y_sample.cpu().detach().numpy())

    raw_preds.append(
        np.array([sample_full_swag(X_sample).cpu().detach().numpy() for _ in range(2000)])
    )

truths = np.concatenate(truths)

_preds = np.concatenate(raw_preds, axis=1)

import time
# numpy sampling is way too slow:

from functools import partial

def fast_truncnorm(
        loc, scale, left=np.inf, right=np.inf,
        d=10000, nsamp=50, seed=0):
    """Fast truncnorm sampling.
    
    Assumes scale and loc have the desired shape of output.
    length is number of elements.
    Select nsamp based on expecting at last one sample
        to fit within your (left, right) range.
    Select d based on memory considerations - need to operate on
        a (d, nsamp) array.
    """
    oldscale = scale
    oldloc = loc
    
    scale = scale.reshape(-1)
    loc = loc.reshape(-1)
    samples = np.zeros_like(scale)
    start = 0

    for start in range(0, scale.shape[0], d):

        end = start + d
        if end > scale.shape[0]:
            end = scale.shape[0]
        
        cd = end-start
        rand_out = np.random.randn(
            nsamp, cd
        )

        rand_out = (
            rand_out * scale[None, start:end]
            + loc[None, start:end]
        )
        
        #rand_out is (nsamp, cd)
        if right == np.inf:
            mask = (rand_out > left)
        elif left == np.inf:
            mask = (rand_out < right)
        else:
            mask = (rand_out > left) & (rand_out < right)
            
        first_good_val = rand_out[
            mask.argmax(0), np.arange(cd)
        ]
        
        samples[start:end] = first_good_val
        
    return samples.reshape(*oldscale.shape)

std = _preds[..., 1]
mean = _preds[..., 0]

loc = mean
scale = std

sample_preds = np.array(
        fast_truncnorm(np.array(mean), np.array(std),
               left=4, d=874000, nsamp=40));

stable_past_9 = sample_preds >= 9

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

sample_preds[stable_past_9] = samples

_preds.shape

import numpy as np
from numpy import sqrt, pi, exp
from scipy.special import erf
from scipy.optimize import minimize
from numba import jit, prange

# # expectation of samples
# preds = np.average(sample_preds, 0)
# stds = np.std(sample_preds, 0)

# # median of samples
# preds = np.median(sample_preds, 0)
# stds = (
#     (lambda x: 0.5*(np.percentile(x, q=50 + 68/2, axis=0) - np.percentile(x, q=50-68/2, axis=0)))
#     (sample_preds)
# )

# # median of dists
preds = np.median(_preds[..., 0], 0)
stds = np.median(_preds[..., 1], 0)

# # fit a truncated dist using avg, var
# tmp = fit_trunc_dist.find_mu_sig(sample_preds.T)
# preds = tmp[:, 0] 
# stds = tmp[:, 1]

# # with likelihood (slow)
# tmp = fit_trunc_dist.find_mu_sig_likelihood(sample_preds[:300, :].T)
# preds = tmp[:, 0]
# stds = tmp[:, 1]

# weighted average of mu
# w_i = 1/_preds[:, :, 1]**2
# w_i /= np.sum(w_i, 0)
# preds = np.average(_preds[:, :, 0], 0, weights=w_i)
# stds = np.average(_preds[:, :, 1]**2, 0)**0.5

# Check that confidence intervals are satisifed. Calculate mean and std of samples. Take abs(truths - mean)/std = sigma. The CDF of this distrubtion should match that of a Gaussian. Otherwise, rescale "scale". 

tmp_mask = (truths > 6) & (truths < 7) #Take this portion since its far away from truncated parts
averages = preds#np.average(sample_preds, 0)
gaussian_stds = stds#np.std(sample_preds, 0)
sigma = (truths[tmp_mask] - np.tile(averages, (2, 1)).T[tmp_mask])/np.tile(gaussian_stds, (2, 1)).T[tmp_mask]

np.save(checkpoint_filename + 'model_error_distribution.npy', sigma)

bins = 30
fig = plt.figure(figsize=(4, 4))
plt.hist(np.abs(sigma), bins=bins, range=[0, 2.5], density=True,
            color=colors[0, 3],
         alpha=1, label='Model error distribution')
np.random.seed(0)
plt.hist(np.abs(np.random.randn(len(sigma))), bins=bins, range=[0, 2.5], density=True,
            color=colors[1, 3],
         alpha=0.5, label='Gaussian distribution')
plt.ylim(0, 1.2)
plt.ylabel('Density', fontsize=14)
plt.xlabel('Error over sigma', fontsize=14)
# plt.xlabel('$|\mu_θ - y|/\sigma_θ$', fontsize=14)
plt.legend()
fig.savefig(checkpoint_filename + 'error_dist.pdf')


# Looks great! We didn't even need to tune it. Just use the same scale as the paper (0.5). Perhaps, however, with epistemic uncertainty, we will need to tune. 

from matplotlib.colors import LogNorm

def density_scatter(x, y, xlabel='', ylabel='', clabel='Sample Density', log=False,
    width_mult=1, bins=30, p_cut=None, update_rc=True, ax=None, fig=None, cmap='viridis', **kwargs):
    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    xy = np.array([x, y]).T
    px = xy[:, 0]
    py = xy[:, 1]
    if p_cut is not None:
        p = p_cut
        range_x = [np.percentile(xy[:, 0], i) for i in [p, 100-p]]
        range_y = [np.percentile(xy[:, 1], i) for i in [p, 100-p]]
        pxy = xy[(xy[:, 0] > range_x[0]) & (xy[:, 0] < range_x[1]) & (xy[:, 1] > range_y[0]) & (xy[:, 1] < range_y[1])]
    else:
        pxy = xy
    px = pxy[:, 0]
    py = pxy[:, 1]
    norm = None
    if log:
        norm = LogNorm()

    h, xedge, yedge, im = ax.hist2d(
        px, py, density=True, norm=norm,
        bins=[int(width_mult*bins), bins], cmap=cmap, **kwargs)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    fig.colorbar(im, ax=ax).set_label(clabel)
    fig.tight_layout()

    return fig, ax

_preds.shape

#confidences_to_plot = 'low med high vhigh vvhigh'.split(' ')
confidences_to_plot = ['low']

# %matplotlib inline

import einops

show_transparency = True

main_shade = 3
main_color = colors[2, main_shade]
off_color = colors[2, main_shade]


plt.style.use('default')
sns.set_style('white')
plt.rc('font', family='serif')

# +
for confidence in confidences_to_plot:
    py = preds
    py = np.clip(py, 4, 9)
    px = np.average(truths, 1)
        
    from scipy.stats import gaussian_kde

    import seaborn as sns

    mask = np.all(truths < 9.99, 1) # np.all(truths < 8.99, 1)

    if confidence != 'low':
        #tmp_std = np.std(sample_preds, 0)/py
        tmp_std = stds/py
        if confidence == 'high':
            mask = mask & ((tmp_std) < np.percentile(tmp_std[mask], 50))
        elif confidence == 'vhigh':
            mask = mask & ((tmp_std) < np.percentile(tmp_std[mask], 25))
        elif confidence == 'vvhigh':
            mask = mask & ((tmp_std) < np.percentile(tmp_std[mask], 10))
        elif confidence == 'med':
            mask = mask & ((tmp_std) < np.percentile(tmp_std[mask], 70))

    ppx = px[mask]
    ppy = py[mask]
    p_std = stds[mask]


    extra = ''
    if confidence != 'low':
        extra = ', '
        extra += {
            'med': '30th',
            'high': '50th',
            'vhigh': '75th',
            'vvhigh': '90th',
        }[confidence]
        extra += ' percentile confidence'
    title = 'Our model'+extra

    fig = plt.figure(figsize=(4, 4), 
                     dpi=300,
                     constrained_layout=True)
    # if args.plot_random:
        # ic('random')
        # ic(len(ppx))
        # alpha = min([0.05 * 72471 / len(ppx), 1.0])
    # else:
        # ic('not random')
        # ic(len(ppx))
        
#     alpha = min([0.05 * 8740 / len(ppx), 1.0])
#     ic(alpha, args.plot_random, len(ppx))
    alpha = 1.0

    #colors[2, 3]
    g = sns.jointplot(ppx, ppy,
                    alpha=alpha,# ax=ax,
                      color=main_color,
#                     hue=(ppy/p_std)**2,
                    s=0.0,
                    xlim=(3, 10),
                    ylim=(3, 10),
                    marginal_kws=dict(bins=15),
                   )

    ax = g.ax_joint
    snr = (ppy/p_std)**2
    relative_snr = snr / max(snr)
    point_color = relative_snr

    print(f'{confidence} confidence gets RMSE of {np.average(np.square(ppx[ppx < 8.99] - ppy[ppx < 8.99]))**0.5:.2f}')
    print(f'Weighted by SNR, this is: {np.average(np.square(ppx[ppx < 8.99] - ppy[ppx < 8.99]), weights=snr[ppx<8.99])**0.5:.2f}')

    ######################################################
    # Bias scores:
    tmpdf = pd.DataFrame({'true': ppx, 'pred': ppy, 'w': snr})
    for lo in range(4, 9):
        hi = lo + 0.99
        considered = tmpdf.query(f'true>{lo} & true<{hi}')
        print(f"Between {lo} and {hi}, the bias is {np.average(considered['pred'] - considered['true']):.3f}",
                f"and the weighted bias is {np.average(considered['pred'] - considered['true'], weights=considered['w']):.3f}")
    ######################################################
    
    #Transparency:
    if show_transparency:
        if args.plot_random:
            transparency_adjuster = 1.0 #0.5 * 0.2
        else:
            transparency_adjuster = 1.0
        point_color = np.concatenate(
            (einops.repeat(colors[2, 3], 'c -> row c', row=len(ppy)),
             point_color[:, None]*transparency_adjuster), axis=1)
    #color mode:
    else:
        point_color = np.einsum('r,i->ir', main_color, point_color) +\
            np.einsum('r,i->ir', off_color, 1-point_color)
         
    
    
    im = ax.scatter(
                ppx,
               ppy, marker='o',
               c=point_color,
               s=10,
               edgecolors='none'
              )
    ax.plot([4-3, 9+3], [4-3, 9+3], color='k')
    ax.plot([4-3, 9+3], [4+0.61-3, 9+0.61+3], color='k', ls='--')
    ax.plot([4-3, 9+3], [4-0.61-3, 9-0.61+3], color='k', ls='--')
    ax.set_xlim(3+0.9, 10-0.9)
    ax.set_ylim(3+0.9, 10-0.9)
    ax.set_xlabel('Truth') 
    ax.set_ylabel('Predicted')
    plt.suptitle(title, y=1.0)
    plt.tight_layout()
    
    
    if confidence == 'low':
        plt.savefig(checkpoint_filename + 'comparison.png', dpi=300)
    else:
        plt.savefig(checkpoint_filename + f'_{confidence}_confidence_' + 'comparison.png', dpi=300)

# +
import matplotlib as mpl

mymap = mpl.colors.LinearSegmentedColormap.from_list(
    'mine', [
        [1.0, 1.0, 1.0, 1.0],
        list(point_color[0, :3]) + [1.0]
    ], N=30
)
fig, ax = plt.subplots(figsize=(6, 1))
fig.subplots_adjust(bottom=0.5)

cmap = mymap
norm = mpl.colors.Normalize(vmin=snr.min(), vmax=snr.max())

cb1 = mpl.colorbar.ColorbarBase(ax,
                                cmap=cmap,
                                norm=norm,
                                orientation='horizontal')
cb1.set_label('SNR')
fig.show()
plt.savefig(checkpoint_filename + 'colorbar.png', dpi=300)
# -

plt.style.use('default')
plt.style.use('science')















# Idea: KDE plot but different stacked versions showing contours of the residual. Compare with other algorithms.

palette = sns.color_palette(['#892665', '#B08431', '#294475', '#82A62E'])
sns.set_palette(palette)
fig, ax = plt.subplots(1, 1, figsize=(4, 3))
sns.distplot((py-px)[(px<8.99)], hist=True, kde=True,
             bins=30, ax=ax,
             hist_kws={'edgecolor':'black', 'range': [-4, 4]},
             kde_kws={'linewidth': 4, 'color': 'k'})
ax.set_xlabel('Residual')
ax.set_ylabel('Probability')
ax.set_title('RMS residual under 9: %.3f'% (np.sqrt(np.average(np.abs(py-px)[px<9])),))

plt.xlim(-3, 3)
plt.ylim(0, 0.7)


fig.savefig(checkpoint_filename + 'residual.pdf')

labels = ['time', 'e+_near', 'e-_near', 'max_strength_mmr_near', 'e+_far', 'e-_far', 'max_strength_mmr_far', 'megno', 'a1', 'e1', 'i1', 'cos_Omega1', 'sin_Omega1', 'cos_pomega1', 'sin_pomega1', 'cos_theta1', 'sin_theta1', 'a2', 'e2', 'i2', 'cos_Omega2', 'sin_Omega2', 'cos_pomega2', 'sin_pomega2', 'cos_theta2', 'sin_theta2', 'a3', 'e3', 'i3', 'cos_Omega3', 'sin_Omega3', 'cos_pomega3', 'sin_pomega3', 'cos_theta3', 'sin_theta3', 'm1', 'm2', 'm3', 'nan_mmr_near', 'nan_mmr_far', 'nan_megno']


truths.shape#.reshape(-1)

from sklearn.metrics import roc_curve, roc_auc_score
plt.style.use('default')
plt.style.use('science')
fpr, tpr, _ = roc_curve(y_true=(truths>=9).reshape(-1),
                        y_score=np.average(np.tile(sample_preds, (2, 1, 1))>9, 1).transpose(1, 0).reshape(-1))
fig = plt.figure(figsize=(4, 4))
plt.plot(fpr, tpr, color=colors[0, 3])
plt.xlabel('fpr')
plt.ylabel('tpr')

import einops
y_roc = truths > 8.99

y_score = np.average(sample_preds>= 9, axis=0)
# ic(y_roc.shape, y_score.shape)

y_roc = einops.rearrange(y_roc, 'sample run -> (sample run)')
y_score = einops.repeat(y_score, 'sample -> (sample run)', run=2)

# ic(y_roc.shape, y_score.shape)
# # Use median of stds:
# preds = np.median(_preds[..., 0], 0)
# stds = np.median(_preds[..., 1], 0)
snr = np.median(_preds[..., 0], 0)**2 / np.median(_preds[..., 1], 0)**2

# Use std of samples:
# snr =  np.average(sample_preds, axis=0)**2/np.std(sample_preds, axis=0)**2
y_weight = einops.repeat(snr, 'sample -> (sample run)', run=2)


roc = roc_auc_score(
    y_true=y_roc,
    y_score=y_score,
)
weight_roc = roc_auc_score(
    y_true=y_roc,
    y_score=y_score,
    sample_weight=y_weight
)
plt.title('AUC ROC = %.3f'%(roc,))

print(f'Model gets ROC of {roc:.3f}')
print(f'Model gets weighted ROC of {weight_roc:.3f}')
# summary_writer.add_figure(
#     'roc_curve',
#     fig)
plt.xlim(0, 1)
plt.ylim(0, 1)
fig.savefig(checkpoint_filename + 'classification.pdf')
plt.style.use('seaborn')


