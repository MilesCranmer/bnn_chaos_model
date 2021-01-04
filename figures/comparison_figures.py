# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: main2
#     language: python
#     name: main2
# ---

from petit20_survival_time import Tsurv

"""nu12, nu23 : Initial period ratios
   masses : planet masses"""

# +
#!/usr/bin/env python
# coding: utf-8

import glob
import seaborn as sns
import matplotlib as mpl
mpl.use('agg')
from matplotlib import pyplot as plt
import sys
sys.path.append('..')

import spock_reg_model
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateLogger, ModelCheckpoint
from icecream import ic

import torch
import numpy as np
from scipy.stats import truncnorm

import time
from tqdm.notebook import tqdm

import fit_trunc_dist

from custom_cmap import custom_cmap



# +
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

# Let's load in the data with summary features:
import sys


base = '../data/summary_features/'

from filenames import cdataset, cdataset_rand

import fire

def is_random(random=False):
    return random

random = fire.Fire(is_random)

# +
# random = True
# -

if random:
    datasets = cdataset + cdataset_rand
else:
    datasets = cdataset

feats = [
    '/orbsummaryfeaturesxgbNorbits10000.0Nout1000window10',
#     '/resparamsv5Norbits10000.0Nout1000window10',
    '/featuresNorbits10000.0Nout1000trio',
    '/additional_featuresNorbits10000.0Nout1000trio'
]


import pandas as pd

# +
data = []
len_random = 'none'
columns = None
for dataset in datasets:
    try:
        print("Loading from", base + dataset + feats[0] + '/trainingdata.csv')
        cdata = pd.concat(
            [pd.read_csv(base + dataset + feat + '/trainingdata.csv')
              for feat in feats], axis=1)

        cdata = pd.concat([cdata, pd.read_csv(base+dataset+'/featuresNorbits10000.0Nout1000trio/massratios.csv')], axis=1)
        cdata = pd.concat([cdata, pd.read_csv(base+dataset+'/featuresNorbits10000.0Nout1000trio/labels.csv')], axis=1)
        if dataset == 'random':
            len_random = len(cdata)
            columns = cdata.columns
            print(f'Random dataset has {len_random} rows')
            print(np.log10(cdata[['instability_time', 'shadow_instability_time']]).describe().to_markdown())
            print(cdata.iloc[0])
        data.append(cdata)
    except:
        print(f"Skipping {dataset}")
    
if random:
    data = pd.concat([cdata[columns] for cdata in data])
else:
    data = pd.concat(data)

# +
data['perturbed'] = np.average(
    np.log10(data[['instability_time', 'shadow_instability_time']]),
    1)
data['perturbed'] += np.random.randn(len(data))*0.43

nu12 = np.array((data['avg_a1']/data['avg_a2'])**(3./2))
nu23 = np.array((data['avg_a2']/data['avg_a3'])**(3./2))
m1 = np.array(data['m1'])
m2 = np.array(data['m2'])
m3 = np.array(data['m3'])
petit_result = np.array([Tsurv(nu12[i], nu23[i], [m1[i], m2[i], m3[i]]) for i in range(len(m1))])
data['petit'] = petit_result

data['petit'] = np.nan_to_num(data['petit'], posinf=1e9, neginf=1e9, nan=1e9)
data['petit'] = np.log10(data['petit'])
# -

#Obertas results:
data['obertas_delta'] = (
    (data[['avg_beta12', 'avg_beta23']]).min(1)
)
b = 0.951 
c = -1.202
data['obertas'] = b * data['obertas_delta'] + c

from xgboost import XGBRegressor, XGBClassifier
from sklearn.linear_model import LinearRegression

spock_xgb_params = {'base_score': 0.5,
 'booster': 'gbtree',
 'colsample_bylevel': 1,
 'colsample_bynode': 1,
 'colsample_bytree': 1,
 'gamma': 0,
 'gpu_id': -1,
 'importance_type': 'gain',
 'interaction_constraints': '',
 'learning_rate': 0.05,
 'max_delta_step': 0,
 'max_depth': 13,
 'min_child_weight': 5,
 'missing': np.nan,
 'monotone_constraints': '()',
 'n_estimators': 100,
 'n_jobs': 0,
 'num_parallel_tree': 1,
 'random_state': 0,
 'reg_alpha': 0,
 'reg_lambda': 1,
 'scale_pos_weight': 1,
 'subsample': 0.95,
 'tree_method': 'auto',
 'validate_parameters': False,
 'verbosity': None}


# Note: can't take all hyperparams the same, since
# they were optimized on a different training set - there
# is some overlap here. So much re-optimize hyperparams (use auto method):

class SimpleSKLearn(object):
    def __init__(self):
        super().__init__()
    
    def fit(self, *args):
        return self
    
    def predict(self, X):
        return X[:, 0]


models = {
    'Obertas+17': {
        'model': SimpleSKLearn(),#LinearRegression(),
        'features': ['obertas'],#['avg_beta12', 'avg_beta23'],
        'title': 'Obertas et al. (2017)'
    },
    'XGBoost': {
        'model': XGBRegressor(**spock_xgb_params),
        'features': 
"EMcrossnear MMRstrengthnear MMRstrengthfar EPstdnear".split(' ')
+ "EMfracstdfar EMfracstdnear EMcrossfar EPstdfar MEGNOstd".split(' ')
+ "MEGNO".split(' '),
        'title': 'Modified T20'
    },
    'XGBoost_classifier': {
        'model': XGBClassifier(**spock_xgb_params),
        'features': 
"EMcrossnear MMRstrengthnear MMRstrengthfar EPstdnear".split(' ')
+ "EMfracstdfar EMfracstdnear EMcrossfar EPstdfar MEGNOstd".split(' ')
+ "MEGNO".split(' '),
        'title': 'T20'
    },
    'Ideal': {
        'model': SimpleSKLearn(),
        'features': ['perturbed'],
        'title': 'Theoretical limit'
    },
    'Petit+20': {
        'model': SimpleSKLearn(),
        'features': ['petit'],
        'title': 'Petit et al. (2020)'
    }
}

from copy import deepcopy as copy

from sklearn.model_selection import train_test_split

import dask

import seaborn as sns

sns.set_style('white')
plt.rc('font', family='serif')



# +
cmodelstr = 'XGBoost'
cfeat = models[cmodelstr]['features']
cdata = copy(data[cfeat + ['instability_time', 'shadow_instability_time']])
cdata[['instability_time', 'shadow_instability_time']] = np.log10(
    cdata[['instability_time', 'shadow_instability_time']]
)
cmodel = models[cmodelstr]['model']
cdata.replace([np.inf, -np.inf], np.nan)
if random:
    cX = np.array(cdata.iloc[:-len_random].query('instability_time >= 4'
                      '& shadow_instability_time >= 4'
        )[cfeat]).copy()
    cy = np.array(cdata.iloc[:-len_random].query('instability_time >= 4'
                      '& shadow_instability_time >= 4'
        )[['instability_time', 'shadow_instability_time']]).copy()


    cX_random = np.array(cdata.iloc[-len_random:].query('instability_time >= 4'
                      '& shadow_instability_time >= 4'
        )[cfeat]).copy()
    cy_random = np.array(cdata.iloc[-len_random:].query('instability_time >= 4'
                      '& shadow_instability_time >= 4'
        )[['instability_time', 'shadow_instability_time']]).copy()

else:
    cdata = cdata.query('instability_time >= 4'
                      '& shadow_instability_time >= 4'
    )
    cy = np.array(cdata[['instability_time', 'shadow_instability_time']]).copy()
    cX = np.array(cdata[cfeat]).copy()

# +
# np.log10(data.iloc[-len_random:].query('instability_time >= 10**4'
#                       '& shadow_instability_time >= 10**4'
#     )[['instability_time', 'shadow_instability_time']]).hist('instability_time')

# +
# plt.hist(cy_random.ravel())
# -





# +
# %matplotlib inline
    
for cmodelstr in ['Obertas+17', 'Petit+20', 'XGBoost', 'XGBoost_classifier', 'Ideal']:

    cfeat = models[cmodelstr]['features']
    cdata = copy(data[cfeat + ['instability_time', 'shadow_instability_time']])
    cdata[['instability_time', 'shadow_instability_time']] = np.log10(
        cdata[['instability_time', 'shadow_instability_time']]
    )
    cmodel = models[cmodelstr]['model']
    cdata.replace([np.inf, -np.inf], np.nan)
    if random:
        cX = np.array(cdata.iloc[:-len_random].query('instability_time >= 4'
                          '& shadow_instability_time >= 4'
            )[cfeat]).copy()
        cy = np.array(cdata.iloc[:-len_random].query('instability_time >= 4'
                          '& shadow_instability_time >= 4'
            )[['instability_time', 'shadow_instability_time']]).copy()


        cX_random = np.array(cdata.iloc[-len_random:].query('instability_time >= 4'
                          '& shadow_instability_time >= 4'
            )[cfeat]).copy()
        cy_random = np.array(cdata.iloc[-len_random:].query('instability_time >= 4'
                          '& shadow_instability_time >= 4'
            )[['instability_time', 'shadow_instability_time']]).copy()

    else:
        cdata = cdata.query('instability_time >= 4'
                          '& shadow_instability_time >= 4'
        )
        cy = np.array(cdata[['instability_time', 'shadow_instability_time']]).copy()
        cX = np.array(cdata[cfeat]).copy()

    print(f'{cmodelstr} gets {np.average(cy)}')


    X_train, X_test, y_train, y_test = train_test_split(
        cX, cy, random_state=0
    )

    if random:
        X_test = cX_random
        y_test = cy_random

    nan_mask_train = ~np.any(np.isnan(X_train), 1)
    nan_mask_test = ~np.any(np.isnan(X_test), 1)
    X_train = X_train[nan_mask_train]
    y_train = y_train[nan_mask_train]
    X_test = X_test[nan_mask_test]
    y_test = y_test[nan_mask_test]
    print(f'{cmodelstr} is size {len(X_test)}')
    print(f'{(y_test<8.99).sum()} are below 9; {(y_test>8.99).sum()} are above')

    print(f'{cmodelstr} gets {np.average(y_test)}')

    import einops
    # Make shadow runs into separate sample:
    y_train = einops.rearrange(y_train, 'sample run -> (sample run)')
    y_test  = einops.rearrange(y_test,  'sample run -> (sample run)')
    X_train = einops.repeat(   X_train, 'sample feature -> (sample run) feature', run=2)
    X_test  = einops.repeat(   X_test,  'sample feature -> (sample run) feature', run=2)

    if cmodelstr == 'XGBoost_classifier':
        above_9 = (y_train > 8.99).astype(np.int)
        cmodel.fit(X_train, above_9)

        yp_test = cmodel.predict_proba(X_test)[:, 1]
        # Need to map to 9.0 or not.
        # p probability of being 11.5, (1-p) probability of being 6.5?
        # preds = yp_test * 11.5 + (1-yp_test) * 6.5
        # Simple mapping.
        p = yp_test
        from scipy.special import erfinv
        from scipy.optimize import minimize_scalar

        # Find best sigma:
        opt_func = lambda s: np.average(
            np.square(
                np.clip(9.0 + s * np.sqrt(2) * erfinv(2*p - 1), 4, 9)
                - truths)
            )
        res = minimize_scalar(opt_func)
        assumed_sigma = res.x
        print(f"Found best sigma for XGBoost = {assumed_sigma}")
        preds = 9.0 + assumed_sigma * np.sqrt(2) * erfinv(2*p - 1)
    else:
        y_train = np.clip(y_train, 4, 9)
        y_train[y_train > 8.99] += 1.0
        cmodel.fit(X_train, y_train)
        yp_test = cmodel.predict(X_test)
        preds = yp_test


    if cmodelstr == 'XGBoost':
        cmodel.save_model('cur_xgboost_regressor.bin')
        print(f"Saved XGBoost model! It expects {cfeat} as input.")

    truths = y_test
    print(f'{cmodelstr} gets {np.average(truths)}')

    py = preds
    px = truths
    if cmodelstr == 'Ideal':
        shape = px.shape
        sigmas_to_sample = np.loadtxt('../data/sigmas.txt')
        sampled_sigmas = sigmas_to_sample[
                np.random.randint(0, len(sigmas_to_sample), size=shape)
            ].reshape(*shape)
        py = (
            (px <= 8.99) * (px + np.random.randn(*shape) * sampled_sigmas) + 
            (px >  8.99) * (10.0 + np.random.randn(*shape) * sampled_sigmas)
        )

    py = np.clip(py, 4, 9)
    px = np.clip(px, 4, 9)


    from scipy.stats import gaussian_kde
    import seaborn as sns

    ppx = px.copy()
    ppy = py.copy()

    fig = plt.figure(figsize=(4, 4), 
                     dpi=300,
                     constrained_layout=True)

    from icecream import ic
    mse = np.average(np.square(ppx[ppx<8.99] - ppy[ppx<8.99]))
    print(models[cmodelstr]['title'], f'gets RMSE of {np.average(np.square(ppx[ppx<8.99] - ppy[ppx<8.99]))**0.5:.2f}')

    from sklearn.metrics import roc_auc_score
    ######################################################
    # Classification scores:
    if cmodelstr == 'XGBoost_classifier':
        classifications = yp_test
        roc = roc_auc_score(truths > 8.99, classifications)
    elif cmodelstr == 'Ideal':
        ## Use nominal time to classify instability time:
        nominal, shadow = (lambda x: (x[:, 0].copy(), x[:, 1].copy()))(
                einops.rearrange(y_test,  '(sample run) -> sample run', run=2)
            )
        shape = nominal.shape
        sigmas_to_sample = np.loadtxt('../data/sigmas.txt')
        sampled_sigmas = sigmas_to_sample[
                np.random.randint(0, len(sigmas_to_sample), size=shape)
            ].reshape(*shape)
        shadow = (
            (nominal <= 8.99) * (nominal + np.random.randn(*shape) * sampled_sigmas) + 
            (nominal >  8.99) * (10.00 + np.random.randn(*shape) * sampled_sigmas)
        )
        # shadow = np.copy(shadow)
        shadow[shadow > 8.99] = 10.0
        sigma = min([2.5, np.sqrt(mse)])
        shadow_classification =  norm.cdf( (shadow - 9.0) / sigma)
        nominal_classification = nominal > 8.99
        roc = roc_auc_score(nominal_classification, shadow_classification)
    else:
        from scipy.stats import norm
        sigma = min([2.5, np.sqrt(mse)])
        classifications = norm.cdf( (yp_test - 9.0) / sigma)
        roc = roc_auc_score(truths > 8.99, classifications)

    print(models[cmodelstr]['title'], f'gets ROC AUC of {roc:.3f}')
    ######################################################

    ######################################################
    # Bias scores:
    tmpdf = pd.DataFrame({'true': ppx, 'pred': ppy})
    for lo in range(4, 9):
        hi = lo + 0.99
        considered = tmpdf.query(f'true>{lo} & true<{hi}')
        print(f"Between {lo} and {hi}, the bias is {np.average(considered['pred'] - considered['true']):.3f}")
    ######################################################




    #colors[2, 3]
    main_shade = 3
    main_color = colors[2, main_shade]
    print(f"Total number of plotted points: {len(ppx)}")
    g = sns.jointplot(ppx, ppy,
                    alpha=1.0,# ax=ax,
                      color=main_color,
                    s=0.0,
                    xlim=(3+0.9, 10-0.9),
                    ylim=(3+0.9, 10-0.9),
                    marginal_kws=dict(bins=15, binrange=(4, 9)),
                   )

    ax = g.ax_joint
    
    #Transparency:
    transparency_adjuster = 0.5 * 0.2
    point_color = list(main_color) + [transparency_adjuster]
         

    print(models[cmodelstr]['title'], f'has length {len(ppx)}')
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
    plt.suptitle(models[cmodelstr]['title'], y=1.0)
    plt.tight_layout()
    if random:
        plt.savefig(f'figures/comparison_{cmodelstr}_random.png', dpi=300)
    else:
        plt.savefig(f'figures/comparison_{cmodelstr}.png', dpi=300)

# -

# RMS plot not done, since this model optimizes mu, but other
# model optimizes std as well as mu. Only fair
# comparison is above plot.

pd.DataFrame({'x': ppy}).describe()


