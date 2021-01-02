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

# +
import glob
import matplotlib as mpl
mpl.use('agg')
import seaborn as sns
sns.set_style('darkgrid')
from matplotlib import pyplot as plt
import sys
sys.path.append('../')

import spock_reg_model
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateLogger, ModelCheckpoint

import torch
import numpy as np
from scipy.stats import truncnorm

import time
from tqdm.notebook import tqdm
# -

import sys
version = int(sys.argv[1])
base = f'../pretrained/steps=300000_*_v{version}_*_output.pkl'
import glob
swag_models = [spock_reg_model.load_swag(filename) for filename in glob.glob(base)]


for swag_model in swag_models:
    swag_model.load(swag_model.w_avg)

# + attributes={"classes": [], "id": "", "n": "20"}
import matplotlib.pyplot as plt
plt.switch_backend('agg')

# + attributes={"classes": [], "id": "", "n": "23"}
from tqdm.notebook import tqdm

# + attributes={"classes": [], "id": "", "n": "24"}

swag_models[0].make_dataloaders()


# +

val_dataloader = swag_models[0]._val_dataloader
# -

from torch.autograd import Variable, grad


# +
def partforward(self, x):
    summary_stats = self.compute_summary_stats(x)
    if self.fix_megno:
        summary_stats = torch.cat([summary_stats, megno_avg_std], dim=1)

    self._cur_summary = summary_stats

    #summary is (batch, feature)
    self._summary_kl = (1/2) * (
            summary_stats**2
            + torch.exp(self.summary_noise_logvar)[None, :]
            - self.summary_noise_logvar[None, :]
            - 1
        )


    mu, std = self.predict_instability(summary_stats)
    #Each is (batch,)

    return torch.cat((mu, std), dim=1)


def gradforward(self, x):
        if self.fix_megno or self.fix_megno2:
            if self.fix_megno:
                megno_avg_std = self.summarize_megno(x)
            #(batch, 2)
            x = self.zero_megno(x)

        if not self.include_mmr:
            x = self.zero_mmr(x)

        if not self.include_nan:
            x = self.zero_nan(x)

        if not self.include_eplusminus:
            x = self.zero_eplusminus(x)

        if self.random_sample:
            x = self.augment(x)
        #x is (batch, time, feature)
        
        x = Variable(x, requires_grad=True)
        
        mu = partforward(self, x)[:, 0]
        return grad(mu.sum(), x)[0], mu


# -

from tqdm.auto import tqdm

# +
importance_for_best_feats = []

for swag_model in tqdm(swag_models):
    saliency_map = torch.cat([gradforward(swag_model.cpu(), X_sample.cpu())[0] for X_sample, _ in val_dataloader])
#     print(saliency_map.shape)
    # importance_for_best_feat = torch.abs(saliency_map).mean((0, 1)).clone().detach().cpu().numpy()
    # importance_for_best_feat = np.median((saliency_map).clone().cpu().detach().std(1).numpy(), 0)
    importance_for_best_feats.append((saliency_map**2).mean((0, 1)).clone().detach().cpu().numpy())
#     importance_for_best_feats.append((saliency_map).std((0, 1)).clone().detach().cpu().numpy())

    
# -

importance_for_best_feats = np.array(importance_for_best_feats)**0.5

importance_for_best_feats /= importance_for_best_feats.sum(1)[:, None]

importance_for_best_feat = np.average(importance_for_best_feats, 0)

labels = ['time', 'e+_near', 'e-_near', 'max_strength_mmr_near', 'e+_far', 'e-_far', 'max_strength_mmr_far', 'megno', 'a1', 'e1', 'i1', 'cos_Omega1', 'sin_Omega1', 'cos_pomega1', 'sin_pomega1', 'cos_theta1', 'sin_theta1', 'a2', 'e2', 'i2', 'cos_Omega2', 'sin_Omega2', 'cos_pomega2', 'sin_pomega2', 'cos_theta2', 'sin_theta2', 'a3', 'e3', 'i3', 'cos_Omega3', 'sin_Omega3', 'cos_pomega3', 'sin_pomega3', 'cos_theta3', 'sin_theta3', 'm1', 'm2', 'm3', 'nan_mmr_near', 'nan_mmr_far', 'nan_megno']

angle_idx = [i for i in range(len(labels)) if 'sin' in labels[i] or 'cos' in labels[i]]
labels_to_combine = [angle_idx[j:j+2] for j in range(0, len(angle_idx), 2) if j+2 <= len(angle_idx)]

str_labels = [r'$t$', r'$e_+$ near', r'$e_-$ near', r'max_strength_mmr near', r'$e_+$ far', r'$e_-$ far', r'max_strength_mmr far', r'megno', r'$a_1$', r'$e_1$', r'$i_1$', r'$\cos(\Omega_1)$', r'$\sin(\Omega_1)$', r'$\cos(\omega_1)$', r'$\sin(\omega_1)$', r'$\cos(\theta_1)$', r'$\sin(\theta_1)$', r'$a_2$', r'$e_2$', r'$i_2$', r'$\cos(\Omega_2)$', r'$\sin(\Omega_2)$', r'$\cos(\omega_2)$', r'$\sin(\omega_2)$', r'$\cos(\theta_2)$', r'$\sin(\theta_2)$', r'$a_3$', r'$e_3$', r'$i_3$', r'$\cos(\Omega_3)$', r'$\sin(\Omega_3)$', r'$\cos(\omega_3)$', r'$\sin(\omega_3)$', r'$\cos(\theta_3)$', r'$\sin(\theta_3)$', r'$m_1$', r'$m_2$', r'$m_3$', r'$nan_mmr near$', r'$nan_mmr far$', r'$nan_megno$']


# +
new_importance = []
new_labels = []
new_str_labels = []
for i in range(len(importance_for_best_feat)):
    if i not in angle_idx:
        new_importance.append(importance_for_best_feat[i])
        new_labels.append(labels[i])
        new_str_labels.append(str_labels[i])
            
for i, j in labels_to_combine:
    new_importance.append(importance_for_best_feat[i] + importance_for_best_feat[j])
#     new_labels.append(labels[i] + ', ' + labels[j])
#     new_str_labels.append(str_labels[j] + ', ' + str_labels[i])
    new_labels.append(labels[i].split('_')[1])# + ', ' + labels[j])
    new_str_labels.append('$' + str_labels[j].split('(')[1].split(')')[0] + '$')
new_importance = np.array(new_importance)
# -

sort_idx = np.argsort(importance_for_best_feat)
new_sort_idx = np.argsort(new_importance)

from copy import deepcopy as copy

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


# +

# %matplotlib inline
plt.style.use('default')
sns.set_style('white')
plt.rc('font', family='serif')
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.figure(figsize=(4, 4), dpi=500)

# plt.barh(
#     y=np.arange(len([i for i in sort_idx if (
#         'megno' not in labels[i] and
#         'nan' not in labels[i] and
#         'mmr' not in labels[i]
#     )]))[::-1],
#     width=importance_for_best_feat[[i for i in sort_idx if (
#         'megno' not in labels[i] and
#         'nan' not in labels[i] and
#         'mmr' not in labels[i]
#     )]],
#     tick_label=[str_labels[i] for i in sort_idx if (
#         'megno' not in labels[i] and
#         'nan' not in labels[i] and
#         'mmr' not in labels[i]
#     )])

#TODO - modify for different label sets!
actual_idx = [i for i in new_sort_idx if (
        'megno' not in new_str_labels[i] and
        'nan' not in new_str_labels[i] and
        'mmr' not in new_str_labels[i] and
        'e_+' not in new_str_labels[i] and
        'e_-' not in new_str_labels[i]
    )]

plt.barh(
    y=np.arange(len(actual_idx))[::-1],
    width=new_importance[actual_idx],
    tick_label=[new_str_labels[i] for i in actual_idx],
    color=colors[2, 2])
from matplotlib import ticker
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

ax.grid(axis='x', color='w')
ax.xaxis.set_zorder(100)

# plt.axis('off')
ax.set_ylim(-1, len(actual_idx)+1)
plt.xlabel(r'Feature importance')
plt.tight_layout()
plt.savefig('importance.pdf')
plt.show()
# -




