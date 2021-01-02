import numpy as np
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
from scipy.stats import truncnorm

import seaborn as sns
plt.style.use('science')

_prior = lambda logT: (
    3.27086190404742*np.exp(-0.424033970670719 * logT) -
    10.8793430454878*np.exp(-0.200351029031774 * logT**2)
)

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

def likelihood(x, mu, std, prior='flat'):
    a = (4  - mu)/std
    b = (12 - mu)/std
    base = truncnorm.pdf(x, a=a, b=b, loc=mu, scale=std)
    cdf_above_9 = base[x>=9].sum()
    base[x>=9] = 0.0
    if prior == 'flat':
        mask = (x>=9) & (x<=12)
        base[mask] = cdf_above_9 / mask.sum()
    elif prior == 'decay':
        base[x>=9] = np.exp(-(x[x>=9]-9))
        base[x>=9] /= base[x>=9].sum()
        base[x>=9] *= cdf_above_9
    elif prior == 'default':
        base[x>=9] = _prior(x[x>=9])
        base[x>=9] /= base[x>=9].sum()
        base[x>=9] *= cdf_above_9
    return base

plt.figure(figsize=(6, 4), dpi=300)
domain = np.linspace(4, 18, 1000)
plt.fill_between(domain, likelihood(domain, mu=9.0, std=1.2), alpha=0.3, ec='k', color=colors[0, [3]])
plt.text(10, 0.05, r'$\mu=9, \sigma=1.2$''\nprior=flat', horizontalalignment='center')
plt.fill_between(domain, likelihood(domain, mu=7, std=0.5), alpha=0.3, ec='k', color=colors[1, [3]])
plt.text(7, 0.85, r'$\mu=7, \sigma=0.5$', horizontalalignment='center')
plt.fill_between(domain, likelihood(domain, mu=11, std=1, prior='default'), alpha=0.3, ec='k', color=colors[2, [3]])
plt.text(10.5, 0.4, r'$\mu=11, \sigma=1$''\nprior=default', horizontalalignment='center')
plt.fill_between(domain, likelihood(domain, mu=4.2, std=0.9), alpha=0.3, ec='k', color=colors[3, [3]])
plt.text(4, 0.8, r'$\mu=4.2, \sigma=0.9$')#, horizontalalignment='center')
plt.ylim(0, 1)
plt.xlim(3.5, 13)

plt.xlabel('Instability Time [Log10(T)]')
plt.ylabel('Probability')

plt.savefig('example_likelihood.png', dpi=300)
