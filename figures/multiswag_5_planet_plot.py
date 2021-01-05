import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import pandas as pd
import seaborn as sns

basedir_bayes = '/mnt/home/mcranmer/local_orbital_physics/miles'
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


def make_plot(cleaned, version, t20=True):
# +
# %matplotlib inline
    plt.style.use('science')
    fig, axarr = plt.subplots(1, 1, figsize=(16/2,4/2), dpi=400, sharex=True)
    plt.subplots_adjust(hspace=0, wspace=0)
    ax = plt.gca()
    kwargs = dict(alpha=0.5,
                  markersize=5/4)
    tmp = cleaned#.query('true > 4')
    tmp.loc[:, 'xgb'] = tmp.loc[:, 'true'] * np.array(tmp['true'] <= 4.0) + tmp.loc[:, 'xgb'] * np.array(tmp['true'] > 4.0)
    tmp2 = tmp.query('true > 4 & delta > 5')
    ax.fill_between(
        tmp2['delta'], tmp2['l'], tmp2['u'], color=colors[2, [3]], alpha=0.2)
    # ax.fill_between(
        # tmp2['delta'], tmp2['ll'], tmp2['uu'], color=colors[2, [3]], alpha=0.1)

    tmp.plot(
        'delta', 'true', ax=ax,
        label='True', 
        c='k'
    )
    if t20:
        tmp.plot(
            'delta', 'xgb', ax=ax,
            label='Modified T20', 
            c=colors[1, 3]
        )
    # tmp.plot(
        # 'delta', 'petit', ax=ax,
        # label='Petit+20, no tuning', 
# #     style='o',
# #     ms=5./4*0.3,
        # c=colors[3, 3]
    # )
    tmp.plot(
        'delta', 'petitf', ax=ax,
        label='Petit+20', 
#     style='o',
#     ms=5./4*0.3, 
        c=colors[0, 3]
    )
    tmp.plot(
        'delta', 'median', ax=ax,
        label='Ours', 
#     style='o',ms=5./4*0.3, color=colors[2, 3]
        c=colors[2, 3]
    )
# xlim = ax.get_xlim()
    ax.plot([0, 14], [9, 9], '--k')
    ax.plot([0, 14], [4, 4], '--k')
    ax.annotate('Training range', (12, 4.5))
# ax.set_xlim(*xlim)
# # plt.plot()
    ax.set_xlim(1, 14)
    ax.set_ylim(0, 12)
# ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: r'$10^{%d}$'%(x,)))
    ax.set_xlabel(r'$\Delta$')
    ax.set_ylabel(r'Instability Time')
    plt.legend(loc='upper left',
               frameon=True, fontsize=8)

    fig.savefig(basedir_bayes + '/' + f'comparison_v{version}' + '5planet.png')


# -



    plt.style.use('default')
    sns.set_style('white')
    plt.rc('font', family='serif')

    for key in 'median petitf'.split(' '):
        px = tmp['true']
        py = tmp[key]


        from scipy.stats import gaussian_kde

        mask = (px > 4)
        px = np.clip(px, 4, 9)
        ppx = px[mask]
        py = np.clip(py, 4, 9)
        ppy = py[mask]
        # bw = 0.2

        fig = plt.figure(figsize=(4, 4), 
                         dpi=300,
                         constrained_layout=True)

        g = sns.jointplot(x=ppx, y=ppy,
                        alpha=1.0,# ax=ax,
                        color=colors[2, 3],
                        s=5,
                        xlim=(3, 10),
                        ylim=(3, 10),
                        marginal_kws=dict(bins=15),
                       )
        ax = g.ax_joint

        ax.plot([4, 9], [4, 9], color='k')

        ## Errorbars:
        # if key == 'median':
            # upper = tmp['u'] 
            # lower = tmp['l']
            # upper = np.clip(upper[mask], 4, 9)
            # lower = np.clip(lower[mask], 4, 9)
            # upper = upper - ppy
            # lower = ppy - lower
            # plt.scatter

            # ax.errorbar(
                    # ppx,
                    # ppy,
                    # yerr=[lower, upper],
                    # fmt='o',
                    # ecolor=list(colors[2, 3]) + [0.2],
                    # ms=5,
                    # color=colors[2, 3]
                # )
        # plt.colorbar(im)

        #Try doing something like this: https://seaborn.pydata.org/examples/kde_ridgeplot.html
        #Stack all contours on the same axis. Have horizontal lines to help compare residuals.

        title = 'Ours' if key == 'median' else 'Petit+20'
        ax.set_xlabel('Truth') 
        ax.set_ylabel('Predicted')
        plt.suptitle(title, y=1.0)
        plt.tight_layout()
        plt.savefig(basedir_bayes + f'/comparison_v{version}_5planet_{key}.png', dpi=300)


