import sys
import seaborn as sns
sns.set_style('darkgrid')
from matplotlib import pyplot as plt
import spock_reg_model
spock_reg_model.HACK_MODEL = True
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateLogger, ModelCheckpoint

import torch
import numpy as np
from scipy.stats import truncnorm

import time
from tqdm.notebook import tqdm

rand = lambda lo, hi: np.random.rand()*(hi-lo) + lo
irand = lambda lo, hi: int(np.random.rand()*(hi-lo) + lo)
log_rand = lambda lo, hi: 10**rand(np.log10(lo), np.log10(hi))
ilog_rand = lambda lo, hi: int(10**rand(np.log10(lo), np.log10(hi)))


from parse_swag_args import parse
args, checkpoint_filename = parse()
seed = args.seed

TOTAL_STEPS = args.swa_steps
TRAIN_LEN = 78660
batch_size = 2000 #ilog_rand(32, 3200)
steps_per_epoch = int(1+TRAIN_LEN/batch_size)
epochs = int(1+TOTAL_STEPS/steps_per_epoch)

swa_args = {
    'swa_lr' : 1e-4, #1e-4 is largest before NaN
    'swa_start' : int(0.5*TOTAL_STEPS), #step
    'swa_recording_lr_factor': 0.5,
    'c': 5,
    'K': 30,
    'steps': TOTAL_STEPS,
}

output_filename = checkpoint_filename + '_output'

total_attempts_to_record = (TOTAL_STEPS - swa_args['swa_start'])/steps_per_epoch/swa_args['c']
total_attempts_to_record


try:
    swag_model = (
        spock_reg_model.SWAGModel.load_from_checkpoint(
        checkpoint_filename + '/version=0_v0.ckpt')
        .init_params(swa_args)
    )
except FileNotFoundError:
    swag_model = (
        spock_reg_model.SWAGModel.load_from_checkpoint(
        checkpoint_filename + '/version=0.ckpt')
        .init_params(swa_args)
    )

max_l2_norm = 0.1*sum(p.numel() for p in swag_model.parameters() if p.requires_grad)

swag_model.hparams.steps = TOTAL_STEPS
swag_model.hparams.epochs = epochs

lr_logger = LearningRateLogger()
name = 'full_swag_post_' + checkpoint_filename
logger = TensorBoardLogger("tb_logs", name=name)
checkpointer = ModelCheckpoint(
    filepath=checkpoint_filename + '.ckpt',
    monitor='swa_loss_no_reg'
)

trainer = Trainer(
    gpus=1, num_nodes=1, max_epochs=epochs,
    logger=logger, callbacks=[lr_logger],
    checkpoint_callback=checkpointer, benchmark=True,
    terminate_on_nan=True, gradient_clip_val=max_l2_norm
)

try:
    trainer.fit(swag_model)
except ValueError:
    print("Model", checkpoint_filename, 'exited early!', flush=True)
    exit(1)

# Save model:

logger.log_hyperparams(
    params=swag_model.hparams,
    metrics={'swa_loss_no_reg': checkpointer.best_model_score.item()})
logger.save()
logger.finalize('success')

spock_reg_model.save_swag(swag_model, output_filename + '.pkl')
import pickle as pkl
pkl.dump(swag_model.ssX, open(output_filename + '_ssX.pkl', 'wb'))

summary_writer = logger.experiment

import matplotlib.pyplot as plt
plt.switch_backend('agg')

from tqdm.notebook import tqdm

truths = []
preds = []
raw_preds = []
val_dataloader = swag_model._val_dataloader

nc = 0
losses = 0.0
do_sample = True
swag_model.eval()
swag_model.w_avg = swag_model.w_avg.cuda()
swag_model.w2_avg = swag_model.w2_avg.cuda()
swag_model.pre_D = swag_model.pre_D.cuda()
swag_model.cuda()
for X_sample, y_sample in tqdm(val_dataloader):
    X_sample = X_sample.cuda()
    y_sample = y_sample.cuda()
    nc += len(y_sample)
    truths.append(y_sample.cpu().detach().numpy())

    raw_preds.append(
        np.array([swag_model.forward_swag(X_sample, scale=0.5).cpu().detach().numpy() for _ in range(1000)])
    )

truths = np.concatenate(truths)

_preds = np.concatenate(raw_preds, axis=1)

_preds.shape

std = _preds[..., 1]
mean = _preds[..., 0]

a = (4 - mean)/std
b = np.inf
sample_preds = truncnorm.rvs(a.reshape(-1), b).reshape(*a.shape)*std + mean

stable_past_9 = sample_preds > 9

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

preds = np.average(sample_preds, 0)
stds = np.std(sample_preds, 0)

# Check that confidence intervals are satisifed. Calculate mean and std of samples. Take abs(truths - mean)/std = sigma. The CDF of this distrubtion should match that of a Gaussian. Otherwise, rescale "scale". 

tmp_mask = (truths > 6) & (truths < 7) #Take this portion since its far away from truncated parts
sigma = (truths[tmp_mask] - np.tile(preds, (2, 1)).T[tmp_mask])/np.tile(stds, (2, 1)).T[tmp_mask]

bins = 30

plt.rc('font', family='serif')
sns.set_style('white')
fig, ax = plt.subplots(1, 1, dpi=300)
plt.hist(np.abs(sigma), bins=bins, range=[0, 2.5], normed=True,
         alpha=1, label='Model error distribution')
np.random.seed(0)
plt.hist(np.abs(np.random.randn(len(sigma))), bins=bins, range=[0, 2.5], normed=True,
         alpha=0.5, label='Gaussian distribution')
plt.ylabel('Density', fontsize=14)
plt.xlabel('$|\mu_θ - y|/\sigma_θ$', fontsize=14)
plt.legend()
summary_writer.add_figure(
    'error_distribution',
    fig)
plt.show()

sns.set_style('darkgrid')

# Looks great! We didn't even need to tune it. Just use the same scale as the paper (0.5). Perhaps, however, with epistemic uncertainty, we will need to tune. 

from matplotlib.colors import LogNorm
plt.rc('font', family='serif')

def density_scatter(x, y, xlabel='', ylabel='', clabel='Sample Density', log=False,
    width_mult=1, bins=30, p_cut=None, update_rc=True, ax=None, fig=None, cmap='viridis', **kwargs):
    if update_rc:
        plt.rc('font', family='serif')
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

py = preds
py = np.clip(py, 4, 9)
px = np.average(truths, 1)
    
name = 'bnn'
for log in [False, True]:
    fig, ax = plt.subplots(1, 1, figsize=(4, 3.3), dpi=300)
    density_scatter(
        y=py,
        x=px,
        log=log,
        ylabel='Predicted',
        xlabel='Truth',
        clabel='Sample Density' if not log else 'Log Sample Density',
        bins=30,
        p_cut=None,
        fig=fig,
        ax=ax,
        cmap='jet',
        vmax=2.5e-1,#2.1e-1,
        range=[[4, 9.1], [4, 9.1]]
    )
    ax.plot([4, 9], [4, 9], color='k')
    ax.axis('equal')
    summary_writer.add_figure(
        'prediction_density_no_log' if not log else 'prediction_density_log',
        fig)
    plt.show()

fig, ax = plt.subplots(1, 1, dpi=300)
ax.hist((py-px)[px<9], normed=True, bins=30)
ax.set_xlabel('Residual')
ax.set_ylabel('Probability')
ax.set_title('RMS residual under 9: %.2f'% (np.sqrt(np.average(np.abs(py-px)[px<9])),))

summary_writer.add_figure(
    'residual_hist',
    fig)
plt.show()

labels = ['time', 'e+_near', 'e-_near', 'max_strength_mmr_near', 'e+_far', 'e-_far', 'max_strength_mmr_far', 'megno', 'a1', 'e1', 'i1', 'cos_Omega1', 'sin_Omega1', 'cos_pomega1', 'sin_pomega1', 'cos_theta1', 'sin_theta1', 'a2', 'e2', 'i2', 'cos_Omega2', 'sin_Omega2', 'cos_pomega2', 'sin_pomega2', 'cos_theta2', 'sin_theta2', 'a3', 'e3', 'i3', 'cos_Omega3', 'sin_Omega3', 'cos_pomega3', 'sin_pomega3', 'cos_theta3', 'sin_theta3', 'm1', 'm2', 'm3', 'nan_mmr_near', 'nan_mmr_far', 'nan_megno']

if 'include_derivatives' in swag_model.hparams and swag_model.hparams['include_derivatives']:
    labels = labels + ['d_' + label for label in labels]
    assert swag_model.hparams['include_angles']

kl = lambda logvar: (1/2)*(torch.exp(logvar) - logvar - 1)
mask_vals = kl(swag_model.input_noise_logvar).cpu().detach().numpy()
sort_idx = np.argsort(mask_vals)
fig = plt.figure(figsize=(8, 8), dpi=300)
plt.barh(y=np.arange(len(labels)), width=mask_vals[sort_idx], tick_label=[labels[i] for i in sort_idx])
plt.xlabel('Importance (KL divergence)')

summary_writer.add_figure(
    'feature_importance',
    fig)
plt.show()

exit(0)

# kls = swag_model._summary_kl.mean(0).cpu().detach().numpy()
# fig = plt.figure(figsize=(8, 8), dpi=300)
# plt.barh(y=np.arange(swag_model.hparams['latent']*2),
         # width=kls,
         # height=1
   # )

# plt.xlim(0, kls.max()*1.2)

# summary_writer.add_figure(
    # 'latent_importance',
    # fig)
# plt.show()

# # checkpointer.best.item()

# from sklearn.metrics import roc_curve, roc_auc_score

# fpr, tpr, _ = roc_curve(y_true=np.max(truths>8.9, 1), y_score=np.average(sample_preds >= 9, axis=0))
# fig = plt.figure(figsize=(8, 8), dpi=300)
# plt.plot(fpr, tpr)
# plt.xlabel('fpr')
# plt.ylabel('tpr')
# plt.title('AUC ROC = %.2f'%(roc_auc_score(y_true=np.max(truths>8.9, 1), y_score=np.average(sample_preds>= 9, axis=0)),))
# summary_writer.add_figure(
    # 'roc_curve',
    # fig)
# plt.show()

# np.average(sample_preds >= 9, 0)

