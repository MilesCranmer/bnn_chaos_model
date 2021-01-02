import seaborn as sns
sns.set_style('darkgrid')
from matplotlib import pyplot as plt

import spock_reg_model
spock_reg_model.HACK_MODEL = True
from pytorch_lightning import Trainer

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateLogger, ModelCheckpoint
import torch
import numpy as np
from scipy.stats import truncnorm
import sys

rand = lambda lo, hi: np.random.rand()*(hi-lo) + lo
irand = lambda lo, hi: int(np.random.rand()*(hi-lo) + lo)
log_rand = lambda lo, hi: 10**rand(np.log10(lo), np.log10(hi))
ilog_rand = lambda lo, hi: int(10**rand(np.log10(lo), np.log10(hi)))

from parse_swag_args import parse
args, checkpoint_filename = parse()
seed = args.seed

lr = 5e-4
TOTAL_STEPS = args.total_steps
TRAIN_LEN = 78660
batch_size = 2000 #ilog_rand(32, 3200)
steps_per_epoch = int(1+TRAIN_LEN/batch_size)
epochs = int(1+TOTAL_STEPS/steps_per_epoch)

epochs, epochs//10


args = {
    'seed': seed,
    'batch_size': batch_size,
    'hidden': args.hidden,#ilog_rand(50, 1000),
    'in': 1,
    'latent': args.latent, #2,#ilog_rand(4, 500),
    'lr': lr,
    'swa_lr': lr/2,
    'out': 1,
    'samp': 5,
    'swa_start': epochs//2,
    'weight_decay': 1e-14,
    'to_samp': 1,
    'epochs': epochs,
    'scheduler': True,
    'scheduler_choice': 'swa',
    'steps': TOTAL_STEPS,
    'beta_in': 1e-5,
    'beta_out': args.beta,#0.003,
    'act': 'softplus',
    'noisy_val': False,
    'gradient_clip': 0.1,
    'fix_megno': args.megno, #avg,std of megno
    'fix_megno2': (not args.megno), #Throw out megno completely
    'include_angles': args.angles,
    'include_mmr': (not args.no_mmr),
    'include_nan': (not args.no_nan),
    'include_eplusminus': (not args.no_eplusminus),
    'power_transform': args.power_transform,
    'lower_std': args.lower_std,
    'train_all': args.train_all,
}

lr_logger = LearningRateLogger()
name = 'full_swag_pre_' + checkpoint_filename
logger = TensorBoardLogger("tb_logs", name=name)

checkpointer = ModelCheckpoint(filepath=checkpoint_filename + '/{version}')

model = spock_reg_model.VarModel(args)

model.make_dataloaders()

labels = ['time', 'e+_near', 'e-_near', 'max_strength_mmr_near', 'e+_far', 'e-_far', 'max_strength_mmr_far', 'megno', 'a1', 'e1', 'i1', 'cos_Omega1', 'sin_Omega1', 'cos_pomega1', 'sin_pomega1', 'cos_theta1', 'sin_theta1', 'a2', 'e2', 'i2', 'cos_Omega2', 'sin_Omega2', 'cos_pomega2', 'sin_pomega2', 'cos_theta2', 'sin_theta2', 'a3', 'e3', 'i3', 'cos_Omega3', 'sin_Omega3', 'cos_pomega3', 'sin_pomega3', 'cos_theta3', 'sin_theta3', 'm1', 'm2', 'm3', 'nan_mmr_near', 'nan_mmr_far', 'nan_megno']
    
max_l2_norm = args['gradient_clip']*sum(p.numel() for p in model.parameters() if p.requires_grad)

trainer = Trainer(
    gpus=1, num_nodes=1, max_epochs=args['epochs'],
    logger=logger, callbacks=[lr_logger],
    checkpoint_callback=checkpointer, benchmark=True,
    terminate_on_nan=True, gradient_clip_val=max_l2_norm
)

# !rm -r tb_logs/custom_plateau/version_n/

try:
    trainer.fit(model)
except ValueError:
    model.load_state_dict(torch.load(checkpointer.best_model_path)['state_dict'])

logger.log_hyperparams(params=model.hparams, metrics={'val_loss': checkpointer.best_model_score.item()})
logger.save()
logger.finalize('success')

logger.save()

model.load_state_dict(torch.load(checkpointer.best_model_path)['state_dict'])
model.make_dataloaders()

summary_writer = logger.experiment

import matplotlib.pyplot as plt
plt.switch_backend('agg')

from tqdm.notebook import tqdm

truths = []
preds = []
raw_preds = []
val_dataloader = model._val_dataloader

nc = 0
losses = 0.0
do_sample = True
model.eval()
model.cuda()
for X_sample, y_sample in tqdm(val_dataloader):
    X_sample = X_sample.cuda()
    y_sample = y_sample.cuda()
    nc += len(y_sample)
    truths.append(y_sample.cpu().detach().numpy())
    model.random_sample = False

    raw_preds.append(
        np.array([model(X_sample).cpu().detach().numpy() for _ in range(100)])
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

kl = lambda logvar: (1/2)*(torch.exp(logvar) - logvar - 1)
mask_vals = kl(model.input_noise_logvar).cpu().detach().numpy()
sort_idx = np.argsort(mask_vals)
fig = plt.figure(figsize=(8, 8), dpi=300)
plt.barh(y=np.arange(len(labels)), width=mask_vals[sort_idx], tick_label=[labels[i] for i in sort_idx])
plt.xlabel('Importance (KL divergence)')

summary_writer.add_figure(
    'feature_importance',
    fig)
plt.show()

exit(0)

# kls = model._summary_kl.mean(0).cpu().detach().numpy()
# fig = plt.figure(figsize=(8, 8), dpi=300)
# plt.barh(y=np.arange(args['latent']*2),
         # width=kls,
         # height=1
   # )

# plt.xlim(0, kls.max()*1.2)

# summary_writer.add_figure(
    # 'latent_importance',
    # fig)
# plt.show()

# checkpointer.best.item()

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

