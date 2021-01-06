"""This file trains a model to minima, then saves it for run_swag.py"""
import seaborn as sns
sns.set_style('darkgrid')
from matplotlib import pyplot as plt
import spock_reg_model
spock_reg_model.HACK_MODEL = True
from pytorch_lightning import Trainer
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import numpy as np
from scipy.stats import truncnorm
import sys
from parse_swag_args import parse

rand = lambda lo, hi: np.random.rand()*(hi-lo) + lo
irand = lambda lo, hi: int(np.random.rand()*(hi-lo) + lo)
log_rand = lambda lo, hi: 10**rand(np.log10(lo), np.log10(hi))
ilog_rand = lambda lo, hi: int(10**rand(np.log10(lo), np.log10(hi)))

args, checkpoint_filename = parse()
seed = args.seed

# Fixed hyperparams:
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
    # Much of these settings turn off other parameters tried:
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

name = 'full_swag_pre_' + checkpoint_filename
logger = TensorBoardLogger("tb_logs", name=name)
checkpointer = ModelCheckpoint(filepath=checkpoint_filename + '/{version}')
model = spock_reg_model.VarModel(args)
model.make_dataloaders()

labels = ['time', 'e+_near', 'e-_near', 'max_strength_mmr_near', 'e+_far', 'e-_far', 'max_strength_mmr_far', 'megno', 'a1', 'e1', 'i1', 'cos_Omega1', 'sin_Omega1', 'cos_pomega1', 'sin_pomega1', 'cos_theta1', 'sin_theta1', 'a2', 'e2', 'i2', 'cos_Omega2', 'sin_Omega2', 'cos_pomega2', 'sin_pomega2', 'cos_theta2', 'sin_theta2', 'a3', 'e3', 'i3', 'cos_Omega3', 'sin_Omega3', 'cos_pomega3', 'sin_pomega3', 'cos_theta3', 'sin_theta3', 'm1', 'm2', 'm3', 'nan_mmr_near', 'nan_mmr_far', 'nan_megno']
    
max_l2_norm = args['gradient_clip']*sum(p.numel() for p in model.parameters() if p.requires_grad)
trainer = Trainer(
    gpus=1, num_nodes=1, max_epochs=args['epochs'],
    logger=logger,
    checkpoint_callback=checkpointer, benchmark=True,
    terminate_on_nan=True, gradient_clip_val=max_l2_norm
)

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
