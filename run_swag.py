"""This file takes a model at its minima, and models the Gaussian at the posterior mode"""
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
