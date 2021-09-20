import sys
import os
import time
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import loss_functions, modules, training, utils
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

from functools import partial
import configargparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

from photon_library import PhotonLibrary


# Example syntax:
# run siren_test.py --output_dir test --experiment_name test --num_epochs 1

# Configure Arguments
p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--output_dir', type=str, default='./results', help='root for logging outputs')
p.add_argument('--experiment_name', type=str, required=True,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

# General training options
p.add_argument('--batch_size', type=int, default=1)
p.add_argument('--lr', type=float, default=5e-5, help='learning rate. default=5e-5') #5e-6 for FH
p.add_argument('--num_epochs', type=int, default=10000,
               help='Number of epochs to train for.')
p.add_argument('--kl_weight', type=float, default=1e-1,
               help='Weight for l2 loss term on code vectors z (lambda_latent in paper).')

p.add_argument('--epochs_til_ckpt', type=int, default=1000,
               help='Time interval in seconds until checkpoint is saved.')
p.add_argument('--steps_til_summary', type=int, default=1000,
               help='Time interval in seconds until tensorboard summary is saved.')

p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')

opt = p.parse_args()

device = list(range(torch.cuda.device_count()))
torch.cuda.manual_seed_all(42)
start = time.time()

# Load plib dataset
print('Load data ...')
plib = PhotonLibrary()
data = plib.numpy() 

output_dir = os.path.join(opt.output_dir, opt.experiment_name)
log_pred = []
log_gt = []
mae = []
std = []

data_shape = data.shape[0:-1]
data_shape = tuple(data_shape)

data = np.expand_dims(np.reshape(data, (-1, data.shape[-1])), axis=0)
data = np.expand_dims(np.sum(data, -1), axis=-1)
data = -np.log(data+1e-7)
data = (data - np.amin(data)) / (np.amax(data) - np.amin(data))

data = {'coords': data}
print('organizing coords')
x = np.linspace(0, data_shape[0] - 1, data_shape[0])
y = np.linspace(0, data_shape[1] - 1, data_shape[1])
z = np.linspace(0, data_shape[2] - 1, data_shape[2])

print('organizing coords')
coordx, coordy, coordz = np.meshgrid(x, y, z)
coord = np.reshape(np.stack([coordx, coordy, coordz], -1), (-1, 3))
coord_real = np.expand_dims(plib._min + (plib._max - plib._min) / plib.shape * (coord + 0.5), axis=0)
coord_real = torch.from_numpy(coord_real.astype(np.float32)).cuda()

coord_real.requires_grad = False
coord_real.cuda()
coord_real = {'coords': coord_real}

print('Assigning Model...')
model = modules.Siren(in_features=3, out_features=1, hidden_features=256, hidden_layers=3, outermost_linear=True, omega=5)
model = model.float()
model = nn.DataParallel(model, device_ids=device)
model.cuda()

model_output = {'model_out': data['coords'], 'model_in': coord_real['coords']}

train_data = utils.DataWrapper(model_output, data_shape, data)

print('at the dataloader')

dataloader = DataLoader(train_data, shuffle=True, batch_size=opt.batch_size, pin_memory=False, num_workers=0)

loss = partial(loss_functions.image_mse)

print('Training Step...')
lp, lg, m, st = training.train(model=model, train_dataloader=dataloader, epochs=opt.num_epochs, lr=opt.lr,
                                       steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,
                                       model_dir=output_dir, data_shape=data_shape, loss_fn=loss)

# log_pred.append(lp)
# log_gt.append(lg)
# mae.append(m)
# std.append(st)

# plot mae and std for each slice
# x = np.arange(0, len(m))
# fig, ax = plt.subplots()
# fig3, ax3 = plt.subplots()

# ax.plot(x, m, label='Mean Absolute Error')
# ax.set_xlabel('X-slice')
# ax.set_title('MAE for each X-Slice')
# fig.savefig(output_dir+'/mae.png', dpi=300, bbox_inches='tight', facecolor='w')
# plt.close()

# ax3.plot(x, st, label='Standard Dev')
# ax3.set_xlabel('X-slice')
# ax3.set_title('STD for each X-Slice')
# fig3.savefig(output_dir+'/std.png', dpi=300, bbox_inches='tight', facecolor='w')
# plt.close()


# fig2, ax2 = plt.subplots()
# plot pred vs gt for each slice
# for i in range(len(log_pred)):
#     ax2.scatter(log_gt[i], log_pred[i])
# ax2.set_xlabel('Truth')
# ax2.set_ylabel('Pred')
# ax2.set_xlim(-1,17)
# ax2.set_ylim(-1,17)
# ax2.grid(True)
# fig2.savefig(output_dir+'/tot_pred_vs_truth.png', dpi=300, bbox_inches='tight', facecolor='w')
# plt.close()


end = time.time()
print('Delta Time: {}'.format(end-start))
print('Complete. :)')
