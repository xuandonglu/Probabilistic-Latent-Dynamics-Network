import torch
import numpy as np
import sys
sys.path.append('../')
from nn import DPN
import h5py
import matplotlib.pyplot as plt
import time
# time.sleep(5600)

model_name = "AE_d40.pth"
name_without_ext = model_name.rsplit(".pth", 1)[0]
model = torch.load(model_name)
model.eval()

b = 5000
file_path = "../data/data.hdf5"
with h5py.File(file_path, 'r') as f:
    u = f['acc_5000_201_81_61'][:]
T = u.shape[1]
X = u.shape[2]
Y = u.shape[3]
u = np.reshape(u, [b * T, X, Y])
u_test = torch.from_numpy(u.astype(np.float32))

u_latent = np.zeros((b, T, model.latent_dim))
u_rec = np.zeros((b, T, X, Y))
for i in range(u_rec.shape[0]):
    single_sample = u_test[i*T:(i+1)*T, ...]
    z = model.encode(single_sample.cuda())
    u_latent[i, ...] = z.view(T, model.latent_dim).detach().cpu().numpy().astype(np.float32)

dataset = f'latent_{name_without_ext}'
with h5py.File(file_path, 'a') as hdf:
    if dataset in hdf:
        hdf[dataset][...] = u_latent.astype('float32')
    else:
        hdf.create_dataset(dataset, data=u_latent, dtype='float32')

# trunk_out = model.trunk(u_test.cuda()).detach().cpu().permute(2, 0, 1).numpy().astype(np.float32)
# dataset_trunk = f'trunk_{name_without_ext}'
# with h5py.File(file_path, 'a') as hdf:
#     if dataset_trunk in hdf:
#         hdf[dataset_trunk][...] = trunk_out.astype('float32')
#     else:
#         hdf.create_dataset(dataset_trunk, data=trunk_out, dtype='float32')