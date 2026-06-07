import torch
from torch import nn
import sys
import numpy as np
sys.path.append('../')
from nn import DPN
import h5py
import matplotlib.pyplot as plt

MSELoss = nn.MSELoss(reduction='mean')

def loss_function(reconstruction, data):
    mse = MSELoss(reconstruction, data)
    return mse

model = torch.load('DPN_d25_fno.pth')
model.eval()

b = 100
with h5py.File('../data/data.hdf5', 'r') as f:
    u = f['acc_5000_201_81_61'][:][-b:, :, -512:]
T = u.shape[1]
X = u.shape[2]
Y = u.shape[3]
u = np.reshape(u, [b * T, X, Y])
u = torch.from_numpy(u.astype(np.float32))
u_rec = torch.zeros_like(u)
for i in range(b):
    u_rec[i*T:(i+1)*T] = model(u[i*T:(i+1)*T, ...].cuda()).cpu()
u_rec = u_rec.reshape([b, T, X, Y])
u = np.reshape(u, [b, T, X, Y])
loss = loss_function(u_rec, u)

u_np = u.cpu().detach().numpy()
u_rec_np = u_rec.cpu().detach().numpy()
for i in range(b):
    # for j in range(40):
    #     clean_sample = u_np[i, :, j*2, 0]
    #     rec_sample = u_rec_np[i, :, j*2, 0]
    #     plt.plot(clean_sample, label='True')
    #     plt.plot(rec_sample, label='Reconstructed')
    #     plt.legend()
    #     plt.title(f'utt_{j * 2}')
    #     plt.show()
    for j in range(100):
        clean_sample = u_np[i, j * 2]
        rec_sample = u_rec_np[i, j * 2]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        cs1 = axes[0].contourf(clean_sample)
        axes[0].set_title("Ground Truth")
        cs2 = axes[1].contourf(rec_sample)
        axes[1].set_title("Prediction")
        fig.colorbar(cs1, ax=axes[:], orientation='vertical', shrink=0.8, label='Value')
        fig.suptitle(f"Slice {j * 2}")
        plt.tight_layout()
        plt.show()
