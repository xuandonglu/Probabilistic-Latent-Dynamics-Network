import torch
from torch import nn
import sys
import numpy as np
sys.path.append('../')
from nn import DPN
import h5py
import scipy.io as io
import matplotlib.pyplot as plt

MSELoss = nn.MSELoss(reduction='mean')

def loss_function(reconstruction, data):
    mse = MSELoss(reconstruction, data)
    return mse

model_name = 'DPN_d20_fno'
model = torch.load(f'{model_name}.pth')
model.eval()

b = 100
with h5py.File('../data/data.hdf5', 'r') as f:
    u = f['acc_5000_401_1024'][:][-b:]
T = u.shape[1]
S = u.shape[2]
u = np.reshape(u, [b * T, S])
u = torch.from_numpy(u.astype(np.float32))
u_rec = torch.zeros_like(u)
for i in range(b):
    u_rec[i*T:(i+1)*T] = model(u[i*T:(i+1)*T, ...].cuda()).cpu()
u_rec = u_rec.reshape([b, T, S])
u = np.reshape(u, [b, T, S])

loss = loss_function(u_rec, u).cpu().detach().numpy()
print(f"[eval] test_loss={loss:.2e}")

u_np = u.cpu().detach().numpy()
u_rec_np = u_rec.cpu().detach().numpy()
io.savemat(f'{model_name}.mat', {'rec': u_rec_np[:3], 'test': u_np[:3]})

# for i in range(b):
#     for j in range(200):
#         clean_sample = u_np[i, j*1, :]
#         rec_sample = u_rec_np[i, j*1, :]
#         plt.plot(clean_sample, label='True')
#         plt.plot(rec_sample, label='Reconstructed')
#         plt.legend()
#         plt.title(f'utt_{j * 1}')
#         plt.show()
#     for j in range(50):
#         clean_sample = u_np[i, :, j*20]
#         rec_sample = u_rec_np[i, :, j*20]
#         plt.plot(clean_sample, label='True')
#         plt.plot(rec_sample, label='Reconstructed')
#         plt.legend()
#         plt.show()
