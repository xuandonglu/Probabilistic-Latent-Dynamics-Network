import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utilities3 import *
from timeit import default_timer
import matplotlib.pyplot as plt
import time
from sklearn.decomposition import PCA
from latent_au import SpectralConv2d_fast, FNO

ntest = 100
batch_size = 1
reduction_model = 'DPN_d25_fno'
path_model = f'model/beam1d_au_N4900_p41_drop2_{reduction_model}'
model = torch.load(path_model)

################################################################
# load data
################################################################
with h5py.File('../reduction/data/data.hdf5', 'r') as f:
    u = f['acc_5000_401_1024'][:]
    u_latent = f[f'latent_{reduction_model}'][:]
    force = f['f_5000_1024'][:]
    tr = f[f'trunk_{reduction_model}'][:]
u = torch.from_numpy(u).float()
u_latent = torch.from_numpy(u_latent).float()
T = u.shape[1]
S = u.shape[2]
Z = u_latent.shape[-1]

pca = PCA(n_components=Z)
fin = torch.from_numpy(pca.fit_transform(force)).float()

x_normalizer = UnitGaussianNormalizer(fin[:1000].unsqueeze(1).unsqueeze(-1))
y_normalizer = UnitGaussianNormalizer(u_latent[:1000])

force = torch.from_numpy(force).float()
test_x = fin[-ntest:]
test_y = u[-ntest:]
test_x = test_x.reshape(ntest, 1, test_x.shape[1], 1).repeat([1, T, 1, 1])

test_x = x_normalizer.encode(test_x)
y_normalizer.cuda()

test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_x, test_y), batch_size=batch_size)
device = torch.device('cuda')
tr = torch.from_numpy(tr).float().to(device)

MSELoss = nn.MSELoss(reduction='mean').cuda()
with torch.no_grad():
    test_loss = 0
    # model.train()
    for x, y in test_loader:
        x, y = x.cuda(), y.cuda()
        out_au_mean = []
        out_au_var = []
        for _ in range(1):
            out = model(x.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
            out_au_mean.append(out[..., 0])
            out_au_var.append(torch.exp(out[..., 1]))
        out_au_mean = torch.stack(out_au_mean, dim=0)
        out_au_var = torch.stack(out_au_var, dim=0)
        out_eu_var = out_au_mean.var(dim=0) * (y_normalizer.eps + y_normalizer.std)
        out_au_var = out_au_var.mean(dim=0) * (y_normalizer.eps + y_normalizer.std)
        out_au_mean = y_normalizer.decode(out_au_mean.mean(dim=0))
        orig_mean = torch.einsum("bti,io->bto", out_au_mean, tr)
        orig_eu_std = torch.sqrt(torch.einsum("bti,io->bto", out_eu_var, tr ** 2))
        orig_au_std = torch.sqrt(torch.einsum("bti,io->bto", out_au_var, tr ** 2))
        loss = MSELoss(orig_mean, y)
        test_loss += loss.item() / ntest

        # for i in range(30):
        #     true = y[0, :, i * 13].detach().cpu()
        #     mean = orig_mean[0, :, i * 13].detach().cpu()
        #     std_eu = orig_eu_std[0, :, i * 13].detach().cpu()
        #     std_au = orig_au_std[0, :, i * 13].detach().cpu()
        #     N = 401
        #     plt.fill_between(torch.linspace(0, N-1, steps=N), mean-2*std_eu, mean+2*std_eu, color='green',
        #                      alpha=0.5, label='EU')
        #     plt.plot(true)
        #     plt.plot(mean)
        #     plt.title(i * 25)
        #     plt.show()
        #
        #     plt.fill_between(torch.linspace(0, N - 1, steps=N), mean - 2 * std_au, mean + 2 * std_au, color='green',
        #                      alpha=0.5, label='AU')
        #     plt.plot(true)
        #     plt.plot(mean)
        #     plt.title(i * 25)
        #     plt.show()
        #
        #     plt.plot(mean-true)
        #     plt.fill_between(torch.linspace(0, N-1, steps=N), -2*std_eu, 2*std_eu, color='green', alpha=0.5, label='EU')
        #     plt.show()
        #
        #     plt.plot(mean - true)
        #     plt.fill_between(torch.linspace(0, N - 1, steps=N), -2 * std_au, 2 * std_au, color='green', alpha=0.5,
        #                      label='AU')
        #     plt.show()

print(f"[eval] test_loss={test_loss:.4e}")
