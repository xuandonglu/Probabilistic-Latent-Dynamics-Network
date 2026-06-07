import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utilities3 import *
from timeit import default_timer
import matplotlib.pyplot as plt
import time
import scipy.io as io
from sklearn.decomposition import PCA
from latent import SpectralConv2d_fast, FNO

ntest = 100
batch_size = 2
reduction_model = 'DPN_d25_fno'
path_model = f'model/pv2d_N4900_p41_drop2_{reduction_model}'
model = torch.load(path_model)

################################################################
# load data
################################################################
with h5py.File('../reduction/data/data.hdf5', 'r') as f:
    u = f['acc_5000_201_81_61'][:]
    force = f['f_5000_201'][:]
    u_latent = f[f'latent_{reduction_model}'][:]
    tr = f[f'trunk_{reduction_model}'][:]
u = torch.from_numpy(u).float()
u_latent = torch.from_numpy(u_latent).float()
T = u.shape[1]
Z = u_latent.shape[-1]

pca = PCA(n_components=Z)
fin = torch.from_numpy(pca.fit_transform(force)).float()
x_normalizer = UnitGaussianNormalizer(fin[:500].unsqueeze(1).unsqueeze(-1))
y_normalizer = UnitGaussianNormalizer(u_latent[:500])

test_x = fin[-ntest:]
test_y = u[-ntest:]
test_x = test_x.reshape(ntest, 1, Z, 1).repeat([1, T, 1, 1])

# test_x = x_normalizer.encode(test_x)
y_normalizer.cuda()

test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_x, test_y), batch_size=batch_size)
device = torch.device('cuda')
tr = torch.from_numpy(tr).float()
MSELoss = nn.MSELoss(reduction='mean').cuda()

def evaluate_uncertainty(mu, var_total, y_true, z=1.96):
    std_total = torch.sqrt(var_total)

    mse = torch.mean((mu - y_true) ** 2)

    nll = 0.5 * torch.log(2 * torch.pi * var_total) + 0.5 * ((y_true - mu)**2) / var_total
    nll_mean = torch.mean(nll)

    sharpness = torch.mean(2 * z * std_total)

    lower = mu - z * std_total
    upper = mu + z * std_total
    covered = ((y_true >= lower) & (y_true <= upper)).float()
    coverage = torch.mean(covered)
    ece = torch.abs(coverage - 0.95)

    return {
        'MSE': mse.item(),
        'NLL': nll_mean.item(),
        'Sharpness': sharpness.item(),
        'ECE': ece.item()
    }

with torch.no_grad():
    test_loss = 0
    results  = {
    'MSE': 0.0,
    'NLL': 0.0,
    'Sharpness': 0.0,
    'ECE': 0.0}
    model.train()
    n = -1
    for x, y in test_loader:
        x = x.cuda()
        out_mean = []
        t1 = default_timer()
        for _ in range(1000):
            out = model(x.permute(0, 2, 1, 3)).permute(0, 2, 1)
            out_mean.append(out.cpu())
        out_mean = torch.stack(out_mean, dim=0)
        out_eu_var = out_mean.var(dim=0)
        out_mean = out_mean.mean(dim=0)
        orig_mean = torch.einsum("bti,ixy->btxy", out_mean, tr)
        orig_eu_var = torch.einsum("bti,ixy->btxy", out_eu_var, tr ** 2)
        orig_var = orig_eu_var
        t2 = default_timer()
        loss = MSELoss(orig_mean, y)
        test_loss += loss.item() / ntest * batch_size
        batch_result = evaluate_uncertainty(orig_mean, orig_var, y, z=1.96)
        for k in results:
            results[k] += batch_result[k] / ntest * batch_size
        n += 1
        print(f"[eval] sample={n} | total_time={t2 - t1:.3f}s")
        io.savemat(f'{reduction_model}.mat', {'rec': orig_mean[:2].cpu().detach().numpy(), 'test': y[:2].cpu().detach().numpy()})

        # for i in range(30):
        #     true = y[0, :, i * 13].detach().cpu()
        #     mean = orig_mean[0, :, i * 13].detach().cpu()
        #     std_eu = torch.sqrt(orig_eu_var[0, :, i * 13]).detach().cpu()
        #     std_au = torch.sqrt(orig_au_var[0, :, i * 13]).detach().cpu()
        #     N = 401
        #     plt.fill_between(torch.linspace(0, N-1, steps=N), mean-2*std_eu-2*std_au, mean+2*std_eu+2*std_au, color='green',
        #                      alpha=0.5, label='EU')
        #     plt.plot(true)
        #     plt.plot(mean)
        #     plt.title(i * 13)
        #     plt.show()

print(f"[eval] test_loss={test_loss:.4e}")
