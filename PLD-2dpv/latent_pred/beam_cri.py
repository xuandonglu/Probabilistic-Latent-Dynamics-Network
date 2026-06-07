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
from beam import SpectralConv3d, FNO

ntest = 100
batch_size = 2
path_model = f'model/pv2d_N4900_p41_drop2'
model = torch.load(path_model)

################################################################
# load data
################################################################
with h5py.File('../reduction/data/data.hdf5', 'r') as f:
    u = f['acc_5000_201_81_61'][:]
    force = f['f_5000_201'][:]
u = torch.from_numpy(u).float()
T = u.shape[1]
X = u.shape[2]
Y = u.shape[3]

fin = torch.from_numpy(force).float()
test_x = fin[-ntest:]
test_y = u[-ntest:]
test_x = test_x.reshape(ntest, T, 1, 1, 1).repeat([1, 1, X, Y, 1])

test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_x, test_y), batch_size=batch_size)
device = torch.device('cuda')

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
        orig_mean = []
        t1 = default_timer()
        for _ in range(1000):
            out = model(x)
            orig_mean.append(out.cpu())
        orig_mean = torch.stack(orig_mean, dim=0)
        orig_eu_var = orig_mean.var(dim=0)
        orig_mean = orig_mean.mean(dim=0)
        orig_var = orig_eu_var
        t2 = default_timer()
        loss = MSELoss(orig_mean, y)
        test_loss += loss.item() / ntest * batch_size
        batch_result = evaluate_uncertainty(orig_mean, orig_var, y, z=1.96)
        for k in results:
            results[k] += batch_result[k] / ntest * batch_size
        n += 1
        print(f"[eval] sample={n} | total_time={t2 - t1:.3f}s")
        io.savemat(f'pn.mat', {'rec': orig_mean[:2].cpu().detach().numpy(), 'test': y[:2].cpu().detach().numpy()})

        # for i in range(30):
        #     true = y[0, :, i * 13, 0].detach().cpu()
        #     mean = orig_mean[0, :, i * 13, 0].detach().cpu()
        #     std_eu = torch.sqrt(orig_eu_var[0, :, i * 13, 0]).detach().cpu()
        #     N = 401
        #     plt.fill_between(torch.linspace(0, N-1, steps=N), mean-2*std_eu, mean+2*std_eu, color='green',
        #                      alpha=0.5, label='EU')
        #     plt.plot(true)
        #     plt.plot(mean)
        #     plt.title(i * 13)
        #     plt.show()

print(f"[eval] test_loss={test_loss:.4e}")
