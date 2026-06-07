import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utilities3 import *
from timeit import default_timer
import matplotlib.pyplot as plt
import time
from sklearn.decomposition import PCA

class SpectralConv2d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d_fast, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat,
                             device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class FNO(nn.Module):
    def __init__(self, modes1, modes2, width):
        super(FNO, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.conv0 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.fc0 = nn.Linear(3, self.width)
        self.fc11 = nn.Linear(self.width, 128)
        self.fc21 = nn.Linear(128, 128)
        self.fc31 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.02)

        self.fc12 = nn.Linear(self.width, 128)
        self.fc22 = nn.Linear(128, 128)
        self.fc32 = nn.Linear(128, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape).cuda()
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        x = F.gelu(self.conv0(x) + self.w0(x))
        x = F.gelu(self.conv1(x) + self.w1(x))
        x = F.gelu(self.conv2(x) + self.w2(x))
        x = x.permute(0, 2, 3, 1)

        x1 = self.fc11(x)
        x1 = self.dropout(x1)
        x1 = F.gelu(x1)
        x1 = self.fc21(x1)
        x1 = self.dropout(x1)
        x1 = F.gelu(x1)
        x1 = self.fc31(x1)

        x2 = self.fc12(x)
        x2 = F.gelu(x2)
        x2 = self.fc22(x2)
        x2 = F.gelu(x2)
        x2 = self.fc32(x2)
        return torch.cat((x1, x2), dim=-1)

    def get_grid(self, shape):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1)

################################################################
# configs
################################################################
if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)
    # time.sleep(22300)

    ntrain = 4900
    ntest = 100
    modes = 10
    width = 24
    batch_size = 10
    epochs = 100
    reduction_model = 'AE_d15'
    model = FNO(8, modes, width).cuda()
    model.register_buffer('train_grid', torch.arange(0, 401, step=10))
    path_model = 'model/beam1d_au_' + f'N{ntrain}_' + f'p{len(model.train_grid)}' + f'_drop2_{reduction_model}'

    ################################################################
    # load derivative models
    ################################################################
    import sys
    sys.path.append('../reduction')
    from nn import AE
    AE = torch.load(f"../reduction/models/{reduction_model}.pth")
    AE.eval()
    for param in AE.parameters():
        param.requires_grad = False

    ################################################################
    # load data
    ################################################################
    with h5py.File('../reduction/data/data.hdf5', 'r') as f:
        u = f['acc_5000_401_1024'][:]
        u_latent = f[f'latent_{reduction_model}'][:]
        force = f['f_5000_1024'][:]
    u = torch.from_numpy(u).float()
    u_latent = torch.from_numpy(u_latent).float()
    T = u.shape[-1]
    S = u.shape[1]
    Z = u_latent.shape[-1]
    M = len(model.train_grid)

    pca = PCA(n_components=Z)
    fin = torch.from_numpy(pca.fit_transform(force)).float()

    x_normalizer = UnitGaussianNormalizer(fin[:1000].unsqueeze(1).unsqueeze(-1))
    y_normalizer = UnitGaussianNormalizer(u_latent[:1000])

    train_x = fin[:ntrain]
    train_y = u_latent[:ntrain]
    test_x = fin[-ntest:]
    test_y = u[-ntest:]
    train_x = train_x.reshape(ntrain, 1, train_x.shape[1], 1).repeat([1, S, 1, 1])
    test_x = test_x.reshape(ntest, 1, test_x.shape[1], 1).repeat([1, S, 1, 1])

    train_x = x_normalizer.encode(train_x)
    test_x = x_normalizer.encode(test_x)
    train_y = y_normalizer.encode(train_y)
    y_normalizer.cuda()

    train_y = train_y[:, model.train_grid, :]

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_x, train_y), batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_x, test_y), batch_size=batch_size)
    device = torch.device('cuda')
    ################################################################
    # training and evaluation
    ################################################################
    print(f"[model] parameters={count_params(model)}")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, total_steps=epochs, div_factor=2,
                                                    final_div_factor=1e-3 / 1e-6, pct_start=0.2)
    MSELoss = nn.MSELoss(reduction='mean').cuda()


    def gaussian_nll_loss(output, target):
        mu = output[..., 0]
        log_var = output[..., 1]
        var = torch.exp(log_var)
        nll = 0.5 * (torch.pow(target - mu, 2) / var + log_var)
        return nll.mean()

    for ep in range(epochs):
        t1 = default_timer()
        model.train()
        train_mse = 0
        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()
            optimizer.zero_grad()
            out = model(x.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)[:, model.train_grid]
            loss = gaussian_nll_loss(out, y)
            loss.backward()
            optimizer.step()
            train_mse += loss.item() / ntrain * batch_size
        scheduler.step()
        t2 = default_timer()
        t_train = t2 - t1
        print(f"[train] epoch={ep + 1:03d}/{epochs:d} | train_mse={train_mse:.4e} | train_time={t_train:.3f}s")

        if (ep + 1) % 10 == 0:
            with torch.no_grad():
                t1 = default_timer()
                test_loss = 0
                for x, y in test_loader:
                    x, y = x.cuda(), y.cuda()
                    out = model(x.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)[..., 0]
                    out = y_normalizer.decode(out)
                    out = AE.decode(out.reshape([batch_size * S, Z])).reshape([batch_size, S, T])
                    loss = MSELoss(out, y)
                    test_loss += loss.item() / ntest * batch_size
                t2 = default_timer()
                t_test = t2 - t1
            print(f"[eval] epoch={ep + 1:03d}/{epochs:d} | test_loss={test_loss:.4e} | test_time={t_test:.3f}s")
            torch.save(model, path_model)
            print(f"[checkpoint] saved={path_model}")
