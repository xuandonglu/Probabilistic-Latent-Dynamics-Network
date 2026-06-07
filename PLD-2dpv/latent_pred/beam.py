import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utilities3 import *
from timeit import default_timer
import matplotlib.pyplot as plt
import time
from sklearn.decomposition import PCA

class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))
        self.weights3 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))
        self.weights4 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))

    def compl_mul3d(self, input, weights):
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1) // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x

class FNO(nn.Module):
    def __init__(self, modes1, modes2, modes3, width):
        super(FNO, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.conv0 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.w0 = nn.Conv3d(self.width, self.width, 1)
        self.w1 = nn.Conv3d(self.width, self.width, 1)
        self.w2 = nn.Conv3d(self.width, self.width, 1)
        self.fc0 = nn.Linear(4, self.width)
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.02)

    def forward(self, x):
        grid = self.get_grid(x.shape).cuda()
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = F.gelu(self.conv0(x) + self.w0(x))
        x = F.gelu(self.conv1(x) + self.w1(x))
        x = F.gelu(self.conv2(x) + self.w2(x))
        x = x.permute(0, 2, 3, 4, 1)

        x = self.fc1(x)
        x = self.dropout(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = F.gelu(x)
        x = self.fc3(x)
        return x.squeeze(-1)

    def get_grid(self, shape):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1)

################################################################
# configs
################################################################
if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)
    # time.sleep(6000)

    ntrain = 4900
    ntest = 100
    modes = 10
    width = 24
    batch_size = 2
    epochs = 100
    learning_rate = 0.001
    model = FNO(modes, modes, modes, width).cuda()
    model.register_buffer('train_grid', torch.arange(0, 201, step=5))
    path_model = 'model/pv2d_' + f'N{ntrain}_' + f'p{len(model.train_grid)}' + '_drop2'

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

    train_x = fin[:ntrain]
    train_y = u[:ntrain, model.train_grid]
    test_x = fin[-ntest:]
    test_y = u[-ntest:]
    train_x = train_x.reshape(ntrain, T, 1, 1, 1).repeat([1, 1, X, Y, 1])
    test_x = test_x.reshape(ntest, T, 1, 1, 1).repeat([1, 1, X, Y, 1])

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
    for ep in range(epochs):
        t1 = default_timer()
        model.train()
        train_mse = 0
        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()
            optimizer.zero_grad()
            out = model(x)
            loss = MSELoss(out[:, model.train_grid], y)
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
                    out = model(x)
                    loss = MSELoss(out, y)
                    test_loss += loss.item() / ntest * batch_size
                t2 = default_timer()
                t_test = t2 - t1
            print(f"[eval] epoch={ep + 1:03d}/{epochs:d} | test_loss={test_loss:.4e} | test_time={t_test:.3f}s")
            torch.save(model, path_model)
            print(f"[checkpoint] saved={path_model}")
