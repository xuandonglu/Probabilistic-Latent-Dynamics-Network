import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class DPN(nn.Module):
    def __init__(self, input_dim, latent_dim, pattern):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.branch_net = self.buildBranch(latent_dim, pattern)
        self.trunk_net = self.buildTrunk(latent_dim, pattern)

    def buildBranch(self, latent_dim, pattern):
        branch = nn.Sequential(ResidualBlock(self.input_dim[0] * self.input_dim[1], 128, 128, 8),
                               ResidualBlock(128, 128, latent_dim, 8), )

        if pattern == 'fno' or pattern == 'fno1':
            model = nn.Sequential(FNO1(24, 24, 1, 1), branch)
        else:
            model = branch
        return model

    def buildTrunk(self, latent_dim, pattern):
        trunk = nn.Sequential(ResidualBlock(2, 128, 128, 8),
                              ResidualBlock(128, 128, latent_dim, 8), )
        if pattern == 'fno' or pattern == 'fno2':
            model = FNO2(16, 16, latent_dim)
        else:
            model = trunk
        return model

    def forward(self, data):
        z = self.encode(data)
        rec = self.decode(z)
        return rec

    def encode(self, data):
        branch_out = self.branch_net(data)
        return branch_out

    def trunk(self, z):
        g = self.get_grid(self.input_dim[0], self.input_dim[1], z.device)
        trunk_out = self.trunk_net(g)
        return trunk_out

    def decode(self, z):
        trunk_out = self.trunk(z)
        y = torch.einsum("bi,xyi->bxy", z, trunk_out)
        return y

    def get_grid(self, size_x, size_y, device):
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(size_x, 1, 1).repeat([1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, size_y, 1).repeat([size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)


class FNO1(nn.Module):
    def __init__(self, modes1, modes2, width, input_dim):
        super(FNO1, self).__init__()
        self.p = nn.Linear(input_dim, width)
        self.conv0 = SpectralConv2d(width, width, modes1, modes2)
        self.q = nn.Linear(width, input_dim)

    def forward(self, x):
        x = self.p(x.unsqueeze(-1))
        x = x.permute(0, 3, 1, 2)
        x = self.conv0(x)
        x = F.gelu(x)
        x = x.permute(0, 2, 3, 1)
        x = self.q(x)
        return x.reshape([x.shape[0], x.shape[1] * x.shape[2]])


class FNO2(nn.Module):
    def __init__(self, modes1, modes2, width):
        super(FNO2, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.fc0 = nn.Linear(2, self.width)
        self.fc1 = nn.Linear(self.width, width)

    def forward(self, x):
        x = self.fc0(x.unsqueeze(0))
        x = x.permute(0, 3, 1, 2)
        x = F.gelu(self.conv0(x) + self.w0(x))
        x = F.gelu(self.conv1(x) + self.w1(x))
        x = F.gelu(self.conv2(x) + self.w2(x))
        x = F.gelu(self.conv3(x) + self.w3(x))
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        return x.squeeze(0)


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.GELU())
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)
        self.residual = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

    def forward(self, x):
        return self.net(x) + self.residual(x)


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
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


max_channel = 256
init_dim = 1
class AE(nn.Module):
    def __init__(self, input_dim, latent_dim, pattern):
        super().__init__()

        self.latent_dim = latent_dim
        self.encoder = self.buildEncoder(latent_dim)
        self.decoder = self.buildDecoder(latent_dim)

    def buildEncoder(self, latent_dim):
        encoder = nn.Sequential(
            nn.ConstantPad2d((1, 0, 1, 0), 0),
            nn.Conv2d(in_channels=init_dim, out_channels=8, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.ConstantPad2d((1, 0, 1, 0), 0),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.ConstantPad2d((0, 0, 1, 0), 0),
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.ConstantPad2d((0, 0, 1, 0), 0),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(max_channel * 6, max_channel),
            nn.ELU(),
            nn.Linear(max_channel, latent_dim),
        )
        return encoder

    def buildDecoder(self, latent_dim):
        decoder = nn.Sequential(
            nn.Linear(latent_dim, max_channel),
            nn.ELU(),
            nn.Linear(max_channel, max_channel * 6),
            nn.ELU(),
            nn.Unflatten(dim=1, unflattened_size=(max_channel, 3, 2)),
            nn.ELU(),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ELU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ConstantPad2d((0, 0, -1, 0), 0),
            nn.ELU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ConstantPad2d((0, 0, -1, 0), 0),
            nn.ELU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ConstantPad2d((-1, 0, -1, 0), 0),
            nn.ELU(),
            nn.ConvTranspose2d(in_channels=8, out_channels=init_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ConstantPad2d((-1, 0, -1, 0), 0),
        )
        return decoder

    def forward(self, data):
        z = self.encoder(data.unsqueeze(1))
        reconstruction = self.decoder(z)
        return reconstruction.squeeze(1)

    def encode(self, data):
        z = self.encoder(data.unsqueeze(1))
        return z

    def decode(self, data):
        return self.decoder(data).squeeze(1)
