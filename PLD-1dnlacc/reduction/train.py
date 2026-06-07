import torch
import torch.nn as nn
import time
import numpy as np
from torch import fft

##############################################
# Functions used for training ae
################################################
MSELoss = nn.MSELoss(reduction='mean').cuda()
def loss_function(rec, data):
    MSELoss = nn.MSELoss(reduction='mean').cuda()
    loss = MSELoss(rec, data)
    # ux1, uxx1 = derivative(rec)
    # ux2, uxx2 = derivative(data)
    # msex = 1e-2 * MSELoss(ux1, ux2)
    # msexx = 1e-5 * MSELoss(uxx1, uxx2)
    return loss# + msex + msexx

def derivative(u):
    dx = 1 / 1000
    ux = (u[:, 2:] - u[:, :-2]) / 2 / dx
    uxx = (u[:, 2:] - 2 * u[:, 1:-1] + u[:, :-2]) / dx ** 2
    return ux, uxx

def train_epoch(model, data, optimizer, device):
    start_epoch_time = time.time()

    loss_batch = []

    for batch in data:
        if not batch.is_cuda:
            batch = batch.to(device, non_blocking=True)

        rec = model(batch)
        loss = loss_function(rec, batch)

        loss_batch.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    return sum(loss_batch) / len(loss_batch),\
            time.time() - start_epoch_time,

def test_epoch(model, data, device):
    start_epoch_time = time.time()

    with torch.no_grad():
        loss_batch = []

        for batch in data:
            if not batch.is_cuda:
                batch = batch.to(device, non_blocking=True)

            rec = model(batch)
            loss = loss_function(rec, batch)

            loss_batch.append(loss.item())

    return  sum(loss_batch) / len(loss_batch),\
            time.time() - start_epoch_time


def printProgress(epoch, epochs, loss, loss_test, elapsed, elapsed_test):
    print(
        f"[train] epoch={epoch:03d}/{epochs:d} | "
        f"train_loss={loss:.4e} | val_loss={loss_test:.4e} | "
        f"train_time={elapsed:.3f}s | val_time={elapsed_test:.3f}s"
    )
