"""
Runners for the ae and temporal-dynamic prediction in latent space 
@yuningw
"""

import os 
import time
from pathlib import Path
import h5py
import numpy as np
import torch 
from torch import nn
from train import *
from datas import *

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

####################################################
### RUNNER for AE
####################################################
# torch.manual_seed(0)
# np.random.seed(0)

class aeRunner(nn.Module):
    def __init__(self, device, datafile) -> None:

        from configs import AE_config as cfg
        from nomenclature import Name_AE

        super(aeRunner, self).__init__()

        self.config = cfg
        self.filename = Name_AE(self.config)
        self.datafile = datafile
        import importlib
        model = getattr(importlib.import_module('nn'), self.config.model)
        self.input_dim = loadData(self.datafile).shape[-2:]
        self.model = model(self.input_dim, self.config.latent_dim, self.config.pattern)

        self.device = device
        self.model.to(device)
        self.fmat = '.pth'

        print(f"[init] model=AE | device={device}")
        print(f"[case] name={self.filename}")

#-------------------------------------------------
    def train(self):
        print("[train] starting")
        self.get_data()
        self.compile()
        self.fit()
        self.train_dl   = None
        self.val_dl     = None
        print("[train] finished | data_loaders=cleared")

#-------------------------------------------------

    def get_data(self):

        u_scaled = loadData(self.datafile)
        u_scaled = u_scaled[::self.config.downsample]
        n_total = u_scaled.shape[0]
        self.n_train = n_total - self.config.n_test
        print(
            f"[data] n_train={self.n_train:d} | "
            f"n_test={self.config.n_test:d} | n_total={n_total:d}"
        )

        self.train_dl, self.val_dl = get_ae_DataLoader(    d_train=u_scaled,
                                                            n_train=self.n_train,
                                                            device= self.device,
                                                            batch_size= self.config.batch_size)
        print(f"[data] train_batches={len(self.train_dl)} | val_batches={len(self.val_dl)}")

#-------------------------------------------------
    def compile(self):

        from torch.optim import lr_scheduler

        print("[compile] starting")

        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.config.lr, weight_decay=0)

        self.opt_sch = lr_scheduler.OneCycleLR(self.opt,
                                            max_lr=self.config.lr,
                                            total_steps=self.config.epochs,
                                            div_factor=2,
                                            final_div_factor=self.config.lr/self.config.lr_end,
                                            pct_start=0.2)

        print("[compile] finished")

#-------------------------------------------------
    def fit(self):

        print(f"[train] case={self.filename}")
        bestloss = 1e6
        loss = 1e6
        ckp_file = f'models/{self.filename}' + self.fmat
        for epoch in range(1, self.config.epochs + 1):
            self.model.train()
            loss, elapsed = train_epoch(model=self.model,
                                        data=self.train_dl,
                                        optimizer=self.opt,
                                        device=self.device)
            self.model.eval()
            loss_test, elapsed_test = test_epoch(model=self.model,
                                                 data=self.val_dl,
                                                 device=self.device)

            self.opt_sch.step()

            printProgress(epoch=epoch,
                          epochs=self.config.epochs,
                          loss=loss,
                          loss_test=loss_test,
                          elapsed=elapsed,
                          elapsed_test=elapsed_test)

            if (loss_test < bestloss and epoch > 100):
                bestloss = loss_test
                torch.save(self.model, ckp_file)
                print(f"[checkpoint] epoch={epoch:d} | val_loss={loss_test:.4e} | saved={ckp_file}")
        torch.save(self.model, ckp_file)
        print(
            f"[checkpoint] final_epoch={self.config.epochs:d} | "
            f"train_loss={loss:.4e} | val_loss={loss_test:.4e} | saved={ckp_file}"
        )
