import time
import torch
import argparse
from runners import aeRunner

parser = argparse.ArgumentParser()
args = parser.parse_args()
device = ('cuda' if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    datafile = 'data/data.hdf5'
    ae = aeRunner(device, datafile)
    # time.sleep(850)
    ae.train()
