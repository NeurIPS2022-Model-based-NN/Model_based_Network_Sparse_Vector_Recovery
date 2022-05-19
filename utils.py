import torch
import torch.linalg
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.interpolate


class Control:
    def __init__(self):
        self.networkArch = 'ista'
        self.is_relu = False
        self.L = 10
        power = 11
        self.num_train = 2**power
        self.num_test = 2**9
        self.num_samples = self.num_train + self.num_test
        self.nx = 20
        self.ny = 10
        self.sparseFactor = 0.85
        self.lambdaT = 3.3
        self.lambdaT_ADMM = 1.2
        self.mu = 1
        self.gamma = 0.05
        self.A = dftmtx(self.nx, self.ny).clone().to(torch.float64).cuda()

        self.numPredictors = 20
        self.init_weights = False
        self.init_weights2 = False
        self.learned_b = True
        self.learnedLambda = True

        self.numSampleSets = 20
        self.numSamples = 1000

        self.noise_std = 0.1
        self.K = np.ceil(self.nx * (1 - self.sparseFactor))
        self.G = torch.linalg.matrix_norm(self.A.cpu(), ord=2) * np.sqrt(self.K) + self.noise_std

        # self.B = 1 + self.lambdaT / (np.sqrt(self.numSamples) * self.G) - 0.01
        self.B = 1 + self.lambdaT / (2 * self.K * np.sqrt(self.numSamples)) - 0.00001


def dftmtx(nx, ny):
    DFT = torch.fft.fft(torch.eye(nx)).real
    random_indices = torch.randperm(nx)[:ny]
    return DFT[random_indices, :]


def interpolate(x, y, x_new):
    if len(x) > 4:
        f = scipy.interpolate.interp1d(x, y, kind='linear')
        y_new = f(x_new)
        return y_new
    else:
        return y
