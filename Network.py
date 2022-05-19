import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

import utils


class istaNet(nn.Module):
    def __init__(self, control):
        super().__init__()
        self.control = control
        self.Ws = nn.ModuleList()
        self.Ws.extend([nn.Linear(in_features=control.nx, out_features=control.nx).to(torch.float64).cuda() for _ in range(control.L)])
        self.ReLU = nn.ReLU()

        if control.learned_b:
            self.W2s = nn.ModuleList()
            self.W2s.extend([nn.Linear(in_features=control.ny, out_features=control.nx).to(torch.float64).cuda() for _ in range(control.L)])

        for i in range(len(self.Ws)):
            if control.init_weights:
                self.Ws[i].weight = torch.nn.Parameter(control.weights[i].to(torch.float64).cuda())
            else:
                self.Ws[i].weight = torch.nn.Parameter((torch.eye(control.nx) + 0.00000001 * torch.rand([control.nx, control.nx])).to(torch.float64).cuda())
            self.Ws[i].bias = None
            if control.learned_b:
                self.W2s[i].weight = torch.nn.Parameter(control.A.transpose(1, 0).clone().to(torch.float64).cuda())
                self.W2s[i].bias = None
                if control.init_weights2:
                    self.W2s[i].weight = torch.nn.Parameter(control.weights2[i].to(torch.float64).cuda())

        if control.learnedLambda:
            self.lambdaPar = nn.ParameterList()
            self.lambdaPar.extend([nn.Parameter(torch.tensor(control.lambdaT, requires_grad=True).to(torch.float64).cuda()) for _ in range(control.L)])
        else:
            self.lambdaPar = [control.lambdaT] * control.L

        self.lin_comb = lambda x, l, b: self.Ws[l](x) + b
        if control.is_relu:
            self.soft_thresh = lambda lin_comb_x, l: self.ReLU(lin_comb_x)
        else:
            self.soft_thresh = lambda lin_comb_x, l: lin_comb_x.sign() * self.ReLU(lin_comb_x.abs() - self.lambdaPar[l])
        self.threshold = 0.5

    def forward(self, y):
        if not self.control.learned_b:
            try:
                b = torch.matmul(self.control.A.transpose(1, 0), y.transpose(1, 0) / self.control.mu).transpose(1, 0).cuda()
            except:
                b = torch.matmul(self.control.A.transpose(1, 0), y).cuda()
        try:
            dummy_shape = y.shape[1]
            x = (torch.rand([y.shape[0], self.control.nx]) > self.control.sparseFactor).to(torch.float64).cuda()
        except:
            x = (torch.rand([self.control.nx]) > self.control.sparseFactor).to(torch.float64).cuda()
        for l in range(self.control.L):
            if not self.control.learned_b:
                lin_comb_x = self.lin_comb(x, l, b)
            else:
                lin_comb_x = self.lin_comb(x, l, self.W2s[l](y))
            x = self.soft_thresh(lin_comb_x, l)
        x = torch.clip(x, min=-1.0, max=1.0)
        return x


class admmNet(nn.Module):
    def __init__(self, control):
        super().__init__()
        self.control = control
        self.Ws = nn.ModuleList()
        self.Ws.extend([nn.Linear(in_features=control.nx, out_features=control.nx).to(torch.float64).cuda() for _ in range(control.L)])

        if control.init_weights:
            for i in range(len(self.Ws)):
                self.Ws[i].weight = torch.nn.Parameter(control.weights[i]).to(torch.float64).cuda()

        if control.learned_b:
            self.W2s = nn.ModuleList()
            self.W2s.extend([nn.Linear(in_features=control.ny, out_features=control.nx).to(torch.float64).to(torch.float64).cuda() for _ in range(control.L)])

        for i in range(len(self.Ws)):
            if control.init_weights:
                self.Ws[i].weight = torch.nn.Parameter(control.weights[i].to(torch.float64).cuda())
            else:
                self.Ws[i].weight = torch.nn.Parameter((torch.eye(control.nx) + 0.00000001 * torch.rand([control.nx, control.nx])).to(torch.float64).cuda())
            self.Ws[i].bias = None
            if control.learned_b:
                self.W2s[i].weight = torch.nn.Parameter(control.A.transpose(1, 0).clone().to(torch.float64).cuda())
                self.W2s[i].bias = None
                if control.init_weights2:
                    self.W2s[i].weight = torch.nn.Parameter(control.weights2[i].to(torch.float64).cuda())

        self.soft_thresh = lambda x: x.sign() * (x.abs() - control.lambdaT_ADMM) * ((x.abs() - control.lambdaT_ADMM) > 0)

    def forward(self, y):
        try:
            b = torch.matmul(self.control.A.transpose(1, 0), y.transpose(1, 0) / self.control.mu).transpose(1, 0).to(torch.float64).cuda()
        except:
            b = torch.matmul(self.control.A.transpose(1, 0), y).to(torch.float64).cuda()
        try:
            dummy_shape = y.shape[1]
            x = torch.zeros([y.shape[0], self.control.nx]).to(torch.float64).cuda()
            u = torch.zeros([y.shape[0], self.control.nx]).to(torch.float64).cuda()
            z = torch.zeros([y.shape[0], self.control.nx]).to(torch.float64).cuda()
        except:
            x = torch.zeros([self.control.nx]).to(torch.float64).cuda()
            u = torch.zeros([self.control.nx]).to(torch.float64).cuda()
            z = torch.zeros([self.control.nx]).to(torch.float64).cuda()

        for l in range(self.control.L):
            if not self.control.learned_b:
                x = self.Ws[l](z + u) + b
            else:
                x = self.Ws[l](z + u) + self.W2s[l](y)
            z = self.soft_thresh(x - u)
            u = u - self.control.gamma * (x - z)
        return x


def trainNetwork(x_train, y_train, x_test, y_test, control):
    control.init_weights = False
    if control.networkArch == 'ista':
        net = istaNet(control)
    else:  # ADMM
        net = admmNet(control)

    plot = False
    l1Loss = nn.L1Loss()
    mseLoss= nn.MSELoss()
    CELoss = nn.CrossEntropyLoss()
    criterion = lambda x, x_pred: l1Loss(x * (x != 0), x_pred * (x != 0))
    optimizer = optim.SGD(net.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.7)

    num_epochs = int(150)
    batch_size = int(control.num_train)
    num_batches = 1
    train_losses = []
    test_losses = []
    if plot:
        plt.ion()
        fig, axes = plt.subplots(1, 1)
    relu = nn.ReLU()
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        # optimizer.zero_grad()
        for b in range(num_batches):
            x = x_train[(b * batch_size):((b + 1) * batch_size)]
            y = y_train[(b * batch_size):((b + 1) * batch_size)]

            # forward + backward + optimize
            x_pred = net(y)
            loss = criterion(x, x_pred)
            loss.backward()

        optimizer.step()
        scheduler.step()

        projected_Ws = []
        for i in range(control.L):
            inf_norm = torch.sum(net.Ws[i].weight.abs(), dim=1).max()
            if inf_norm > 10:#control.B:
                projected_Ws.append((net.Ws[i].weight / inf_norm * 5).to(torch.float64))#control.B
            else:
                projected_Ws.append(net.Ws[i].weight.to(torch.float64))
        control.init_weights = True
        control.weights = projected_Ws

        if control.networkArch == 'ista':
            net = istaNet(control)
        else:  # ADMM
            net = admmNet(control)
        net = net.cuda()
        optimizer = optim.Adam(net.parameters(), lr=0.01)

        if plot:
            train_loss = criterion(x_train, net(y_train))
            test_loss = criterion(x_test, net(y_test))
            train_losses.append(train_loss.item())
            test_losses.append(test_loss.item())
            if epoch % 200 == 200 - 1:
                axes.cla()
                axes.plot(train_losses, label='Train loss')
                axes.plot(test_losses, label='Test loss')
                plt.xlabel('Epochs')
                plt.ylabel('Loss [dB]')
                axes.legend()
                if control.is_relu:
                    plt.title("Learned ReLU")
                elif control.networkArch == 'ista':
                    plt.title("Learned ISTA")
                else: # ADMM
                    plt.title("Learned ADMM")
                fig.show()
                fig.canvas.draw()
                fig.canvas.flush_events()

    control.init_weights = False
    test_loss = criterion(x_test, net(y_test))
    return net, test_loss


def createDataset(control):
    num_samples = control.num_train + control.num_test
    x = (2*(torch.rand([num_samples, control.nx]) - 0.5)).to(torch.float64).cuda()
    for ii in range(num_samples):
        random_indices = torch.randperm(control.nx)[int(control.K):]
        x[ii, random_indices] = 0

    newK = x.sum(dim=1).max().item()
    e = control.noise_std * torch.randn([num_samples, control.ny]).to(torch.float64).cuda()

    y = torch.matmul(control.A, x.transpose(1, 0)).transpose(1, 0) + e
    return x[:control.num_train], y[:control.num_train], x[control.num_train:], y[control.num_train:], newK


def testNetwork(net, x_test, y_test, control):
    plt.ion()
    fig, axes = plt.subplots(1, 1)
    for i in range(control.num_test):
        axes.cla()
        axes.plot(x_test[i].detach().cpu(),  label='GT')
        axes.plot(net(y_test[i]).detach().cpu(), label='Network')
        # axes.plot(torch.matmul(AHAA, y_test[i]).detach().cpu(), label='Naive LS')
        axes.legend()
        plt.xlabel('Samples')
        plt.ylabel('Signal')
        plt.ylim([-1, 1])
        fig.show()
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(3)


def normNet(net, control):
    norm = sum([torch.sum(net.Ws[i].weight.abs(), dim=1).max() for i in range(control.L)]) / control.L
    return norm


def emiricalLoss(dataset_x, dataset_y, x_test, y_test, control):
    loss = []
    for _ in range(5):
        trainnedNet, testLoss_tmp = trainNetwork(dataset_x, dataset_y, x_test, y_test, control)
        loss.append(testLoss_tmp)
    return sum(loss) / len(loss), np.std([l.detach().cpu().numpy() for l in loss])


def estimationErrorGraphM(folder, control):
    plot = False
    if plot:
        plt.ion()
        fig, axes = plt.subplots(1, 1)

    dataset_size = control.num_train
    control.num_train = dataset_size
    dataset_x, dataset_y, x_test, y_test, newK = createDataset(control)
    trainedNet, expected_loss = trainNetwork(dataset_x, dataset_y, x_test, y_test, control)

    EE = []
    test_losses = []
    stds = []
    ms = np.logspace(start=0, stop=4, num=20).astype(np.int)
    for m in ms:
        control.num_train = m

        empirical_loss, std = emiricalLoss(dataset_x, dataset_y, x_test, y_test, control)
        test_losses.append(empirical_loss.detach().cpu().numpy())
        stds.append(std)
        EE_tmp = (empirical_loss - expected_loss).abs().item()
        EE.append(EE_tmp)

        if plot:
            axes.cla()
            axes.set_title('GE')
            axes.plot(ms[:len(EE)], EE,  'b-', label='ST')
            axes.legend()
            plt.xlabel('Depth')
            plt.ylabel('GE')
            fig.show()
            fig.canvas.draw()
            fig.canvas.flush_events()

    name = folder + '/'
    if control.is_relu:
        name += 'ReLU_depth_' + str(control.L)
    elif control.networkArch == 'ista':
        name += 'ISTA_depth_' + str(control.L)
    else: # ADMM
        name += 'ADMM_depth_' + str(control.L)
    if control.learned_b:
        name += '_learned_b'
    else:
        name += '_const_b'
    name += 'lambda_' + str(control.lambdaT)
    np.save(name, (EE, test_losses, stds))
    return EE, test_losses, stds


def estimationErrorGraphsISTA_ReLU(folder, control):
    const_b = True
    learned_b = True

    if const_b:
        control.learned_b = False
        control.is_relu = False
        ST = estimationErrorGraphM(folder, control)
        control.is_relu = True
        relu = estimationErrorGraphM(folder, control)

    if learned_b:
        control.learned_b = True
        control.is_relu = False
        ST_learned_b = estimationErrorGraphM(folder, control)
        control.is_relu = True
        relu_learned_b = estimationErrorGraphM(folder, control)


    plt.rcParams['font.size'] = 16
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['lines.linewidth'] = 3
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    plt.rcParams['legend.fontsize'] = 14
    plt.rcParams['figure.titlesize'] = 16
    plt.ion()
    fig, axes = plt.subplots(1, 2)

    ls = np.logspace(start=0, stop=4, num=20).astype(int)
    alpha_val = 0.8
    alpha_err = 0.6
    if const_b:
        axes[0].semilogx(ls, ST[0],  'g-', alpha=alpha_val, label='ISTA with constant bias')
        axes[0].errorbar(ls, ST[0], yerr=ST[2], marker='.', color='k', fmt='none', alpha=alpha_err)
        axes[0].semilogx(ls, relu[0],  'r-', alpha=alpha_val, label='ReLU with constant bias')
        axes[0].errorbar(ls, relu[0], yerr=relu[2], marker='.', color='k', fmt='none', alpha=alpha_err)

        axes[1].semilogx(ls, ST[1],  'g-', alpha=alpha_val, label='ISTA with constant bias')
        axes[1].errorbar(ls, ST[1], yerr=ST[2], marker='.', color='k', fmt='none', alpha=alpha_err)
        axes[1].semilogx(ls, relu[1],  'r-', alpha=alpha_val, label='ReLU with constant bias')
        axes[1].errorbar(ls, relu[1], yerr=relu[2], marker='.', color='k', fmt='none', alpha=alpha_err)

    if learned_b:
        axes[0].semilogx(ls, ST_learned_b[0],  'g--', alpha=alpha_val, label='ISTA with learned bias')
        axes[0].errorbar(ls, ST_learned_b[0], yerr=ST_learned_b[2], marker='.', color='k', fmt='none', alpha=alpha_err)
        axes[0].semilogx(ls, relu_learned_b[0],  'r--', alpha=alpha_val, label='ReLU with learned bias')
        axes[0].errorbar(ls, relu_learned_b[0], yerr=relu_learned_b[2], marker='.', color='k', fmt='none', alpha=alpha_err)

        axes[1].semilogx(ls, ST_learned_b[1],  'g--', alpha=alpha_val, label='ISTA with learned bias')
        axes[1].errorbar(ls, ST_learned_b[1], yerr=ST_learned_b[2], marker='.', color='k', fmt='none', alpha=alpha_err)
        axes[1].semilogx(ls, relu_learned_b[1],  'r--', alpha=alpha_val, label='ReLU with learned bias')
        axes[1].errorbar(ls, relu_learned_b[1], yerr=relu_learned_b[2], marker='.', color='k', fmt='none', alpha=alpha_err)


    axes[0].set_xlabel('Number of training samples')
    axes[1].set_xlabel('Number of training samples')
    axes[0].set_ylabel('Estimation error')
    axes[1].set_ylabel('$L_1$ loss')
    axes[0].legend()
    fig.show()
    fig.canvas.draw()
    fig.canvas.flush_events()


def estimationErrorGraphsLambda(folder, control):

    control.learned_b = True
    control.is_relu = False
    ST_learned_b = []

    plt.rcParams['font.size'] = 16
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    plt.rcParams['legend.fontsize'] = 14
    plt.rcParams['figure.titlesize'] = 16
    plt.rcParams['lines.linewidth'] = 3
    plt.ion()
    fig, axes = plt.subplots(1, 2)
    alpha_val = 0.7
    ls = np.logspace(start=0, stop=4, num=20).astype(np.int)

    alpha_err = 0.6
    lambdas = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    for lambdaT in lambdas:
        control.lambdaT = lambdaT
        ST_learned_b.append(estimationErrorGraphM(folder, control))
        axes[0].semilogx(ls, ST_learned_b[-1][0], alpha=alpha_val, label=r'$\lambda = ' + str(control.lambdaT) + r'$')
        axes[0].errorbar(ls, ST_learned_b[-1][0], yerr=ST_learned_b[-1][2], marker='.', color='k', fmt='none', alpha=alpha_err)
        axes[1].semilogx(ls, ST_learned_b[-1][1], alpha=alpha_val, label=r'$\lambda = ' + str(control.lambdaT) + r'$')
        axes[1].errorbar(ls, ST_learned_b[-1][1], yerr=ST_learned_b[-1][2], marker='.', color='k', fmt='none', alpha=alpha_err)

        axes[0].set_xlabel('Number of training samples')
        axes[1].set_xlabel('Number of training samples')
        axes[0].set_ylabel('Estimation error')
        axes[1].set_ylabel('$L_1$ loss')
        axes[0].legend()
        fig.show()
        fig.canvas.draw()
        fig.canvas.flush_events()
