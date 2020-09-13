import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import utility as util
from torch import optim
import numpy as np
import time
import sklearn.metrics as sk
import create_database
from sty import fg, bg, ef, rs
import matplotlib.pyplot as plt
import class_ProgressBar
import signal
import os
import pandas as pd




global logger
util.setup_log()
logger = util.logger
use_cuda = torch.cuda.is_available()
# logger.info("Is CUDA available? %s.", use_cuda)

class rnn:
    def __init__(self, pmdb_train, pmdb_test, train_len, pre_len, input_size, hidden_size, num_layer, batch_size, learning_rate,
                k1, k2, coder_size_1, coder_size_2, args):
        self.pmdb_test = pmdb_test
        self.dl_train = torch.utils.data.DataLoader(pmdb_train, batch_size = batch_size, drop_last=True, num_workers=4)
        self.dl_test = torch.utils.data.DataLoader(pmdb_test, batch_size=batch_size, drop_last=True, num_workers=4)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.k1 = k1
        self.k2 = k2
        self.coder_size_1 = coder_size_1
        self.coder_size_2 = coder_size_2
        self.cuda_use = False
        device = torch.device('cpu')
        self.model = rnn_model(input_size = input_size, hidden_size = hidden_size, num_layer = num_layer, pre_len = pre_len, train_len = train_len,
                                coder_size_1 = coder_size_1, coder_size_2 = coder_size_2)
        self.model.to(device)

        self.optimizer = optim.Adam(params = self.model.parameters(), lr = learning_rate)
        self.train_size = int(len(self.dl_train))
        self.test_size = int(len(self.dl_test))
        self.loss_fun = nn.MSELoss()
        self.train_len = train_len
        self.pre_len = pre_len
        self.success = False
        self.args = args
       

    def train(self, n_epochs =60):
        loss_iter = list()
        if self.coder_size_1<self.coder_size_2:
            return self.success
        iter_per_epoch = self.train_size
        self.iter_losses_train = np.zeros(n_epochs * iter_per_epoch)
        self.epoch_losses_train = np.zeros(n_epochs)
        self.epoch_losses_test = np.zeros(n_epochs)
        n_iter = 0
        for i in range(n_epochs):
            bar =  class_ProgressBar.ProgressBar(total = self.train_size)
            self.model.train()
            data_iter = iter(self.dl_train)
            for j in range(iter_per_epoch):
                bar.move()
                bar.log()
                X, y_target, y_history = next(data_iter)
                if self.cuda_use:
                    X = X.cuda()
                    y_history = y_history.cuda()
                    y_target = y_target.cuda()
                loss = self.train_iteration(X, y_target)
                loss_iter.append(loss)
                self.iter_losses_train[int(i * iter_per_epoch + j / self.batch_size)] = loss
                n_iter += 1
                if  i > 1:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * 1
            self.epoch_losses_train[i] = np.mean(self.iter_losses_train[range(i * iter_per_epoch, (i + 1) * iter_per_epoch)])
            text = fg.red +'(Train) Loss at epoch {}: {}'.format(i, self.epoch_losses_train[i]) + fg.rs
            print(text)
            RMSE = 0
            if i % 1 ==0 and i >0:
                y_test_pred = self.predict()
                text = fg.red +'(Test) Loss and RMSE at epoch {}: {} and {}'.format(i, y_test_pred[0], y_test_pred[1]) + fg.rs
                print(text)
                PATH = os.path.join('model/', 'pm_rnn-@{}-{}.pth'.format(i, round(y_test_pred[1],2)))
                torch.save(self.model.state_dict(), PATH)
        # data = pd.DataFrame(self.epoch_losses_train)
        # data.to_csv('C:\\Users\\xilei\\OneDrive - Macquarie University\\1 Published papers\\RNN for PM\\Code\\RNN-6\\Result\\loss_epoch_0.005_1024.csv')
        # data = pd.DataFrame(loss_iter)
        # data.to_csv('C:\\Users\\xilei\\OneDrive - Macquarie University\\1 Published papers\\RNN for PM\\Code\\RNN-6\\Result\\loss_iter_0.005_1024.csv')

            # if RMSE == y_test_pred[1]:
            #     return self.success
            # if RMSE > y_test_pred[1]:
            #     increase_time +=1
            # else:
            #     increase_time = 0
            # if increase_time>50:
            #     return self.success
            # RMSE = y_test_pred[1]
            # if i>10 and y_test_pred[1]>45:
            #     return self.success
            # if y_test_pred[1]<20:
            #     self.success = True
            #     self.save_time = str(time.time())
            #     PATH = os.path.join('model/', 'pm_rnn_{}_{}_{}_{}.pth'.format(round(y_test_pred[1], 2), i, j, self.save_time))
            #     torch.save(self.model.state_dict(), PATH)
            #     with open('model/log/' + str(round(y_test_pred[1],2)) + '-' +self.save_time +'.txt', 'w') as f:
            #         print(self.args, file = f)


    def train_iteration(self, X, y_target):
        self.optimizer.zero_grad()
        y_pred = self.model(X)
        if self.cuda_use:
            y_pred = y_pred.squeeze().cuda()
        loss_mse = self.loss_fun(y_pred.squeeze(), y_target.float())
        loss_l1 = 0
        loss_l2 = 0
        j=0
        for param in self.model.parameters():
            if j==0:
                loss_l1 += torch.sum(torch.abs(param))
                loss_l2 += torch.sum(param ** 2)
                j+=1
        
        k1 = self.k1
        k2 = self.k2
        loss = loss_mse + k1/2*((1-k2)*loss_l1 + k2*loss_l2)
        loss.backward()
        self.optimizer.step()
        return loss.item()


    def predict(self, on_test = True):
        self.model.eval()
        if on_test:
            iter_per_epoch = self.test_size
            bs = self.batch_size
            data_iter = iter(self.dl_test)
        else:
            iter_per_epoch = self.train_size
            bs = self.batch_size
            data_iter = iter(self.dl_train)
        iter_losses = np.zeros(iter_per_epoch)
        pm_pred = np.zeros(iter_per_epoch * bs)
        pm_target = np.zeros(iter_per_epoch * bs)
        bar =  class_ProgressBar.ProgressBar(total = iter_per_epoch)

        # use_input_str = input()
        # if use_input_str != 'Y':
        #     return 0, 0

        for i in range(iter_per_epoch):
            bar.move()
            bar.log()
            X, y_target, y_history = next(data_iter)
            if self.cuda_use:
                X = X.cuda()
                y_history = y_history.cuda()
                y_target = y_target.cuda()
            y_pred = self.model(X)
            if self.cuda_use:
                y_pred = y_pred.cuda()
            iter_losses[i*bs : (i+1)*bs] = self.loss_fun(y_pred, y_target.float()).item()
            pm_pred[i*bs : (i+1)*bs] = self.pmdb_test.reverse(y_pred[:,-1].detach().cpu())
            pm_target[i*bs : (i+1)*bs] = self.pmdb_test.reverse(y_target[:,-1].detach().cpu())
        RMSE = np.sqrt(sk.mean_squared_error(pm_pred, pm_target))
        print('pm_pred@1：{}',round(pm_pred[0],2))
        ###
        # result_eval.alarm_acc(pm_target, pm_pred)
        # if RMSE < 18.5:
        #    plt.figure()
        #    plt.plot(range(300), pm_target[4100:4400], label = 'pm_target')
        #    plt.plot(range(300), pm_pred[4100:4400], label = "Pred")
        #    plt.legend(loc = 'upper left')
        #    plt.show()



# plt.figure()
# plt.plot(range(100), pm_target[100:200], label = 'pm_target')
# plt.plot(range(100), pm_pred[100:200], label = "Pred")
# plt.legend(loc = 'upper left')

# plt.show()
        return np.mean(iter_losses), RMSE




class rnn_model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layer, pre_len, train_len, coder_size_1, coder_size_2):
        super(rnn_model, self).__init__()
        k = int(input_size/train_len)
        self.fc_u = nn.Linear(k, hidden_size)
        self.fc_d = nn.Linear(hidden_size + k, hidden_size)
        self.fc_c = nn.Linear(hidden_size + k, hidden_size)
        self.fc_v = nn.Linear(hidden_size + k, 1)
        self.fc_relu = nn.ReLU()
        self.fc_sig = nn.Sigmoid()

        self.hidden_size = hidden_size
        self.AE = AE_model(input_size, coder_size_1, coder_size_2)
        self.num_layer = num_layer
        self.pre_len = pre_len
        self.train_len = train_len
    
    def forward(self, X):

        X = self.AE(X)
        X = X.view(X.shape[0], -1, self.train_len)
        h_0 = self.init_hidden(X)
        if torch.cuda.is_available():
            h_0 = h_0.cuda()
        pred = Variable(torch.zeros([X.shape[0], self.pre_len]))
        for i in range(self.pre_len):
            for j in range(self.train_len):
                if j == 0:
                    h_t = self.fc_relu(self.fc_u(X[:,:,0]))
                elif j <= self.train_len - 3:
                    h_t = self.fc_relu(self.fc_d(torch.cat([h_t, X[:,:,j]], dim = 1)))
                elif j == self.train_len - 2:
                    h_0 = self.fc_relu(self.fc_d(torch.cat([h_t, X[:,:,j]], dim = 1)) + self.fc_c(torch.cat([h_0, X[:,:,j]], dim = 1)))
                elif j == self.train_len - 1:
                    pred[:,i] = self.fc_v(torch.cat([h_0, X[:,:,j]], dim = 1))[:,0]

        return pred
    
    def init_hidden(self, X):
        return torch.zeros([X.shape[0], self.hidden_size], dtype = torch.float32)

class AE_model(nn.Module):
    def __init__(self, input_size, coder_size_1, coder_size_2):
        super(AE_model, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, coder_size_1),
            nn.ReLU(True),
            nn.Linear(coder_size_1, coder_size_2),
            nn.ReLU(True))
        self.decoder = nn.Sequential(
            nn.Linear(coder_size_2, coder_size_1),
            nn.ReLU(True),
            nn.Linear(coder_size_1, input_size),
            nn.Sigmoid())

    def forward(self, x):
        x = self.encoder(x.float())
        x = self.decoder(x.float())
        return x

if __name__ == '__main__':
    X = torch.tensor([[10,20,30,40,50]])
    rnn = rnn_model(5, 300, 5, 2)
    pred = rnn(X.float())
    a = 1


