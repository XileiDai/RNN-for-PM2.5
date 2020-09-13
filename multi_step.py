import torch
import matplotlib.pyplot as plt
import numpy as np
import _init_paths
import create_database
import class_ProgressBar
from torch.utils.data import DataLoader
import model_2
import sklearn.metrics as sk
import pandas as pd

if __name__ == '__main__':
    torch.manual_seed(0)
    file_test = 'Test.csv'
    sep_test = 'Test-sep.csv'
    file_train = 'Train.csv'
    sep_train = 'Train-sep.csv'
    model_name = 'pm_rnn_s2.pth'
    time = pd.read_csv(file_test)['time']
    input_size = 744
    hidden_size = 300
    num_layer = 6
    pre_len = 6
    train_len = 24
    coder_size_1 = 300
    coder_size_2 = 64
    bs = 1
    num_step = 2
    pmdb_test = create_database.pmdb(file_train, sep_train, file_test, sep_test, False, 24 + num_step*6, pre_len)
    dl_test = DataLoader(pmdb_test, batch_size = bs, drop_last=True, num_workers=4)
    test_size = len(dl_test)
    device = torch.device('cpu')
    model = model_2.rnn_model(input_size, hidden_size, num_layer, pre_len , train_len, coder_size_1, coder_size_2)
    model.load_state_dict(torch.load('model/'+ model_name, map_location='cpu'))
    model.to(device)
    bar =  class_ProgressBar.ProgressBar(total = test_size)
    data_iter = iter(dl_test)
    model.eval()
    t_weight = np.zeros(24)
    weights = model.AE.encoder[0].weight.data
    weights = torch.sum(weights, dim = 0)

    pm_pred = np.zeros(test_size * bs)
    pm_target = np.zeros(test_size * bs)
    for i in range(test_size):
        bar.move()
        bar.log()
        X, y_target, y_history = next(data_iter)
        if torch.cuda.is_available():
            X = X.cuda()
            y_history = y_history.cuda()
            y_target = y_target.cuda()
        for j in range(num_step+1):
            # print(j)
            input_index = np.zeros(input_size)
            for k in range(7):
                input_index[k*24:(k+1)*24] = np.arange(0,24)+k*(24+6*num_step) + 6*j
            for k in range(7,31):
                input_index[k*24:(k+1)*24] = 7*(24+6*num_step)+np.arange(0,24)+(k-7+j*6)*24
            y_pred = model(X[:,input_index])
            X[:,(24+6*j):(30+6*j)]=y_pred
        if torch.cuda.is_available():
            y_pred = y_pred.cuda()
        # iter_losses[i*bs : (i+1)*bs] = self.loss_fun(y_pred, y_target.float()).item()
        pm_pred[i*bs : (i+1)*bs] = pmdb_test.reverse(y_pred[:,-1].detach().cpu())
        pm_target[i*bs : (i+1)*bs] = pmdb_test.reverse(y_target[:,-1].detach().cpu())
    RMSE = np.sqrt(sk.mean_squared_error(pm_pred, pm_target))
    print(RMSE)