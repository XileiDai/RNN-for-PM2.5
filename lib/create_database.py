import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


class pmdb(object):
    def __init__(self, file_path_train, sep_path_train, file_path_test, sep_path_test, on_train, train_len = 24, pred_len = 6):
        self.file_path_train = file_path_train
        self.sep_path_train = sep_path_train
        self.file_path_test = file_path_test
        self.sep_path_test  = sep_path_test
        self.train_len = train_len
        self.pred_len = pred_len
        self.sample_len = train_len + pred_len

        ori_data_train = pd.read_csv(self.file_path_train).values[:,2:-1]
        ori_data_test = pd.read_csv(self.file_path_test).values[:,2:-1]

        self.scaler = MinMaxScaler()
        self.scaler.fit(ori_data_train)
        ori_data_train = self.scaler.transform(ori_data_train)
        ori_data_test = self.scaler.transform(ori_data_test)

        if on_train:
            self.sep = pd.read_csv(self.sep_path_train)['index'].values
            time_of_day = np.array(pd.read_csv(self.file_path_train).values[:,-1])
            self.ori_data = np.column_stack([ori_data_train, time_of_day]).astype(float)
        else:
            self.sep = pd.read_csv(self.sep_path_test)['index'].values
            time_of_day = pd.read_csv(self.file_path_test).values[:,-1]
            self.ori_data = np.column_stack([ori_data_test, time_of_day]).astype(float)

        self.begin_index = []
        for i in range(len(self.sep)):
            if i == 0:
                begin = 0
            else:
                begin = self.sep[i-1]+1
            end = self.sep[i]
            self.begin_index = np.append(self.begin_index, list(range(begin, end-self.sample_len + 1)))
            self.begin_index = self.begin_index.astype(int)


    def __getitem__(self, index):
        t_begin = self.begin_index[index]
        t_end = t_begin + self.train_len
        t_range = np.arange(t_begin,t_end)
        pm_out = self.ori_data[t_range,1]
        pm_in = self.ori_data[t_range,0]
        tem_out = self.ori_data[t_range,3]
        tem_in = self.ori_data[t_range,2]
        rh_out = self.ori_data[t_range,5]
        rh_in = self.ori_data[t_range,4]
        co2 = self.ori_data[t_range,6]
        time_t_range = self.ori_data[t_range,7]
        time_input = np.zeros([self.train_len * 24])
        for i in range(len(t_range)):
            time_input[i*24 + int(time_t_range[i])]=1
        pm_in_target = self.ori_data[t_end : (t_end + self.pred_len), 0].squeeze()

        ## You can comment the below lines to make the corresponding inputs to be zero
        # tem_out = np.zeros(tem_out.shape)
        # tem_in = np.zeros(tem_in.shape)
        # rh_in = np.zeros(rh_in.shape)
        # rh_out = np.zeros(rh_out.shape)
        # co2 = np.zeros(co2.shape)
        time_input = np.zeros(time_input.shape)


        pre_input = np.concatenate([pm_in, pm_out, tem_out, tem_in, rh_in, rh_out, co2, time_input]).squeeze()
        

        return pre_input.astype(float).squeeze(), pm_in_target.astype(float).squeeze(), pm_in.astype(float).squeeze()

    def __len__(self):
        return len(self.begin_index)

    def reverse(self, rnn_output):
        other = np.ones([rnn_output.shape[0], self.ori_data.shape[1] - 2])
        pred = self.scaler.inverse_transform(np.column_stack([rnn_output.detach().numpy(), other]))[:,0]
        return pred.squeeze()
    
    def pre_show(self, rnn_output, gt):
        pred, gt = self.reverse(rnn_output, gt)
        df = pd.DataFrame({'pred': pred[:,0,0], 'gt':gt[0,:]})
        plt.plot('pred', data = df,color='red')
        plt.plot('gt', data = df,color='green')
        plt.show()

if __name__=='__main__':
    sep_path_train = 'Train-sep.csv'
    sep_path_test = 'Test-sep.csv'
    file_path_train = 'Train.csv'
    file_path_test = 'Test.csv'
    pmdb = pmdb(file_path_train, sep_path_train, file_path_test, sep_path_test, on_train=True)
    a = pmdb[2]
    a=2

