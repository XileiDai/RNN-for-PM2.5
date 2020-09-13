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

        ori_data_train = pd.read_csv(self.file_path_train).values
        ori_data_test = pd.read_csv(self.file_path_test).values

        # self.scaler = MinMaxScaler()
        # self.scaler.fit(ori_data_train)
        # ori_data_train = self.scaler.transform(ori_data_train)
        # ori_data_test = self.scaler.transform(ori_data_test)

        if on_train:
            self.sep = pd.read_csv(self.sep_path_train)['index'].values
        #     time_of_day = np.array(pd.read_csv(self.file_path_train).values[:,-1])
            self.ori_data = ori_data_train
        else:
            self.sep = pd.read_csv(self.sep_path_test)['index'].values
        #     time_of_day = pd.read_csv(self.file_path_test).values[:,-1]
            self.ori_data = ori_data_test

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

        pm_in = self.ori_data[t_end+self.pred_len-1,2]
        time = self.ori_data[t_end+self.pred_len-1, 1]
        result = [time, pm_in]
        

        return result

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

    # def get_seq(self, testset = True):


if __name__=='__main__':
    sep_path_train = 'Train-sep.csv'
    sep_path_test = 'Test-sep.csv'
    file_path_train = 'Train.csv'
    file_path_test = 'Test.csv'
    pmdb = pmdb(file_path_train, sep_path_train, file_path_test, sep_path_test, on_train=False, train_len = 6)
    testset = []
    for i in range(len(pmdb)):
        testset.append(pmdb[i])
    testset = np.array(testset)
    result = pd.DataFrame({'time':testset[:,0], 'Target':testset[:,1]})
    result.to_csv('Result/testset-6.csv')
    a=2

