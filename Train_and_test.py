import multiprocessing
multiprocessing.set_start_method('spawn', True)
import torch
import torch.utils.data
import os
import glob
import _init_paths
import create_database
import model_2
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
from sty import fg, bg, ef, rs
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.metrics as sk
import utility as util
import argparse

global logger
util.setup_log()
logger = util.logger
use_cuda = torch.cuda.is_available()
# logger.info("Is CUDA available? %s.", use_cuda)

def parse_args():
    parser = argparse.ArgumentParser(description='Train a RNN for indoor PM')
    parser.add_argument('--hidden_size_1', dest='hidden_size_1',
                        default=300, type=int)
    parser.add_argument('--hidden_size_2', dest='hidden_size_2',
                        default=256, type=int)
    parser.add_argument('--hidden_size_3', dest='hidden_size_3',
                        default=1024, type=int)
    parser.add_argument('--coder_size_1', dest='coder_size_1',
                        default=300, type=int)
    parser.add_argument('--coder_size_2', dest='coder_size_2',
                        default=64, type=int)
    parser.add_argument('--gamma', dest='gamma',
                        default=0.9, type=float)
    parser.add_argument('--lr', dest='lr',
                        default=1e-2, type=float)
    parser.add_argument('--k1', dest='k1',
                        default=10e-5, type=float)
    parser.add_argument('--k2', dest='k2',
                        default=0.8, type=float)
    parser.add_argument('--num_layer', dest='num_layer',
                        default=6, type=int)
    parser.add_argument('--batch_size', dest='batch_size',
                        default=1024, type=int)

    parser.add_argument('--train_len', dest='train_len',
                        default=24, type=int)
    parser.add_argument('--test_len', dest='test_len',
                        default=12, type=int)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    optimizer = 'Adam'
    loss_fun = 'MSELoss'
    init_method = 'normal'
    hidden_size_1 = args.hidden_size_1
    coder_size_1 = args.coder_size_1
    coder_size_2 = args.coder_size_2
    gamma = args.gamma
    lr = args.lr
    k1 = args.k1
    k2 = args.k2
    train_len = args.train_len
    test_len = args.test_len
    num_layer = args.num_layer
    batch_size = args.batch_size
    torch.manual_seed(1)
    file_train = 'Train.csv'
    sep_train = 'Train-sep.csv'
    file_test = 'Test.csv'
    sep_test = 'Test-sep.csv'
    optimizer = 'Adam'
    loss_fun = 'MSELoss'
    init_method = 'normal'
    pmdb_train = create_database.pmdb(file_train, sep_train, file_test, sep_test, True, 24, 6)
    pmdb_test = create_database.pmdb(file_train, sep_train, file_test, sep_test, False, 24, 6)
    train_size = len(pmdb_train)
    test_size = len(pmdb_test)
    # model = model_1.da_rnn(logger = logger, pmdb_train = pmdb_train, pmdb_test = pmdb_test, batch_size = bs, input_size = 28, train_len = 24, pre_len = 6,
    #                         T = 25, learning_rate = lr)
    model = model_2.rnn(pmdb_train, pmdb_test, train_len = 24, pre_len = 6, input_size = 744, hidden_size = hidden_size_1, num_layer = num_layer,
                        batch_size = batch_size, learning_rate = lr,
                        k1 = k1, k2 = k2, coder_size_1 = coder_size_1, coder_size_2 = coder_size_2,args = args)
    model.train()
    y_pred = model.predict()