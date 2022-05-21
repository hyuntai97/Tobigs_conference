import os
from torch.utils.data import DataLoader
import torch
import numpy as np
import pandas as pd
from dataset import WindowDataset
from sklearn.preprocessing import MinMaxScaler
import joblib

def get_dataloader(target_feature, data_root, log_dir, data_name, batch_size, input_window, output_window, stride=1):
    '''
    ---Return--- 
    
    DataLoader
    test dataframe 
    ------------
    ''' 
    
    data_path = os.path.join(data_root, data_name)
    df = pd.read_csv(data_path)
    
    df['datetime'] = pd.to_datetime(df['Unnamed: 0'], format="%Y-%m-%d %H:%M:%S")
    df.drop(['Unnamed: 0', 'ticker'], axis=1, inplace=True)
    df.set_index('datetime', inplace=True)
    
    scaler = MinMaxScaler()
    joblib.dump(scaler, os.path.join(log_dir, 'scaler.pkl'))
    
    df[target_feature] = scaler.fit_transform(df[target_feature].to_numpy().reshape(-1, 1))
    
    train = pd.DataFrame(df.iloc[:-output_window][target_feature])
    test = pd.DataFrame(df.iloc[-output_window:][target_feature])
    
    
    train_dataset = WindowDataset(train, input_window, output_window)
    train_dataloader = DataLoader(train_dataset, batch_size)
    
    return train_dataloader, test 
