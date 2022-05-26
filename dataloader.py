import os
from torch.utils import data
from torch.utils.data import DataLoader
import torch
import numpy as np
import pandas as pd
from dataset import WindowDataset
from sklearn.preprocessing import MinMaxScaler
import joblib

def get_dataloader(target_feature, data_root, log_dir, data_name, batch_size, input_window, output_window, train_rate, stride=1):
    '''
    ---Return--- 
    
    DataLoader
    test dataframe 
    ------------
    ''' 

    #-- temporary 서인천IC-부평IC 평균속도 dataset 
    if data_name == '평균속도.csv':
        df = pd.read_csv(os.path.join(data_root, data_name), encoding='CP949')
        ts_selected = df['평균속도']
        train_size = int((len(ts_selected)-output_window) * train_rate)
        valid_size = (len(ts_selected)-output_window)- train_size
        train_data = np.array(ts_selected.iloc[:train_size]).reshape(-1,1)
        valid_data = np.array(ts_selected.iloc[train_size:train_size+valid_size]).reshape(-1,1)
        scaler = MinMaxScaler()
        scaler.fit(train_data)
        train_data_std = scaler.transform(train_data)
        valid_data_std = scaler.transform(valid_data)       
        joblib.dump(scaler, os.path.join(log_dir, 'scaler.pkl'))

        #-- get dataloader
        train = pd.DataFrame(train_data_std)
        valid = pd.DataFrame(valid_data_std)

        train_dataset = WindowDataset(train, input_window, output_window)
        train_dataloader = DataLoader(train_dataset, batch_size)  

        valid_dataset = WindowDataset(valid, input_window, output_window)
        valid_dataloader = DataLoader(valid_dataset, batch_size)     

        return train_dataloader, valid_dataloader



    
    data_path = os.path.join(data_root, data_name)
    df = pd.read_csv(data_path)
    
    #-- make datetime index
    df['index'] = pd.to_datetime(df['index'], format="%Y-%m-%d %H:%M:%S")
    df.set_index('index', inplace=True)

    #-- train validation split 
    ts_selected = df[target_feature]
    train_size = int((len(ts_selected)-output_window) * train_rate)
    valid_size = (len(ts_selected)-output_window)- train_size
    train_data = np.array(ts_selected.iloc[:train_size]).reshape(-1,1)
    valid_data = np.array(ts_selected.iloc[train_size:train_size+valid_size]).reshape(-1,1)

    #-- normalize data
    scaler = MinMaxScaler()
    scaler.fit(train_data)
    train_data_std = scaler.transform(train_data)
    valid_data_std = scaler.transform(valid_data)   
    joblib.dump(scaler, os.path.join(log_dir, 'scaler.pkl'))

    #-- get dataloader
    train = pd.DataFrame(train_data_std)
    valid = pd.DataFrame(valid_data_std)

    train_dataset = WindowDataset(train, input_window, output_window)
    train_dataloader = DataLoader(train_dataset, batch_size)  

    valid_dataset = WindowDataset(valid, input_window, output_window)
    valid_dataloader = DataLoader(valid_dataset, batch_size)     

    return train_dataloader, valid_dataloader
    
