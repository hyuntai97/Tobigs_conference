from torch.utils.data import Dataset
import torch


class WindowDataset(Dataset):
    '''
    Build a custom dataset 
    
    --Return--
    
    x: inputs
    y: targets 
    z: features
    ----------
    
    '''
    
    def __init__(self, data, input_window, output_window, stride=1):
        # total data length 
        L = data.shape[0]
        
        # total number of samples with stride
        num_samples = (L - input_window - output_window) // stride + 1
        
        # input, output 
        X = []
        Y = []
        Z = []
        
        for i in range(num_samples):
            start_x = stride*i
            end_x = start_x + input_window 
            X.append(data.iloc[start_x: end_x, :].values)
            
            start_y = stride*i + input_window
            end_y = start_y + output_window 
            Y.append(data.iloc[start_y:end_y, 0].values)
            Z.append(data.iloc[start_y:end_y, 1:].values)
            
        self.x = X
        self.y = Y
        self.z = Z
        
        self.len = len(X)
            
    def __len__(self):
        return self.len 
    
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x[idx])
        y = torch.FloatTensor(self.y[idx])
        y = y.unsqueeze(-1)
        z = torch.FloatTensor(self.z[idx])
        
        return x, y, z