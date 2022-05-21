import torch
import time 
import datetime
import numpy as np 
from tqdm import tqdm 



class ModelTrain:
    '''
    Model Training
    '''
    
    def __init__(
        self, 
        model, 
        trainloader, 
        epochs, 
        optimizer, 
        criterion, 
        device,
        teacher_forcing_ratio):
        
        self.model = model
        self.trainloader = trainloader
        self.device = device 
        self.optimizer = optimizer
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.criterion = criterion
        
        #-- Training time check 
        total_start = time.time()
        
        #-- Train history 
        train_loss_lst = []
                            
        #-- Iteration Epoch
        with tqdm(range(epochs)) as tr:
            for i in tr:
                train_loss = self.train()
                train_loss_lst.append(train_loss)
                
            train_loss_lst.append(train_loss)
            tr.set_postfix(loss="{0:.5f}".format(train_loss))
            
        end = time.time() - total_start
        total_time = datetime.timedelta(seconds=end)
        print('\nFinish Train: Training Time: {}\n'.format(total_time))
        
        #-- Save history 
        self.history = {}
        self.history['train'] = []
        self.history['train'].append({'train_loss':train_loss_lst})
        
        self.history['time'] = []
        self.history['time'].append({
            'epoch':epochs,
            'total':str(total_time)
        })        

        
    def train(self):
        self.model.train()
        
        train_loss = []

        for x,y,z in self.trainloader:
            self.optimizer.zero_grad()
            x = x.to(self.device).float()
            y = y.to(self.device).float()
            z = z.to(self.device).float()
            output = self.model(x,y, self.teacher_forcing_ratio).to(self.device)
            loss =self.criterion(output, y)
            loss.backward()
            self.optimizer.step()
            train_loss.append(loss.item())
            
        train_loss = np.mean(train_loss)
        
        return train_loss
            