from dataloader import get_dataloader
from model import lstm_encoder, lstm_decoder, seq2seq
from train import ModelTrain

import torch
import argparse
import os 
import json
import pickle

def config_args(parser):
    parser.add_argument('--train', action='store_true', help='training model')
    parser.add_argument('--predict', action='store_true', help='forecast time series')
    
    # directory
    parser.add_argument('--datadir', type=str, default='./dataset', help='data directory')
    parser.add_argument('--logdir',type=str, default='./logs', help='logs directory')
    
    # data
    parser.add_argument('--dataname', type=str, default='sample_upbit_daydata.csv', help='dataset name')
    parser.add_argument('--target_feature', type=str, default='open', help='the target feature')
    parser.add_argument('--input_window', type=int, default=336, help='input window size')
    parser.add_argument('--output_window', type=int, default=168, help='output window size')
    parser.add_argument('--hidden_size', type=int, default=16, help='lstm hidden size')
    parser.add_argument('--stride', type=int, default=1, help='stride size')
    
    # train options
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='number of batch')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--teacher_forcing', type=float, default=0.7, help='teacher forcing ratio of seq2seq model')
    
    args = parser.parse_args()
    
    return args


if __name__ == '__main__':
    # set arguments
    parser = argparse.ArgumentParser(description='time series forcasting')
    args = config_args(parser)
    
    # gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Current cuda device ', torch.cuda.current_device()) 
    
    # save arguments
    json.dump(vars(args), open(os.path.join(args.logdir,'arguments.json'),'w'), indent=4)
    
    # set seed 
    
    # load data 
    train_dataloader, test = get_dataloader(target_feature=args.target_feature,
                                            data_root=args.datadir,
                                            log_dir=args.logdir,
                                            data_name=args.dataname,
                                            batch_size=args.batch_size,
                                            input_window=args.input_window,
                                            output_window=args.output_window,
                                            stride=args.stride)  

    # build models
    model = seq2seq(1, args.hidden_size, args.output_window)
    model.to(device)
    
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # criterion
    criterion = torch.nn.MSELoss()
    
    # Training 
    if args.train:
        modeltrain = ModelTrain(
                        model, 
                        train_dataloader,
                        args.epochs,
                        optimizer,
                        criterion,
                        device,
                        args.teacher_forcing)
    
        # save history 
        pickle.dump(modeltrain.history, open(os.path.join(args.logdir, 'history.pkl'),'wb'))
        # save model 
        torch.save(modeltrain.model.state_dict(), os.path.join(args.logdir, 'model.pth'))        
    
        
        

        