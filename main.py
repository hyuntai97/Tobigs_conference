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
    parser.add_argument('--test', action='store_true', help='testing model')
    parser.add_argument('--predict', action='store_true', help='forecast time series')
    
    # directory
    parser.add_argument('--datadir', type=str, default='./dataset', help='data directory')
    parser.add_argument('--logdir',type=str, default='./logs', help='logs directory')
    parser.add_argument('--savedir',type=str, default='./save', help='save directory')
    
    # data
    parser.add_argument('--dataname', type=str, default='upbit_ohlcv_1700.csv', help='dataset name')
    parser.add_argument('--target_feature', type=str, default='open', help='the target feature')
    parser.add_argument('--input_window', type=int, default=90, help='input window size')
    parser.add_argument('--output_window', type=int, default=28, help='output window size')
    parser.add_argument('--hidden_size', type=int, default=32, help='lstm hidden size')
    parser.add_argument('--stride', type=int, default=1, help='stride size')
    parser.add_argument('--train_rate', type=float, default=0.7, help='train rate')
    
    # train options
    parser.add_argument('--epochs', type=int, default=3000, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='number of batch')
    parser.add_argument('--lr', type=float, default=0.004, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')
    parser.add_argument('--teacher_forcing', type=float, default=1, help='teacher forcing ratio of seq2seq model')
    
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
    train_dataloader, valid_dataloader = get_dataloader(target_feature=args.target_feature,
                                            data_root=args.datadir,
                                            log_dir=args.logdir,
                                            data_name=args.dataname,
                                            batch_size=args.batch_size,
                                            input_window=args.input_window,
                                            output_window=args.output_window,
                                            train_rate=args.train_rate,
                                            stride=args.stride)  

    # build models
    model = seq2seq(1, args.hidden_size, args.output_window)
    model.to(device)
    
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # criterion
    criterion = torch.nn.MSELoss()
    
    # Training 
    if args.train:
        modeltrain = ModelTrain(
                        model, 
                        train_dataloader,
                        valid_dataloader,
                        args.epochs,
                        optimizer,
                        criterion,
                        device,
                        args.teacher_forcing)
    
        # save history 
        pickle.dump(modeltrain.history, open(os.path.join(args.logdir, 'history.pkl'),'wb'))
        # save model 
        torch.save(modeltrain.model.state_dict(), os.path.join(args.logdir, 'model.pth'))        
    
    # Test 
    if args.test:
        print('hi')

        

        