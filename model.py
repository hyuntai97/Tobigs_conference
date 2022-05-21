import torch
import torch.nn as nn
import random

#-- Encoder
class lstm_encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers = 1):
        super(lstm_encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, batch_first=True)

    def forward(self, x_input):
        lstm_out, self.hidden = self.lstm(x_input)
        return lstm_out, self.hidden

#-- Decoder     
class lstm_decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers = 1):
        super(lstm_decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size,num_layers = num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)           

    def forward(self, x_input, encoder_hidden_states):
        lstm_out, hidden = self.lstm(x_input.unsqueeze(1), encoder_hidden_states)
        output = self.linear(lstm_out)
        
        return output, hidden
    
    
#-- Seq2Seq    
class seq2seq(nn.Module):
    def __init__(self, input_size, hidden_size, target_len):
        super(seq2seq, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.target_len = target_len 
        
        self.encoder = lstm_encoder(input_size = input_size, hidden_size = hidden_size)
        self.decoder = lstm_decoder(input_size = input_size, hidden_size = hidden_size)
        
    def forward(self, inputs, targets, teacher_forcing_ratio):    
        batch_size = inputs.shape[0]
        input_size = inputs.shape[2]
        
        outputs = torch.zeros(batch_size, self.target_len, 1)
        
        _, hidden = self.encoder(inputs)
        decoder_input = inputs[:, -1, :]  # (batch_size, len_features)
        
        for t in range(self.target_len):
            out, hidden = self.decoder(decoder_input, hidden)  # out: (batch_size, 1, 1), hidden: (n_layers, batch_size, hidden_size)
            out = out.squeeze(1)
            
            if random.random() < teacher_forcing_ratio:
                decoder_input = targets[:, t, :]  # 실제 레이블을 다음 decoder의 input으로 
            else:
                decoder_input = out  # 이전 예측값을 다음 decoder의 input으로 
                
            outputs[:, t, :] = out
                
        return outputs
    
    def predict(self, inputs):
        self.eval()
        inputs = inputs.unsqueeze(0)
        batch_size = inputs.shape[0]
        input_size = inputs.shape[2]
        outputs = torch.zeros(batch_size, self.target_len, input_size)
        _, hidden = self.encoder(inputs)
        decoder_input = inputs[:,-1, :]
        for t in range(self.target_len): 
            out, hidden = self.decoder(decoder_input, hidden) #  out: (batch_size, 1, 1), hidden: (n_layers, batch_size, hidden_size)
            out =  out.squeeze(1)
            decoder_input = out
            outputs[:,t,:] = out
        return outputs.detach().numpy()[0,:,0]
    
    
