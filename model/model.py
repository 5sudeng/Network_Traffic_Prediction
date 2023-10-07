import torch.nn as nn
import torch.nn.functional as F
import torch
from base import BaseModel
from torch.autograd import Variable


class LSTM(nn.Module):
  def __init__(self, n_features, n_hidden, seq_len, n_layers, dropout=0.2): 
      super(LSTM, self).__init__()
      self.dtype = torch.float32
      self.n_hidden = n_hidden
      self.seq_len = seq_len
      self.n_layers = n_layers
      self.lstm = nn.LSTM(
          input_size=n_features,
          hidden_size=n_hidden,
          num_layers=n_layers,
          dropout = dropout
      )
      self.linear = nn.Linear(in_features=n_hidden, out_features=1)
  def reset_hidden_state(self, *args):
      self.hidden = (
          torch.zeros(self.n_layers, self.seq_len, self.n_hidden),
          torch.zeros(self.n_layers, self.seq_len, self.n_hidden)
      )
  def forward(self, sequences):
      batch_size, seq_len = sequences.size()
      self.reset_hidden_state(batch_size)

      sequences = sequences.to(dtype=self.dtype)
      self.hidden = (self.hidden[0].to(dtype=self.dtype), self.hidden[1].to(dtype=self.dtype))

      lstm_out, self.hidden = self.lstm(
          sequences.view(len(sequences), self.seq_len, -1),
          self.hidden
      )
      last_time_step = lstm_out.view(self.seq_len, len(sequences), self.n_hidden)[-1]
      y_pred = self.linear(last_time_step)
      return y_pred
    
class CNN_LSTM(nn.Module):
    def __init__(self, n_features, n_hidden, seq_len, n_layers, dropout=0.2):
        super(CNN_LSTM, self).__init__()
        self.dtype = torch.float32
        self.n_hidden = n_hidden
        self.seq_len = seq_len
        self.n_layers = n_layers
        self.c1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size = 2, stride = 1) # 1D CNN 레이어 추가
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=n_hidden,
            num_layers=n_layers,
            dropout = dropout
        )
        self.linear = nn.Linear(in_features=n_hidden, out_features=1)
    def reset_hidden_state(self, batch_size):
        self.hidden = (
            torch.zeros(self.n_layers, batch_size, self.n_hidden),
            torch.zeros(self.n_layers, batch_size, self.n_hidden)
        )
    def forward(self, sequences):
        batch_size, seq_len  = sequences.size()
        self.reset_hidden_state(batch_size)

        sequences = sequences.to(dtype=self.dtype)
        self.hidden = (self.hidden[0].to(dtype=self.dtype), self.hidden[1].to(dtype=self.dtype))

        sequences = self.c1(sequences.view(batch_size, 1, seq_len))  
        lstm_out, self.hidden = self.lstm(
            sequences.view(seq_len - 1, batch_size, -1),  
            self.hidden
        )
        last_time_step = lstm_out.view(seq_len - 1, batch_size, self.n_hidden)[-1]  
        y_pred = self.linear(last_time_step)
        return y_pred
    

class RNN(nn.Module):
    def __init__(self, n_features, n_hidden, seq_len, n_layers, dropout=0.2):
        super(RNN, self).__init__()
        self.dtype = torch.float32
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.seq_len = seq_len
        self.rnn = nn.RNN(
            n_features, 
            n_hidden, 
            n_layers, 
            batch_first=True, 
            dropout=dropout
        )
        self.rnn = self.rnn.to(self.dtype)
        self.fc = nn.Sequential(nn.Linear(n_hidden * seq_len, 1), nn.Sigmoid())

    def forward(self, x):
        x = x.unsqueeze(2).type(self.dtype)
        h0 = torch.zeros(self.n_layers, x.size(0), self.n_hidden, dtype=self.dtype) # 초기 hidden state 설정
        out, _ = self.rnn(x, h0) # out: RNN의 마지막 레이어로부터 나온 output feature 를 반환 hn: hidden state를 반환
        out = out.reshape(out.shape[0], -1) # many to many 전략
        out = self.fc(out)
        return out
  

class GRU(nn.Module) :
    def __init__(self, n_features, n_hidden, seq_len, n_layers, dropout=0.2, n_classes = 1) :
        super(GRU, self).__init__()
        self.dtype = torch.float32
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.seq_len = seq_len
        
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=n_hidden,
            num_layers=n_layers,
            batch_first=True, 
            dropout=dropout
            )
        self.fc_1 = nn.Linear(n_hidden, 128)
        self.fc = nn.Linear(128, n_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x) :
        x = x.unsqueeze(2).type(self.dtype)
        h_0 = Variable(torch.zeros(self.n_layers, x.size(0), self.n_hidden))
        output, (hn) = self.gru(x, (h_0))
        hn = hn.view(-1, self.n_hidden)
        out = self.relu(hn)
        out = self.fc_1(out)
        out = self.relu(out)
        out = self.fc(out)
        return out


#python test.py --resume saved/models/CNNLSTMmodel/1004_001611/model_best.pth
#python test.py --resume saved/models/LSTMmodel/1004_004611/model_best.pth
#python test.py --resume saved/models/RNNmodel/1004_031931/model_best.pth
#python test.py --resume saved/models/GRUmodel/1004_035725/model_best.pth