import torch.nn as nn
import torch.nn.functional as F
import torch
from base import BaseModel
from torch.autograd import Variable
import random


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
    def reset_hidden_state(self):
        self.hidden = (
            torch.zeros(self.n_layers, self.seq_len-1, self.n_hidden),
            torch.zeros(self.n_layers, self.seq_len-1, self.n_hidden)
        )
    def forward(self, sequences):
        batch_size, seq_len  = sequences.size()
        self.reset_hidden_state()

        sequences = sequences.to(dtype=self.dtype)
        self.hidden = (self.hidden[0].to(dtype=self.dtype), self.hidden[1].to(dtype=self.dtype))

        sequences = self.c1(sequences.view(len(sequences), 1, -1))  
        lstm_out, self.hidden = self.lstm(
            sequences.view(batch_size, seq_len - 1, -1),  
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
  

class GRU(nn.Module):
    def __init__(self, n_features, n_hidden, seq_len, n_layers, dropout=0.2, n_classes = 1) :
        super(GRU, self).__init__()
        self.dtype = torch.float32
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.n_features = n_features
        self.seq_len = seq_len
        
        # Hidden layers의 뉴런 수를 리스트로 저장
        hidden_units = [128, 64, 32, 16]
        self.hidden_units = hidden_units

        # GRU 레이어 정의
        self.gru_layers = nn.ModuleList([
            nn.GRU(
                input_size=n_features,
                hidden_size=hidden_unit,
                num_layers=1,  # 각 GRU 레이어는 1개의 hidden layer만 사용
                batch_first=True,
                dropout=dropout
            ) for hidden_unit in hidden_units
        ])

        # 선형 레이어 추가
        self.fc = nn.Linear(hidden_units[-1], n_classes)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = x.unsqueeze(2).type(self.dtype)
        batch_size = x.size(0)
        h_0 = [Variable(torch.zeros(1, batch_size, hidden_unit)) for hidden_unit in self.hidden_units]
        h_n = []

        for i in range(len(self.gru_layers)):
            gru_layer = self.gru_layers[i]
            out, hn = gru_layer(x, h_0[i])
            h_n.append(hn)

        out = self.tanh(h_n[-1].view(batch_size, -1))  # 마지막 hidden layer의 출력
        out = self.fc(out)

        return out