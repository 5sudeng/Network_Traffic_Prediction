import pandas as pd
import numpy as np
import datetime
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, data_dir, sequence_length, training):
        df = pd.read_csv(data_dir, delimiter=";")
        df["clock"] = df["clock"].apply(lambda x: datetime.datetime.fromtimestamp(x).strftime('%Y/%m/%d %H:%M:%S'))
        df = df.sort_values("clock")

        self.X, self.y = create_sequences(df["value_avg"], sequence_length)
        self.MIN = self.X.min()
        self.MAX = self.X.max()
        self.X = MinMaxScale(self.X, self.MIN, self.MAX)
        self.y = MinMaxScale(self.y, self.MIN, self.MAX)

        split_index = int(len(self.X) * 0.9)
        if training:
            self.X = self.X[:split_index]
            self.y = self.y[:split_index]
        else:
            self.X = self.X[split_index:]
            self.y = self.y[split_index:]


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    def get_min_max_values(self):
        return self.MIN, self.MAX 
    
def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data)-seq_length):
        x = data.iloc[i:(i+seq_length)]
        y = data.iloc[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def MinMaxScale(array, min, max):
    return (array - min) / (max - min)