from torch import nn
import torch
import pandas as pd
import numpy as np
import math
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

### Hyperparameters

batch_size = 32
n_stocks = 10
block_size = 20
n_eval = 200
eval_interval = 1000
n_epochs = 5000
learning_rate = 1e-6
d_model = 64
p_eval = 0.1

### Data
scaler = StandardScaler()
data = pd.read_feather('data/BigTestDataset/CLOSEDATAFINAL.feather').to_numpy()
data = scaler.fit_transform(data)
nan_counts = np.isnan(data).sum(axis=1)
invalidTimesteps = np.argwhere(nan_counts > data.shape[1] - n_stocks).squeeze()
for value in invalidTimesteps:
    invalidTimesteps = np.append(invalidTimesteps, np.arange(value-block_size+1, value))
invalidTimesteps = np.unique(invalidTimesteps)
data = np.delete(data, invalidTimesteps, axis=0)
split = int(data.shape[0]*(1-p_eval))
train_data = data[:split]
val_data = data[split:]
np.random.seed(0)
def get_batch(split='train'):
    data_split = train_data if split == 'train' else val_data
    batch_time_indices = np.random.randint(0, data_split.shape[0] - block_size - 1, size=batch_size)
    x_list = []
    y_list = []
    stocks_list = []
    for i in range(batch_size):
        time_i = batch_time_indices[i]
        data_i = data_split[time_i:time_i + block_size + 1]
        valid_stocks = np.where(~np.isnan(data_i).any(axis=0))[0]
        if len(valid_stocks) >= n_stocks:
            stocks_i = np.random.choice(valid_stocks, n_stocks, replace=False)
        else:
            stocks_i = np.random.choice(valid_stocks, n_stocks, replace=True)
        x_i = data_i[:-1, stocks_i].T
        returns = data_i[-1, stocks_i] - data_i[-2, stocks_i]
        returns = np.where(data_i[-2, stocks_i] != 0, returns / data_i[-2, stocks_i], 0)
        y_i = returns.argmax().astype("long")
        x_list.append(torch.tensor(x_i, dtype=torch.float32))
        y_list.append(torch.tensor(y_i, dtype=torch.long))
        stocks_list.append(stocks_i)
    x = torch.stack(x_list).to('cuda')
    y = torch.stack(y_list).to('cuda')
    stocks = torch.as_tensor(np.array(stocks_list), dtype=torch.long, device='cuda')
    return x, y, stocks


### Model

class PositionalEmbedding(nn.Module):
    def __init__(self, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class TokenEmbedding(nn.Module):
    def __init__(self):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=1, out_channels=d_model, 
                                    kernel_size=3, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x)
        return x.transpose(1,2)

class DataEmbedding(nn.Module):
    def __init__(self):
        super(DataEmbedding, self).__init__()
        self.token = TokenEmbedding()
        self.position = PositionalEmbedding()
        self.stockEmbedding = nn.Embedding(data.shape[1], d_model)
    def forward(self, x, stocks):
        B, S, T = x.size()
        x = x.view(B*S, 1, T)
        token = self.token(x)
        position = self.position(x)
        stocks = stocks.flatten()
        stockEmbedding = self.stockEmbedding(stocks)
        embedding = token + position + stockEmbedding.unsqueeze(1)
        return embedding.view(B*S, T, d_model)


class Informer(nn.Module):
    def __init__(self):
        super(Informer, self).__init__()
        self.dataEmbedding = DataEmbedding()
        self.lm_head = nn.Linear(d_model, 1)
    def forward(self,x,target,stocks):
        B, S, T = x.size()
        x = self.dataEmbedding(x, stocks)
        x = x[:, -1, :]
        x = self.lm_head(x)
        x= x.view(B, S)
        if target is None:
            loss = None
        else:
            loss = F.cross_entropy(x, target)
        return x, loss

model = Informer().to('cuda')

### Loss Estimation
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(n_eval)
        for k in range(n_eval):
            X, y, stocks= get_batch(split)
            logits ,loss = model(X,y, stocks)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


### Training Loop
optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)
for epoch in tqdm(range(n_epochs)):
    model.train()
    loss = model(*get_batch('train'))[1]
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if(epoch%eval_interval == 0):
        # print(f"Epoch: {epoch}, Train Loss: {loss:.4f}")
        losses = estimate_loss()
        print(f"Epoch: {epoch}, Train Loss: {losses['train']:.4f}, Val loss: {losses['val']:.4f}")

