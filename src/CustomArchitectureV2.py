from torch import nn
import torch
import pandas as pd
import numpy as np
import math
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler, RobustScaler
from tqdm import tqdm
import warnings
from utils import find_max_non_outlier
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

### Hyperparameters

batch_size = 32
block_size = 40
n_eval = 100
eval_interval = 500
n_epochs = 5001
learning_rate = 1e-4
d_model = 64
p_eval = 0.1
eps = 1e-8
n_encoder_layers = 6

### Data

data = pd.read_feather('data/BigTestDataset/CLOSEDATAFINAL.feather').to_numpy()
active_mask = ~np.isnan(data)
data_filled = pd.DataFrame(data).ffill().to_numpy()
active_mask = active_mask[1:] & active_mask[:-1]
data_filled[np.isnan(data_filled)] = eps
log_returns = np.nan_to_num(np.diff(np.log(data_filled), axis=0), nan=0.0, posinf=0.0, neginf=0.0)
T, S = log_returns.shape
split = int(T * (1 - p_eval))
train_data, train_active_mask = log_returns[:split],active_mask[:split]
val_data, val_active_mask = log_returns[split:], active_mask[split:]
np.random.seed(1)
def get_batch(split='train'):
    if split == 'train':
        data_split = train_data  # Shape: (T_train, S)
        active_mask_split = train_active_mask  # Shape: (T_train, S)
    else:
        data_split = val_data  # Shape: (T_val, S)
        active_mask_split = val_active_mask  # Shape: (T_val, S)
    batch_x = []
    batch_y = []
    batch_stocks = []
    max_attempts = batch_size * 10
    attempts = 0
    while len(batch_x) < batch_size and attempts < max_attempts:
        stock_idx = np.random.randint(0, S)
        stock_log_returns = data_split[:, stock_idx]  # Shape: (T_split,)
        stock_active_mask = active_mask_split[:, stock_idx]  # Shape: (T_split,)
        active_indices = np.where(stock_active_mask)[0]
        if len(active_indices) < block_size + 8:
            attempts += 1
            continue 
        start_idx = np.random.choice(active_indices[:-(block_size + 7) + 1])
        input_indices = np.arange(start_idx, start_idx + block_size)
        future_indices = np.arange(start_idx + block_size, start_idx + block_size + 7)
        if not np.all(stock_active_mask[input_indices]) or not np.all(stock_active_mask[future_indices]):
            attempts += 1
            continue  # Inactive periods in window, try another selection
        x_seq = stock_log_returns[input_indices]  # Shape: (block_size,)
        price_0 = 1.0  # Arbitrary starting price
        cumulative_log_returns = np.cumsum(x_seq)
        prices = price_0 * np.exp(cumulative_log_returns)
        last_price_in_block = prices[-1]
        future_log_returns = stock_log_returns[future_indices]
        future_cumulative_log_returns = np.cumsum(future_log_returns)
        future_prices = last_price_in_block * np.exp(future_cumulative_log_returns)
        max_future_price = np.max(future_prices)
        target_log_return = np.log(max_future_price / last_price_in_block)
        batch_x.append(torch.tensor(x_seq, dtype=torch.float32))
        batch_y.append(torch.tensor(target_log_return, dtype=torch.float32))
        batch_stocks.append(stock_idx)
        attempts += 1
    if len(batch_x) == 0:
        raise ValueError("Unable to create a batch with valid data.")
    x_batch = torch.stack(batch_x).to('cuda')
    y_batch = torch.stack(batch_y).to('cuda')
    stock_batch = torch.tensor(batch_stocks).to('cuda')
    
    return x_batch, y_batch, stock_batch

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
        B, T = x.size()
        x = x.unsqueeze(dim=1)
        token = self.token(x)
        position = self.position(x)
        stocks = stocks.flatten()
        stockEmbedding = self.stockEmbedding(stocks)
        embedding = token + position + stockEmbedding.unsqueeze(1)
        return embedding.view(B, T, d_model)

class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.attention = nn.MultiheadAttention(d_model, 4)
        self.proj = nn.Linear(d_model, d_model)
        self.feedForward = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model)
        )
        self.layerNorm = nn.LayerNorm(d_model)
        self.layerNorm2 = nn.LayerNorm(d_model)
    def forward(self, x):
        x = x.transpose(0,1)
        x = self.layerNorm(x)
        x = self.attention(x, x, x)[0] + x
        x = x + self.feedForward(self.layerNorm2(x))
        return x.transpose(0,1)

class Informer(nn.Module):
    def __init__(self):
        super(Informer, self).__init__()
        self.dataEmbedding = DataEmbedding()
        self.lm_head = nn.Linear(d_model, 64)
    def forward(self,x,stocks):
        x = self.dataEmbedding(x, stocks)
        x = x[:, -1, :]
        return x

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
    try:
        x_batch, y_batch, stock_batch = get_batch('train')
    except ValueError as e:
        print(f"Epoch {epoch}: {e}")
        continue
    logits, loss = model(x_batch, y_batch, stock_batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % eval_interval == 0:
        losses = estimate_loss()
        print(f"Epoch: {epoch}, Train Loss: {losses['train']:.4f}, Val Loss: {losses['val']:.4f}")

