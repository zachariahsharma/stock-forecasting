# Source: src/CustomArchitecture.py
# Separate archived transformer experiment; experimental implementation.
# Viewing excerpt; see original notebook for context.

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
