# Source: notebooks/TrainAlgoTraderGRUNormalized.ipynb · cell 87
# Recovered baseline architecture; exact V3 checkpoint not verified.
# Viewing excerpt; see original notebook for context.

input_dim = 1
hidden_dim = 32
num_layers = 2
output_dim = 1
num_epochs = 100
class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(GRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn) = self.gru(x, (h0.detach()))
        out = self.fc(out[:, -1, :])
        return out
