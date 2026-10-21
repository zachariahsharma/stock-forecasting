# Source: notebooks/TrainStrengthArchitecture.ipynb · cell 38
# Recovered TCN/BiGRU variant; BatchNorm + ReLU in each convolution block.
# Viewing excerpt; see original notebook for context.

class BeefModel(nn.Module):
    def __init__(self, input_dim, tcn_channels, kernel_size, lstm_hidden_dim, transformer_heads, transformer_hidden_dim, output_dim):
        super(BeefModel, self).__init__()
        self.tcn1 = nn.Sequential(
            nn.Conv1d(input_dim, tcn_channels, kernel_size, dilation=1, padding=1 * (kernel_size - 1) // 2),
            nn.BatchNorm1d(tcn_channels),
            nn.ReLU()
        )
        self.tcn2 = nn.Sequential(
            nn.Conv1d(tcn_channels, tcn_channels, kernel_size, dilation=2, padding=2 * (kernel_size - 1) // 2),
            nn.BatchNorm1d(tcn_channels),
            nn.ReLU()
        )
        self.tcn3 = nn.Sequential(
            nn.Conv1d(tcn_channels, tcn_channels, kernel_size, dilation=4, padding=4 * (kernel_size - 1) // 2),
            nn.BatchNorm1d(tcn_channels),
            nn.ReLU()
        )
        self.lstm = nn.GRU(tcn_channels, lstm_hidden_dim, batch_first=True, bidirectional=True)
        # self.transformer_layer = nn.TransformerEncoderLayer(d_model=2 * lstm_hidden_dim, nhead=transformer_heads, dim_feedforward=transformer_hidden_dim)
        # self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=1)
        self.fc1 = nn.Linear(2 * lstm_hidden_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_dim)
        self.dropout = nn.Dropout(0.5)
