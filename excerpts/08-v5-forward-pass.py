# Source: notebooks/TrainStrengthArchitecture.ipynb · cell 38
# Convolution blocks feed a bidirectional GRU and dense output layers.
# Viewing excerpt; see original notebook for context.

        x = self.tcn1(x)
        x = self.tcn2(x)
        x = self.tcn3(x)
        x = x.permute(0, 2, 1)  # (batch_size, seq_len, tcn_channels)
        x, _ = self.lstm(x)  # (batch_size, seq_len, 2 * lstm_hidden_dim)
        
        # x = self.transformer(x)  # (batch_size, seq_len, 2 * lstm_hidden_dim)
        
        x = x[:, -1, :]  # (batch_size, 2 * lstm_hidden_dim)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        output = self.fc3(x)
        return output

input_dim = 14 
tcn_channels = 64
