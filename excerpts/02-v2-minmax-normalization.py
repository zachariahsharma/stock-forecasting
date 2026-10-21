# Source: notebooks/TrainAlgoTraderGRUNormalized.ipynb · cell 29
# Recovered GRU preprocessing example; original V2 MLP not recovered.
# Viewing excerpt; see original notebook for context.

def add_feature_engineering(df, stock):
    price = df.reshape(-1,1)
    scaler = MinMaxScaler(feature_range=(-1,1))
    def split_data(stock, lookback):
        data = []
        for index in range(len(stock) - lookback):
            data.append(scaler.fit_transform(stock[index: index + lookback]))
        data = np.array(data)
        test_set_size = int(np.round(0.2*data.shape[0]))
        train_set_size = data.shape[0] - (test_set_size)
        x_train = data[:train_set_size,:-1,:]
        y_train = data[:train_set_size,-1,:]
        x_test = data[train_set_size:,:-1]
        y_test = data[train_set_size:,-1,:]
        return [x_train, y_train, x_test, y_test]
    lookback = 20 # choose sequence length
    x_train, y_train, x_test, y_test = split_data(price, lookback)
    return x_train, y_train, x_test, y_test
