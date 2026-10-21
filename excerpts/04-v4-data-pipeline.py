# Source: notebooks/TrainAlgoTraderGRUNormalized.ipynb · cell 24
# 20-step inputs and future targets in the recovered preprocessing code.
# Viewing excerpt; see original notebook for context.

    data = []
    y = []
    scalar = MinMaxScaler(feature_range=(-1,1))
    for index in range(len(dataframe[0])-21):
        subData = []
        subData.append(scalar.fit_transform(dataframe[-1][index:index+20].reshape(-1,1)))
        y.append(scalar.transform(dataframe[-1][index+21].reshape(-1,1)))
        for listOfData in dataframe[[0,1,3,4,5,7,8,9,12]]:
