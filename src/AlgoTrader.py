from datetime import datetime, time, timedelta
from lumibot.strategies import Strategy
from lumibot.traders import Trader
from lumibot.brokers import Alpaca
from lumibot.backtesting import YahooDataBacktesting
from torch import nn
from alpaca_trade_api import REST
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from lumibot.entities import Asset
import os
import math
import polars as pl
import yfinance as yf
import logging
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
TESTING = True

logger = logging.getLogger('yfinance')
logger.disabled = True
logger.propagate = False
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
          h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to('cuda')
          out, (hn) = self.gru(x, (h0.detach()))
          out = self.fc(out[:, -1, :])
          return out
class e(Exception):
    def __init__(self, *args):
        self.message = args
        super().__init__(self.message)
BUDGET = 1000
API_KEY = os.environ.get("API_KEY", "")
API_SECRET = os.environ.get("API_SECRET", "")
BASE_URL = "https://paper-api.alpaca.markets"
ALPACA_CREDS = {
    "API_KEY": API_KEY,
    "API_SECRET": API_SECRET,
    "PAPER": True,
}
start_date = datetime(2024,1,1)
end_date = datetime(2024,6,21)

if TESTING:
    stockPriceValues = pd.read_feather("./FULLTESTER.feather")
    openPrices = pd.read_feather('./FULLDATASETFINALOPEN.feather')
model = GRU(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers).to("cuda")
model.load_state_dict(torch.load(f'./FULL2.pt'))
class MLPTrader(Strategy):
    def initialize(self, cash_at_risk: float = 0.5):
        self.sleeptime = "1D"
        self.last_trade = None
        self.cash_at_risk = cash_at_risk
        self.api = REST(base_url=BASE_URL, key_id=API_KEY, secret_key=API_SECRET)
    # def getStockPrices(self, stock, stockPrices):
    #     try:
    #         reshapedRows = stockPrices[stock].values.reshape(-1,1)
    #         scaler = MinMaxScaler(feature_range=(-1,1))
    #         data = torch.from_numpy((np.array(([scaler.fit_transform(reshapedRows)[-19:]])))).type(torch.Tensor)
    #         if stock == 'JAGX':
    #             print(data)
    #         return (data, scaler)
    #     except Exception as e:
    #         return (None, None)
    def getStockPrices(self, stock, stockPrices):
        # try:
            allValues = []
            scaler = MinMaxScaler(feature_range=(-1,1))
            reshapedRows = scaler.fit_transform(stockPrices[stock].values.reshape(-1,1))
            for index in range(len(reshapedRows)):
                 if(len(reshapedRows[index-20:index]) != 20):
                      continue
                 allValues.append(reshapedRows[index-20:index])
            arrayPrevious = np.array([allValues[-2]])
            arrayNow = np.array([allValues[-1]])
            data = torch.from_numpy(arrayPrevious).type(torch.Tensor)
            dataNow = torch.from_numpy(arrayNow).type(torch.Tensor)
            return (data, scaler, dataNow)
        # except Exception as e:
        #     print("beans")
        #     return (None, None)
    def calculateBuyOrSell(self, results, stock, lastPrice,resultNow):
        growthStandings = float(resultNow/results) * float(resultNow/lastPrice)
        if growthStandings > 3:
            return False, 1, stock, resultNow[0][0], lastPrice
        if(growthStandings < 1):
            return False, growthStandings, stock, resultNow[0][0], lastPrice
        else:
            return True, growthStandings, stock, resultNow[0][0],lastPrice
    def handleStockAnalysis(self, stockPrices, todaysData, stocks):
        data = [self.getStockPrices(stock, stockPrices) for stock in stocks]
        dataTensors = torch.concat([piece[0]for piece in data if piece[0] != None]).to('cuda')
        dataTensorsNow = torch.concat([piece[2]for piece in data if piece[0] != None]).to('cuda')
        results = model(dataTensors).to('cpu').detach().numpy()
        resultsNow = model(dataTensorsNow).to('cpu').detach().numpy()
        resultValues = [] 
        resultValuesNow = [] 
        for i, result in enumerate(results):
            resultValues.append(data[i][1].inverse_transform(result.reshape(-1,1)))
            resultValuesNow.append(data[i][1].inverse_transform(resultsNow[i].reshape(-1,1)))
        buyOrSellResults = [self.calculateBuyOrSell(result, stocks[i], todaysData[stocks[i]], resultValuesNow[i]) for i, result in enumerate(resultValues) if not np.isnan(result[0][0])]
        return buyOrSellResults
    def volatility(self,stock, stockPrices):
        df = stockPrices[stock]
        TRADING_DAYS = 252
        returns = np.log(df/df.shift(1))
        returns.fillna(0, inplace=True)
        volatility = returns.rolling(window=TRADING_DAYS, min_periods=0).std()*np.sqrt(TRADING_DAYS)
        return volatility[-1]
    def getQuantities(self, results, trueFalse, stockPrices, todaysData):
        results = pl.DataFrame({"Stock": [result[2] for result in results if result[0] == trueFalse], "Growth": [result[1] for result in results if result[0] == trueFalse], 'Expected': [result[3] for result in results if result[0] == trueFalse], 'Last': [result[4] for result in results if result[0] == trueFalse]})
        results = results.drop_nulls().sort('Growth').head(50) if not trueFalse else results.drop_nulls().sort('Growth').tail(50)
        volatilities = []
        for row in results.iter_rows():
            volatilities.append(self.volatility(row[0], stockPrices))
        results = results.with_columns(Volatilities=pl.Series(volatilities))
        results= results.sort('Volatilities').head(10)
        prices = todaysData
        cashPerStock = self.get_cash()/10
        cashUsed = self.get_cash()
        pricesList, quantities= np.array([]), np.array([])
        for row in (results.iter_rows()):
            stock = row[0]
            if math.floor(cashPerStock/prices[stock]) < 0:
                raise e(cashPerStock, prices[stock], self.get_cash(), self.get_portfolio_value())
            pricesList= np.append(pricesList, [prices[stock]])
            quantities = np.append(quantities, [math.floor(cashPerStock/prices[stock])])
            cashUsed = cashUsed - (math.floor(cashPerStock/prices[stock]) * prices[stock])
        addQuant = pricesList.min()
        addIndex = pricesList.argmin()
        addQuant = math.floor(cashUsed / addQuant)
        cashUsed = cashUsed - (addQuant * pricesList.min())
        quantities[addIndex] += addQuant
        results = (results.with_columns(Prices=pl.Series(pricesList), Quantity=pl.Series(quantities)))
        results = results.with_columns(pl.col(pl.Float32, pl.Float64).fill_nan(None)).drop_nulls()

        # results = results.with_columns(pl.col())
        # results.write_csv('./MommyIsAnnoyingV2.csv')
        print(results, self.get_datetime())
        for row in results.iter_rows():
            stock = row[0]
            if(row[6] <= 0):
                continue
            if prices[stock] -.1 < 0:
                continue
            if prices[stock]*.2 < .01:
                takeProfit = prices[stock] + .01 if trueFalse else max(prices[stock] -.01, 0)
            else:
                takeProfit = prices[stock]*1.20 if trueFalse else prices[stock]*.8
            if prices[stock] * .05 < .01:
                stopLoss = prices[stock] + .01 if trueFalse else max(prices[stock] -.01,0)
            else:
                stopLoss = prices[stock]*.95 if trueFalse else prices[stock]*1.05
            order = self.create_order(
                stock, 
                row[6], 
                "buy", 
                type="bracket", 
                take_profit_price=takeProfit, 
                stop_loss_price=stopLoss
            )
            self.submit_order(order)
    def on_trading_iteration(self):
        
        if not TESTING:
            stocks = self.api.list_assets(status='active')
            stocks = [stock.__dict__['_raw']['symbol'] for stock in stocks if stock.__dict__['_raw']['class'] == 'us_equity']
            data = yf.Tickers(stocks).download(start=(self.get_datetime()-timedelta(365)).strftime('%Y-%m-%d'), end=(self.get_datetime()).strftime('%Y-%m-%d'), repair=True, progress=False)['Close']
        else:
            data = stockPriceValues.loc[(stockPriceValues.index.tz_convert(None) > np.datetime64(pd.DatetimeIndex([self.get_datetime()-timedelta(days=365)])[0])) & (stockPriceValues.index.tz_convert(None) < np.datetime64(pd.DatetimeIndex([self.get_datetime()-timedelta(days=1)])[0]))]['Close']
            todaysData = openPrices.loc[(openPrices.index.tz_localize(None) < np.datetime64(pd.DatetimeIndex([self.get_datetime()])[0]))].iloc[-1]
            stocks = list(todaysData.dropna().index)
            stocks = [stock for stock in stocks if stock in list(data.columns)]
            for stock in stocks:
                if abs(data[stock][-1] - todaysData[stock]) > 1:
                    stocks.remove(stock)
        results = self.handleStockAnalysis(data, todaysData, stocks)
        if results == None:
            return
        if results == None or data.empty:
            raise Exception('Shit hit the fan')
        self.getQuantities(results, True, data, todaysData)
        # if(self.get_portfolio_value() > 2000 and TESTING):
        #     self.getQuantities(results, False, data)


# Instantiate Trader\
trader = Trader()
broker = Alpaca(ALPACA_CREDS)
strategy = MLPTrader(name="GRUTrader", broker=broker, budget=BUDGET)

# Start BackTrading
if TESTING:
# if True:
    benchmark_asset = Asset(symbol="GOOGL", asset_type="stock")

    backtesting_start = start_date
    backtesting_end = end_date
    strategy.backtest(YahooDataBacktesting, backtesting_start, backtesting_end,budget=500, benchmark_asset=benchmark_asset, parameters={
        "cash_at_risk": 0.5,
        "symbol": 'GOOGL',
    })
else:
    trader.add_strategy(strategy)
    trader.run_all()