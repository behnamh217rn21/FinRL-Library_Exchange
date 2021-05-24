from twelvedata import TDClient
from time import sleep
import pandas as pd
from pandas import to_datetime

# Initialize client - apikey parameter is requiered
td = TDClient(apikey="772d7db2959540db9975d00276a8f0b2")

with open("symbols.txt", "r") as file:
    _symbols = eval(file.readline())
_symbols_i1 = []
for i in range(0, len(_symbols)):
    _symbols_i1.append(_symbols[i][1])

df = pd.DataFrame()
for symbol in(_symbols_i1):
    _tic = "{}/{}".format(symbol[0:3], symbol[3:6])
    # Construct the necessary time serie
    ts = td.time_series(symbol=_tic,
    	                interval="1h",
    	                outputsize=5000)
    print(symbol)
    # Returns pandas.DataFrame
    df_t = ts.with_bbands().with_ema().with_macd().with_cci().with_rsi().with_adx()
    sleep(60)
    df_t = df_t.with_stoch(slow_k_period="3",fast_k_period="5").as_pandas()
    df_t = df_t.round(decimals=5)
    df_t["tic"] = symbol
    df = df.append(df_t)

df.drop(['macd','middle_band'], axis=1, inplace=True)
df.reset_index(inplace=True)
df["datetime"] = pd.to_datetime(df["datetime"])
df = df.sort_values(by=['datetime','tic']).reset_index(drop=True)

df.to_csv("data.csv")
print(df)

"""
_data = pd.read_csv("data.csv", sep=',', low_memory=False, index_col=[0])
_data.reset_index(drop=True, inplace=True)
_data = _data.round(decimals=5)
_data.to_csv("data.csv")
"""
