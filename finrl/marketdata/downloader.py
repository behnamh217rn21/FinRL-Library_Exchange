"""Contains methods and classes to collect data from drive
"""
import pandas as pd
from finrl.config import config


class Downloader:
    """Provides methods for retrieving daily stock data from drive
    Methods
    -------
    fetch_data()
        Fetches data from drive
    """
    def __init__(self):
        self.data_df = pd.DataFrame()


    def fetch_data(self) -> pd.DataFrame:
        """Fetches data from drive
        Parameters
        ----------
        Returns
        -------
        `pd.DataFrame`
            7 columns: A date, open, high, low, close, volume and tick symbol
            for the specified stock ticker
        """
        # Download and save the data in a pandas DataFrame:
        self.data_df = pd.DataFrame()
        for tic in self.ticker_list:
            df_name =  "datasets/data/{}_Candlestick_1_Hour_BID_05.05.2003-13.02.2021.csv".format(tic)
            temp_df = pd.read_csv(df_name, sep=',', low_memory=False)
            temp_df['tic'] = tic
            self.data_df = self.data_df.append(temp_df)

        # reset the index, we want to use numbers as index instead of dates
        self.data_df = self.data_df.reset_index()

        try:
            # convert the column names to standardized names
        self.data_df.columns = [
                                "date",
                                "open",
                                "high",
                                "low",
                                "close",
                                "volume",
                                "tic",
                               ]

            if self.ticker_list in (config.FX_TICKER):
                # use adjusted close price instead of close price
                self.data_df["close"] =  self.data_df["adjcp"]
                # drop the adjusted close price column
                self.data_df =  self.data_df.drop("adjcp", 1)

        except NotImplementedError:
            print("the features are not supported currently")

        data_df['date'] = pd.to_datetime(data_df.date, format='%d.%m.%Y %H:%M:%S.%f')
        # create day of the week column (monday = 0)
        data_df["day"] = data_df["date"].dt.dayofweek

        for i in range(0, len(data_df)):
            data_df.loc[i,"D_H_Order"] = str(data_df.loc[i,"day"]) + str(data_df["Gmt_time"][i]).split(' ')[1].split(':')[0][1]

        # convert date to standard string format, easy to filter
        data_df["date"] = data_df.date.apply(lambda x: x.strftime("%d.%m.%Y %H:%M:%S.%f"))

        # drop missing data
        data_df = data_df.dropna()
        data_df = data_df.reset_index(drop=True)
        print("Shape of DataFrame: ", data_df.shape)
        # print("Display DataFrame: ", data_df.head())

        data_df = data_df.sort_values(by=['D_H_Order','tic']).reset_index(drop=True)

        return data_df


    def select_equal_rows_stock(self, df):
        df_check = df.tic.value_counts()
        df_check = pd.DataFrame(df_check).reset_index()
        df_check.columns = ["tic", "counts"]
        mean_df = df_check.counts.mean()
        equal_list = list(df.tic.value_counts() >= mean_df)
        names = df.tic.value_counts().index
        select_stocks_list = list(names[equal_list])
        df = df[df.tic.isin(select_stocks_list)]
        return df
