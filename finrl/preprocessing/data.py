from __future__ import division, absolute_import, print_function
import numpy as np
import pandas as pd
import datetime


def load_dataset(*, file_name: str) -> pd.DataFrame:
    """
    load csv dataset from path
    :return: (df) pandas dataframe
    """
    # _data = pd.read_csv(f"{config.DATASET_DIR}/{file_name}")
    _data = pd.read_csv("./" + config.DATASET_DIR + "/mt4_dataset.csv", sep=',', low_memory=False, index_col=[0])
   
    for i in range(0, len(df)):
        _data.loc[i, "close"] = adjusted_prices(_data.loc[i, "tic"], _data.loc[i, "close"])
    return _data

def data_split(df, start, end):
    """
    split the dataset into training or testing using date
    :param data: (df) pandas dataframe, start, end
    :return: (df) pandas dataframe
    """
    start_date = datetime.datetime.strptime(start, "%Y-%m-%d %H:%M:00")
    end_date = datetime.datetime.strptime(end, "%Y-%m-%d %H:%M:00")
    df["date"] = pd.to_datetime(df["date"])

    data = df[(df.date >= start_date) & (df.date < end_date)]
    data = data.sort_values(["date", "tic"], ignore_index=True)
    data.index = data.date.factorize()[0]
    return data

def convert_to_datetime(time):
    time_fmt = "%Y-%m-%dT%H:%M:%S"
    if isinstance(time, str):
        return datetime.datetime.strptime(time, time_fmt)
