import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")

import warnings
warnings.filterwarnings('ignore')

#############################################################################
from sklearn import preprocessing
import datetime

#############################################################################
from finrl.config import config
from finrl.marketdata.yahoodownloader import YahooDownloader
from finrl.preprocessing.preprocessors import FeatureEngineer
from finrl.preprocessing.data import data_split, load_dataset
from finrl.env.env_stocktrading_stoploss import StockTradingEnvStopLoss
from finrl.model.models import DRLAgent
from finrl.trade.backtest import backtest_plot, backtest_stats

#############################################################################
import multiprocessing

# Append path for main project folder
import sys
sys.path.append("..\\FinRL-Library_Exchange")


#############################################################################
#############################################################################
def main():
    """
    train an agent
    """ 
    print("==============Start Training===========")
    print("****Start Fetching Data****")
    df = load_dataset(file_name="data.csv")
    print(df.head())
    
    print("****Start Feature Engineering****")
    fe = FeatureEngineer(use_technical_indicator=True,
                         tech_indicator_list=config.TECHNICAL_INDICATORS_LIST,
                         use_turbulence=True,
                         user_defined_feature=False)
    processed = fe.preprocess_data(df)
    np.seterr(divide = 'ignore')
    processed['log_volume'] = np.where((processed.volume*processed.close)>0, np.log(processed.volume*processed.close), 0)
    processed['change'] = (processed.close-processed.open)/processed.close
    processed['daily_variance'] = (processed.high-processed.low)/processed.close
    processed.to_csv("./" + config.DATA_SAVE_DIR + "/Dataframe/data_df.csv")

    print("****Training & Trading data split****")
    # Training & Trading data split
    train_df = data_split(processed, config.START_DATE, config.START_TRADE_DATE)
    print("train dataset length: {}".format(str(len(train_df))))
    trade_df = data_split(processed, config.START_TRADE_DATE, config.END_DATE)
    print("trade dataset length: {}".format(str(len(trade_df))))
    
    print("****Environment Document****")
    print(StockTradingEnvStopLoss.__doc__)
    
    print("****Build Train Environment****")
    file = open("./" + config.DATA_SAVE_DIR + "/balance.txt","r+") 
    initial_amount = file.read()
    initial_amount = float(initial_amount)
    file.close()
    information_cols = ["close", "macd", "boll_ub", "boll_lb", "rsi_30", "cci_30", "dx_30", "close_30_sma", "close_60_sma", "log_volume", "change", "daily_variance"]
    env_train_kwargs = {'initial_amount': initial_amount*500,
                        'hmax': 100, 
                        'cache_indicator_data': True,
                        'daily_information_cols': information_cols,
                        'print_verbosity': 500, 
                        'discrete_actions': True}
    e_train_gym = StockTradingEnvStopLoss(df = train_df, **env_train_kwargs)
    # this is our training env. It allows multiprocessing
    env_train, _ = e_train_gym.get_sb_env()
    
    print("****Build Trade Environment****")
    env_trade_kwargs = {'initial_amount': initial_amount*500,
                        'hmax': 100,
                        'daily_information_cols': information_cols, 
                        'print_verbosity': 500, 
                        'random_start': False,
                        'discrete_actions': True}
    e_trade_gym = StockTradingEnvStopLoss(df = trade_df, **env_trade_kwargs)
    # this is our observation environment. It allows full diagnostics
    env_trade, _ = e_trade_gym.get_sb_env()

    print("****Implement DRL Algorithms****")
    agent = DRLAgent(env=env_train)
    ddpg_params = {"actor_lr": 5e-06,
                   "critic_lr": 5e-06,
                   "gamma": 0.99,
                   "batch_size": 1024,
                   "eval_env": env_trade}

    DDPG_model = agent.get_model("ddpg",
                                 model_kwargs = ddpg_params,
                                 verbose = 0)
    
    print("****Train_Model****")
    DDPG_model = agent.train_model(model=DDPG_model, 
                                   total_timesteps=32600000, 
                                   log_interval=1)
    
    print("****Model Saving****")
    DDPG_model.save("./" + config.TRAINED_MODEL_DIR + "/DDPG.model")
    
    print("==============Start Trading===========")
    print("****Model Prediction****")
    df_account_value, df_actions = DRLAgent.DRL_prediction(model=DDPG_model, 
                                                           environment = e_trade_gym)
    
    print("****Prediction Resault Saving****")
    now = datetime.datetime.now().strftime("%Y-%m-%d-%HH%MM")
    df_account_value.to_csv("./" + config.RESULTS_DIR + "/_df_account_value" + now + ".csv")
    df_actions.to_csv("./" + config.RESULTS_DIR + "/_df_actions" + now + ".csv")
    
    print("****Get Backtest Results****")
    perf_stats_all = backtest_stats(account_value=df_account_value, value_col_name = 'total_assets')
    perf_stats_all = pd.DataFrame(perf_stats_all)
    perf_stats_all.to_csv("./" + config.RESULTS_DIR + "/_perf_stats_all" + now + ".csv")
    

if __name__ == "__main__":
    main()
