import pandas as pd
import numpy as np

#############################################################################
import warnings
warnings.filterwarnings('ignore')
import datetime

#############################################################################
from finrl.config import config
from finrl.marketdata.yahoodownloader import YahooDownloader
from finrl.preprocessing.preprocessors import FeatureEngineer
from finrl.preprocessing.data import data_split, load_dataset
from finrl.env.env_stocktrading_stoploss import StockTradingEnvStopLoss
from finrl.model.models import DRLAgent
from finrl.trade.backtest import backtest_plot, backtest_stats

from finrl.marketdata import MT4_Data_Downloader

#############################################################################
import multiprocessing

# Append path for main project folder
import sys
sys.path.append("..\\FinRL-Library_Exchange")


#############################################################################
#############################################################################
def Data_Downloader(_symbols):
    # creates object with a predefined configuration
    print('running rates_historic process ...')
    func = MT4_Data_Downloader.rates_historic(_instruments=_symbols)
    func.run()
    # Waits example termination
    print('Waiting rates_historic process termination...\n')
    while not func.isFinished():
        sleep(1)
        
def main():
    """
    train an agent
    """
    print("==============Start Training===========")
    print("****Start Fetching Data****")
    #df = YahooDownloader(start_date=config.START_DATE, end_date=config.END_DATE, \
                          #ticker_list=config.CUSTOM_US_TICKER, interval_period="1h").fetch_data()
    """
    with open("./" + config.DATA_SAVE_DIR + "/symbols.txt", "r") as file:
        _symbols = eval(file.readline())
    _symbols_i1 = []
    for i in range(0, len(_symbols)):
        _symbols_i1.append(_symbols[i][1])
    #Data_Downloader(_symbols_i1)
    #df = load_dataset(file_name="mt4_dataset.csv")
    """
    """
    df = load_dataset(file_name="data.csv")
    print(df.head())
     
    print("****Start Feature Engineering****")
    fe = FeatureEngineer(use_technical_indicator=True,
                         tech_indicator_list=config.TECHNICAL_INDICATORS_LIST,
                         use_turbulence=True,
                         user_defined_feature=False)
    processed = fe.preprocess_data(df)
    np.seterr(divide = 'ignore')
    processed['log_volume'] = np.where((processed.volume * processed.close) > 0, \
                                       np.log(processed.volume * processed.close), 0)
    processed['change'] = (processed.close - processed.open) / processed.close
    processed['daily_variance'] = (processed.high - processed.low) / processed.close
    print(processed.head())
    processed.to_csv("./" + config.DATASET_DIR + "data.csv")
    """
    processed = load_dataset(file_name="data.csv")
    print(processed.head())
    
    print("****Training & Trading data split****")
    # Training & Trading data split
    train_df = data_split(processed, config.S_SPLIT, config.T_SPLIT)
    print("train dataset length: {}".format(str(len(train_df))))
    trade_df = data_split(processed, config.T_SPLIT, config.F_SPLIT)
    print("trade dataset length: {}".format(str(len(trade_df))))
    
    #print("****Environment Document****")
    #print(StockTradingEnvStopLoss.__doc__)
    
    print("****Build Train Environment****")
    file = open("./" + config.DATA_SAVE_DIR + "/balance.txt","r+") 
    initial_amount = file.read()
    initial_amount = float(initial_amount)
    file.close()
    
    information_cols = ["close", "macd", "boll_ub", "boll_lb", "rsi_30", "cci_30", "dx_30", \
                        "close_30_sma", "close_60_sma", "log_volume", "change", "daily_variance"]

    from pathlib import Path
    path = Path(__file__).resolve().parents[4].joinpath("AppData/Roaming/MetaQuotes/Terminal/2E8DC23981084565FA3E19C061F586B2/" \
                                                        "MQL4/Files/Leverage.txt")
    with open(path, 'r') as reader:
        Leverage = reader.read()
    print("Leverage : {}".format(Leverage))
    env_train_kwargs = {'initial_amount': initial_amount*float(Leverage),
                        'sell_cost_pct': 0,
                        'buy_cost_pct': 0,
                        'hmax': 10,
                        'cash_penalty_proportion': 0.2,
                        'daily_information_cols': information_cols,
                        'print_verbosity': 500, 
                        'discrete_actions': False}
    e_train_gym = StockTradingEnvStopLoss(df = train_df, **env_train_kwargs)
    # this is our training env. It allows multiprocessing
    env_train, _ = e_train_gym.get_sb_env()
    
    print("****Build Trade Environment****")
    env_trade_kwargs = {'initial_amount': initial_amount*float(Leverage),
                        'sell_cost_pct': 0,
                        'buy_cost_pct': 0,
                        'hmax': 10,
                        'cash_penalty_proportion': 0.2,
                        'daily_information_cols': information_cols, 
                        'print_verbosity': 500, 
                        'random_start': False,
                        'discrete_actions': False}
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
                                   total_timesteps=500000, 
                                   log_interval=1)
    
    print("****Model Saving****")
    DDPG_model.save("./" + config.TRAINED_MODEL_DIR + "/DDPG.model")
    
    print("==============Start Trading===========")
    print("****Model Prediction****")
    df_account_value, df_actions = DRLAgent.DRL_prediction(model=DDPG_model, 
                                                           environment = e_trade_gym)
    
    print("****Prediction Resault Saving****")
    now = datetime.datetime.now().strftime("%Y-%m-%d-%HH%MM")
    df_account_value.to_csv("./" + config.RESULTS_DIR + "/_df_account_value_" + now + ".csv")
    df_actions.to_csv("./" + config.RESULTS_DIR + "/_df_actions_" + now + ".csv")
    
    print("****Get Backtest Results****")
    perf_stats_all = backtest_stats(account_value=df_account_value, value_col_name = 'total_assets')
    perf_stats_all = pd.DataFrame(perf_stats_all)
    perf_stats_all.to_csv("./" + config.RESULTS_DIR + "/_perf_stats_all_" + now + ".csv")
    

if __name__ == "__main__":
    main()
