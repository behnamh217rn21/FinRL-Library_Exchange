import os
import numpy as np

import warnings
warnings.filterwarnings('ignore')

from finrl.config import config
from finrl.marketdata.yahoodownloader import YahooDownloader
from finrl.preprocessing.preprocessors import FeatureEngineer
from finrl.preprocessing.data import data_split, load_dataset
from finrl.env.StockTradingEnvStopLoss import StockTradingEnvStopLoss
from finrl.model.models import DRLAgent

from finrl.marketdata import MT4_Data_Downloader

import multiprocessing

#############################################################################
# Append path for main project folder
import sys
sys.path.append("..\\FinRL-Library_Master")


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
    #df = YahooDownloader(start_date=config.START_DATE, end_date=config.END_DATE, ticker_list=ticker, interval_period="30m").fetch_data()   
    with open(".\\" + config.DATA_SAVE_DIR + "\\symbols.txt", "r") as file:
        _symbols = eval(file.readline())
    _symbols_i1 = []
    for i in range(0, len(_symbols)):
        _symbols_i1.append(_symbols[i][1])
    #Data_Downloader(_symbols_i1)
    #df = load_dataset(file_name="mt4_dataset.csv")
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
    print(processed.head())
    processed.to_csv(".\\" + config.DATA_SAVE_DIR + "\\Dataframe\\data_df.csv")
        
    print("****Training & Trading data split****")
    # Training data split
    train = data_split(processed, config.START_DATE, config.END_DATE)
    print("train dataset length: {}".format(str(len(train))))

    print("****Environment Document****")
    print(StockTradingEnvStopLoss.__doc__)
    
    print("****Build Train Environment****")
    file = open(".\\" + config.DATA_SAVE_DIR + "\\balance.txt","r+") 
    initial_amount = file.read()
    file.close()
    information_cols = ["close", "macd", "boll_ub", "boll_lb", "rsi_30", "cci_30", "dx_30", "close_30_sma", "close_60_sma", "log_volume", "change", "daily_variance"]
    env_trade_kwargs = {'initial_amount': initial_amount*500,
                        'hmax': 100, 
                        'cache_indicator_data': True,
                        'daily_information_cols': information_cols,
                        'print_verbosity': 500, 
                        'discrete_actions': True}
    e_train_gym = StockTradingEnvStopLoss(df = train, **env_train_kwargs)
    # for this example, let's do multiprocessing with n_cores-2
    n_cores = multiprocessing.cpu_count() - 2
    print(f"using {n_cores} cores")  
    # this is our training env. It allows multiprocessing
    env_train, _ = e_train_gym.get_multiproc_env(n = n_cores)
    #env_train, _ = e_train_gym.get_sb_env()
    
    print("****Implement DRL Algorithms****")
    agent = DRLAgent(env=env_train)
    ddpg_params ={"actor_lr": 5e-06,
                  "critic_lr": 5e-06,
                  "gamma": 0.99,
                  "batch_size": 1024,
                  "eval_env": env_trade}  
    
    policy_kwargs = {"net_arch": ["lstm", "lstm", dict(pi=[dict(lstm_L1=24, dropout_L2=0.2, lstm_L3=24, dropout_L4=0.2)], \
                                                       vf=[dict(dense_L1=64, dense_L2=16)])],
                     "n_lstm": 10}
    
    model = agent.get_model("ddpg",
                            policy="LstmLstmPolicy",
                            model_kwargs = ddpg_params,
                            policy_kwargs = policy_kwargs,   
                            verbose = 0)
        
    print("****Train_Model****")
    trained_ddpg = agent.train_model(model=model, 
                                     total_timesteps=32600000,
                                     log_interval=1)
    
    print("****Model Saving****")
    trained_ddpg.save(".\\" + config.TRAINED_MODEL_DIR + "\\DDPG.model")
    
    
if __name__ == "__main__":
    main()
