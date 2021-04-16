import os
import numpy as np

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
sys.path.append('../../..')
sys.path.append("./FinRL-Library_Master")


#############################################################################
#############################################################################
def Data_Downloader(_symbols):
    # creates object with a predefined configuration
    print('\nrunning rates_historic process ...')
    func = MT4_Data_Downloader.rates_historic(_instruments=_symbols)
    func.run()
    # Waits example termination
    print('\nWaiting rates_historic process termination...\n')
    while not func.isFinished():
        sleep(1)
        
def main():
    """
    train an agent
    """
    print("==============Start Fetching Data===========")
    #df = YahooDownloader(start_date=config.START_DATE, end_date=config.END_DATE, ticker_list=ticker, interval_period="30m").fetch_data()   
    with open("./" + config.DATA_SAVE_DIR + "/symbols.txt", "r") as file:
        _symbols = eval(file.readline())
    _symbols_list = []
    for i in range(0, len(_symbols)):
        _symbols_list.append(_symbols[i][1])
    Data_Downloader(_symbols_list)
    df = load_dataset(file_name="mt4_dataset.csv")
    print(df.head())

    print("==============Start Feature Engineering===========")
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=config.TECHNICAL_INDICATORS_LIST,
        use_turbulence=True,
        user_defined_feature=False,
    )
    processed = fe.preprocess_data(df)
    processed['change'] = (processed.close-processed.open)/processed.close
    processed['log_volume'] = np.log(processed.volume*processed.close)
    processed['daily_variance'] = (processed.high-processed.low)/processed.close
    processed.to_csv("./" + config.DATA_SAVE_DIR + "/Dataframe/data_df.csv")
        
    # Training data split
    train = data_split(processed, config.START_DATE, config.START_TRADE_DATE)

    print(StockTradingEnvStopLoss.__doc__)
    information_cols = ["daily_variance", "change", "log_volume", "close", 
                        "macd", "boll_ub", "boll_lb", "rsi_30", "cci_30", "dx_30", "close_30_sma", "close_60_sma"]
    env_train_kwargs = {
        'initial_amount': 10000,
        'hmax': 100, 
        'cache_indicator_data': True,
        'cash_penalty_proportion': 0.2, 
        'daily_information_cols': information_cols, 
        'print_verbosity': 500, 
        'random_start': True ,
        'discrete_actions': True
    }
    e_train_gym = StockTradingEnvStopLoss(df = train, **env_train_kwargs)
    
    # for this example, let's do multiprocessing with n_cores-2
    n_cores = multiprocessing.cpu_count() - 2
    print(f"using {n_cores} cores")  
    # this is our training env. It allows multiprocessing
    #env_train, _ = e_train_gym.get_multiproc_env(n = n_cores)
    env_train, _ = e_train_gym.get_sb_env()

    agent = DRLAgent(env=env_train)
    print("==============Model Training===========")
    ddpg_params ={'actor_lr': 5e-06,
                  'critic_lr': 5e-06,
                  'gamma': 0.99,
                  'batch_size': 1024,
                  'eval_env': env_trade,
                  'nb_eval_steps': 50
                 }
    
    policy_kwargs = {
        "net_arch": ["lstm", "lstm", dict(pi=[dict(lstm_L1=24, dropout_L2=0.2, lstm_L3=24, dropout_L4=0.2)], \
                                          vf=[dict(dense_L1=64, dense_L2=16)])],
        "n_lstm": 10
    }
    
    model_name = "ddpg"
    model = agent.get_model(model_name,
                            policy="LstmLstmPolicy",
                            model_kwargs = ddpg_params,
                            policy_kwargs = policy_kwargs,
                            verbose = 0
                           )
        
    trained_ddpg = agent.train_model(model=model, 
                                     tb_log_name="ddpg", 
                                     total_timesteps=80000,
                                     log_interval=1
                                    )
    model.save("./" + config.TRAINED_MODEL_DIR + "/DDPG.model")
    
    
if __name__ == "__main__":
    main()
    
