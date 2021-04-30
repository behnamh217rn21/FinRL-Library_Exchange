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
from finrl.model.models import DRLAgent
from finrl.env.trade_env.env_stocktrading_stoploss_online import StockTradingEnvStopLossOnline
from finrl.env.dwx_zeromq_connector.strategies import rates_subscriptions_v1

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
    def rates_subscriptions(_symbols):
        # creates object with a predefined configuration
        print('running rates subscriptions process ...')
        func = rates_subscriptions_v1.rates_subscriptions(_instruments=_symbols)
        func.run()
        # Waits example termination
        print('Waiting rates subscriptions process termination...\n')
        while not func.isFinished():
            sleep(1)
        
    print("==============Start Training===========")
    print("****Start Fetching Data (rates subscriptions process)****")
    with open("./" + config.DATA_SAVE_DIR + "/symbols.txt", "r") as file:
        _symbols = eval(file.readline())
    process = multiprocessing.Process(target=rates_subscriptions, args=(_symbols))
    process.start()
    sleep(60)
    
    print("****Environment Document****")
    print(StockTradingEnvStopLossOnline.__doc__)
    
    print("****Build Train Environment****")
    file = open("./" + config.DATA_SAVE_DIR + "/balance.txt","r+") 
    initial_amount = file.read()
    initial_amount = float(initial_amount)
    file.close()
    information_cols = ["close", "macd", "boll_ub", "boll_lb", "rsi_30", "cci_30", "dx_30", \
                        "close_30_sma", "close_60_sma", "log_volume", "change", "daily_variance"]
    env_train_kwargs = {'initial_amount': initial_amount*1000,
                        'assets': _symbols_i1,
                        'sell_cost_pct': 0,
                        'buy_cost_pct': 0,
                        'hmax': 10,
                        'cash_penalty_proportion': 0.2,
                        'daily_information_cols': information_cols, 
                        'print_verbosity': 500, 
                        'discrete_actions': True}
    e_train_gym = StockTradingEnvStopLossOnline(df = train_df, **env_train_kwargs)
    # this is our training env. It allows multiprocessing
    env_train, _ = e_train_gym.get_sb_env()

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
    

if __name__ == "__main__":
    main()
