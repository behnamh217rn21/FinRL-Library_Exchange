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
from finrl.env.env_stocktrading_stoploss import StockTradingEnvStopLoss
from finrl.model.models import DRLAgent

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
    print("****Start Fetching Data (rates subscriptions process)****")
    with open("./" + config.DATA_SAVE_DIR + "/symbols.txt", "r") as file:
        _symbols = eval(file.readline())
    process = multiprocessing.Process(target=rates_subscriptions(), args=(_symbols,))
    process.start()
    sleep(60)
    
    print("****Environment Document****")
    print(StockTradingEnvStopLoss.__doc__)
    
    print("****Build Train Environment****")
    file = open("./" + config.DATA_SAVE_DIR + "/balance.txt","r+") 
    initial_amount = file.read()
    initial_amount = float(initial_amount)
    file.close()
    information_cols = ["close", "macd", "boll_ub", "boll_lb", "rsi_30", "cci_30", "dx_30", \
                        "close_30_sma", "close_60_sma", "log_volume", "change", "daily_variance"]
    env_train_kwargs = {'initial_amount': initial_amount*1000,
                        'sell_cost_pct': 0,
                        'buy_cost_pct': 0,
                        'hmax': 10,
                        'cash_penalty_proportion': 0.2,
                        'cache_indicator_data': True,
                        'daily_information_cols': information_cols,
                        'print_verbosity': 500, 
                        'discrete_actions': True}
    e_train_gym = StockTradingEnvStopLoss(df = train_df, **env_train_kwargs)
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