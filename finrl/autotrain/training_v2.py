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

#############################################################################
import multiprocessing
from time import sleep

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
    #print("****Environment Document****")
    #print(StockTradingEnvStopLossOnline.__doc__)
    
    print("****Build Train Environment****")
    information_cols = ["close", "upper_band", "lower_band", "ema", "macd_signal", "macd_hist", "cci", "atr", "rsi", "adx"]
    env_train_kwargs = {'sell_cost_pct': 0,
                        'buy_cost_pct': 0,
                        'hmax': 1,
                        'cash_penalty_proportion': 0.2,
                        'daily_information_cols': information_cols, 
                        'print_verbosity': 1, 
                        'discrete_actions': False,
                        'patient': True}
    e_train_gym = StockTradingEnvStopLossOnline(**env_train_kwargs)
    # this is our training env.
    env_train, _ = e_train_gym.get_sb_env()

    print("****Implement DRL Algorithms****")
    agent = DRLAgent(env=env_train)
    ddpg_params = {"actor_lr": 5e-06,
                   "critic_lr": 5e-06,
                   "gamma": 0.99,
                   "batch_size": 64}

    DDPG_model = agent.get_model("ddpg",
                                 model_kwargs = ddpg_params,
                                 verbose = 0)
    
    print("****Train_Model****")
    DDPG_model = agent.train_model(model=DDPG_model, 
                                   total_timesteps=100000, 
                                   log_interval=1)
    
    print("****Model Saving****")
    DDPG_model.save("./" + config.TRAINED_MODEL_DIR + "/DDPG.model")
    

if __name__ == "__main__":
    main()
