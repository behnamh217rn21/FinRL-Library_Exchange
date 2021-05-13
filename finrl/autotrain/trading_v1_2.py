import pandas as pd
import datetime

#############################################################################
import warnings
warnings.filterwarnings('ignore')

#############################################################################
from finrl.config import config
from finrl.model.models import DRLAgent
from finrl.trade.backtest import backtest_plot, backtest_stats

from finrl.env.trade_env.env_stocktrading_stoploss_online import StockTradingEnvStopLossOnline

#############################################################################
# Append path for main project folder
import sys
sys.path.append("..\\FinRL-Library_Master")


#############################################################################
#############################################################################                 
def main():
    """
    agent trading
    """
    
    print("==============Start Trading===========")
    DDPG_model_path = "./" + config.TRAINED_MODEL_DIR + "/DDPG.model"

    #print("****Environment Document****")
    #print(StockTradingEnvStopLoss_online.__doc__)

    print("****Build Trade Environment****")
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
    env_trade_kwargs = {'initial_amount': initial_amount*float(Leverage),
                        'sell_cost_pct': 0,
                        'buy_cost_pct': 0,
                        'hmax': 0.1,
                        'cash_penalty_proportion': 0.2,
                        'daily_information_cols': information_cols, 
                        'print_verbosity': 1, 
                        'discrete_actions': False}
    e_trade_gym = StockTradingEnvStopLossOnline(**env_trade_kwargs)
    env_trade, obs_trade = e_trade_gym.get_sb_env()
    
    print("****Model Prediction****")
    agent = DRLAgent(env=env_trade)
    ddpg_params = {"actor_lr": 5e-06,
                   "critic_lr": 5e-06,
                   "gamma": 0.99,
                   "batch_size": 1024,
                   "eval_env": env_trade}

    model = agent.get_model("ddpg",
                                 model_kwargs = ddpg_params,
                                 verbose = 0)
    
    DDPG_model = model.load(DDPG_model_path)
    df_account_value, df_actions = DRLAgent.DRL_prediction_online(model=DDPG_model,
                                                                  obs=obs_trade,
                                                                  environment=env_trade,
                                                                  n_hrs=14)
    
    #df_account_value, df_actions = DRLAgent.DRL_prediction_online(model=DDPG_model,
                                                                   #obs=obs_trade,
                                                                   #environment=env_trade)
    
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
