import pandas as pd
import matplotlib
matplotlib.use("Agg")
import datetime

#############################################################################
from finrl.config import config
from finrl.model.models import DRLAgent
from finrl.trade.backtest import backtest_plot, backtest_stats

from finrl.env.env_stocktrading_stoploss_online import StockTradingEnvStopLoss_online
from finrl.env.dwx_zeromq_connector.strategies import rates_subscriptions_v1

#############################################################################
#############################################################################
import multiprocessing
from time import sleep

#############################################################################
# Append path for main project folder
import sys
sys.path.append('../../..')
sys.path.append("./FinRL-Library_Master")


#############################################################################
#############################################################################
def rates_subscriptions(_symbols):
    # creates object with a predefined configuration
    print('\nrunning rates subscriptions process ...')
    func = rates_subscriptions_v1.rates_subscriptions(_instruments=_symbols)
    func.run()
    # Waits example termination
    print('\nWaiting rates subscriptions process termination...\n')
    while not func.isFinished():
        sleep(1)
                  
def main():
    trained_ddpg = model.load("./" + config.TRAINED_MODEL_DIR + "/DDPG.model")
    print(StockTradingEnvStopLoss_online.__doc__)
    
    with open("./" + config.DATA_SAVE_DIR + "/symbols.txt", "r") as file:
        _symbols = eval(file.readline())
    process = multiprocessing.Process(target=rates_subscriptions(), args=(_symbols,))
    process.start()
    sleep(60)

    file = open("./" + config.DATA_SAVE_DIR + "/balance.txt","r+") 
    initial_amount = file.read()
    file.close()
        
    env_trade_kwargs = {
        initial_amount = initial_amount,
        hmax = 5000, 
        turbulence_threshold = None, 
        currency='$',
        buy_cost_pct=3e-3,
        sell_cost_pct=3e-3,
        cash_penalty_proportion=0.2,
        cache_indicator_data=True,
        daily_information_cols = information_cols, 
        print_verbosity = 500, 
        random_start = True
    }
    e_trade_gym = StockTradingEnvStopLoss_online(turbulence_threshold=250, **env_kwargs)
    
    # this is our observation environment. It allows full diagnostics
    env_trade, _ = e_train_gym.get_sb_env()
    
    print("==============Start Trading===========")
    df_account_value, df_actions = DRLAgent.DRL_prediction(model=trained_ddpg, 
                                                           environment = e_trade_gym
                                                          )
    
    now = datetime.datetime.now().strftime("%Y%m%d-%Hh%M")
    df_account_value.to_csv("./" + config.RESULTS_DIR + "/_df_account_value" + now + ".csv")
    df_actions.to_csv("./" + config.RESULTS_DIR + "/_df_actions" + now + ".csv")
    
    print("==============Get Backtest Results===========")
    perf_stats_all = backtest_stats(account_value=df_account_value, value_col_name = 'total_assets')
    perf_stats_all = pd.DataFrame(perf_stats_all)
    perf_stats_all.to_csv("./" + config.RESULTS_DIR + "/_perf_stats_all" + now + ".csv")

    
if __name__ == "__main__":
    main()
    
