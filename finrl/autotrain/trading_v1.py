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
sys.path.append("../FinRL-Library_Master")


#############################################################################
#############################################################################
def rates_subscriptions(_symbols):
    # creates object with a predefined configuration
    print('running rates subscriptions process ...')
    func = rates_subscriptions_v1.rates_subscriptions(_instruments=_symbols)
    func.run()
    # Waits example termination
    print('Waiting rates subscriptions process termination...\n')
    while not func.isFinished():
        sleep(1)
                  
def main():
    print("==============Start Trading===========")
    trained_ddpg = model.load("./" + config.TRAINED_MODEL_DIR + "/DDPG.model")
    
    print("****Environment Document****")
    print(StockTradingEnvStopLoss_online.__doc__)
    
    print("****rates subscriptions process****')
    with open("./" + config.DATA_SAVE_DIR + "/symbols.txt", "r") as file:
        _symbols = eval(file.readline())
    process = multiprocessing.Process(target=rates_subscriptions(), args=(_symbols,))
    process.start()
    sleep(60)

    print("****Build Trade Environment****")
    file = open("./" + config.DATA_SAVE_DIR + "/balance.txt","r+") 
    initial_amount = file.read()
    file.close()
    information_cols = ["close", "macd", "boll_ub", "boll_lb", "rsi_30", "cci_30", "dx_30", "close_30_sma", "close_60_sma", "log_volume", "change", "daily_variance"]
    env_trade_kwargs = {'initial_amount': initial_amount*500,
                        'hmax': 100,
                        'daily_information_cols': information_cols, 
                        'print_verbosity': 500, 
                        'random_start': False,
                        'discrete_actions': True}
    e_trade_gym = StockTradingEnvStopLoss_online(**env_trade_kwargs)
    # this is our observation environment. It allows full diagnostics
    env_trade, _ = e_trade_gym.get_sb_env()
    
    print("==============Start Trading===========")
    print("****Model Prediction****")
    df_account_value, df_actions = DRLAgent.DRL_prediction(model=trained_ddpg, 
                                                           environment = e_trade_gym)
    
    print("****Prediction Resault Saving****")
    now = datetime.datetime.now().strftime("%Y%m%d-%Hh%M")
    df_account_value.to_csv("./" + config.RESULTS_DIR + "/_df_account_value" + now + ".csv")
    df_actions.to_csv("./" + config.RESULTS_DIR + "/_df_actions" + now + ".csv")
    
    print("****Get Backtest Results****")
    perf_stats_all = backtest_stats(account_value=df_account_value, value_col_name = 'total_assets')
    perf_stats_all = pd.DataFrame(perf_stats_all)
    perf_stats_all.to_csv("./" + config.RESULTS_DIR + "/_perf_stats_all" + now + ".csv")

    
if __name__ == "__main__":
    main()
