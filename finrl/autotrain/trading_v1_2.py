import pandas as pd
import datetime
import matplotlib
matplotlib.use("Agg")

import warnings
warnings.filterwarnings('ignore')

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
sys.path.append("..\\FinRL-Library_Master")


#############################################################################
#############################################################################                 
def main():
    def rates_subscriptions(_symbols):
        # creates object with a predefined configuration
        print('running rates subscriptions process ...')
        func = rates_subscriptions_v1.rates_subscriptions(_instruments=_symbols)
        func.run()
        # Waits example termination
        print('Waiting rates subscriptions process termination...\n')
        while not func.isFinished():
            sleep(1)
            
    print("==============Start Trading===========")
    DDPG_model = model.load("./" + config.TRAINED_MODEL_DIR + "/DDPG.model")

    #print("****Environment Document****")
    #print(StockTradingEnvStopLoss_online.__doc__)
    
    print("****rates subscriptions process****")
    with open("./" + config.DATA_SAVE_DIR + "/symbols.txt", "r") as file:
        _symbols = eval(file.readline())
    process = multiprocessing.Process(target=rates_subscriptions, args=(_symbols,))
    process.start()
    sleep(60)

    print("****Build Trade Environment****")
    file = open("./" + config.DATA_SAVE_DIR + "/balance.txt","r+") 
    initial_amount = file.read()
    initial_amount = float(initial_amount)
    file.close()
    
    _symbols_i1 = []
    for i in range(0, len(_symbols)):
        _symbols_i1.append(_symbols[i][1])
        
    information_cols = ["close", "macd", "boll_ub", "boll_lb", "rsi_30", "cci_30", "dx_30", 
                        "close_30_sma", "close_60_sma", "log_volume", "change", "daily_variance"]
    
    from pathlib import Path
    path = Path(__file__).resolve().parents[4].joinpath("AppData/Roaming/MetaQuotes/Terminal/58F16B8C9F18D6DD6A5DAC862FC9CB62/" \
                                                        "MQL4/Files/Leverage.txt")
    with open(path, 'r') as reader:
        Leverage = reader.read()
    print("Leverage : {}".format(Leverage))
    env_trade_kwargs = {'initial_amount': initial_amount*float(Leverage),
                        'assets': _symbols_i1,
                        'sell_cost_pct': 0,
                        'buy_cost_pct': 0,
                        'hmax': 100,
                        'cash_penalty_proportion': 0.2,
                        'daily_information_cols': information_cols, 
                        'print_verbosity': 500, 
                        'discrete_actions': False}
    e_trade_gym = StockTradingEnvStopLoss_online(**env_trade_kwargs)
    
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
