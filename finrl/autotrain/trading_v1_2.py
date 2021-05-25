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
    information_cols = ["close", "upper_band", "lower_band", "ema", "macd_signal", "macd_hist", "cci", "atr", "rsi", "adx"]
    env_trade_kwargs = {'sell_cost_pct': 0,
                        'buy_cost_pct': 0,
                        'hmax': 1,
                        'cash_penalty_proportion': 0.2,
                        'daily_information_cols': information_cols, 
                        'print_verbosity': 1, 
                        'discrete_actions': False,
                        'patient': True}
    e_trade_gym = StockTradingEnvStopLossOnline(**env_trade_kwargs)
    env_trade, _ = e_trade_gym.get_sb_env()
    
    print("****Model Prediction****")
    agent = DRLAgent(env=env_trade)
    ddpg_params = {"actor_lr": 5e-06,
                   "critic_lr": 5e-06,
                   "gamma": 0.99,
                   "batch_size": 64}

    model = agent.get_model("ddpg",
                            model_kwargs = ddpg_params,
                            verbose = 0)
    
    DDPG_model = model.load(DDPG_model_path)
    DRLAgent.DRL_prediction_online(model=DDPG_model,
                                   environment=e_trade_gym)
    
    """
    df_account_value, df_actions = DRLAgent.DRL_prediction(model=DDPG_model,
                                                           #environment=e_trade_gym)
    
    print("****Prediction Resault Saving****")
    now = datetime.datetime.now().strftime("%Y-%m-%d-%HH%MM")
    df_account_value.to_csv("./" + config.RESULTS_DIR + "/_df_account_value_" + now + ".csv")
    df_actions.to_csv("./" + config.RESULTS_DIR + "/_df_actions_" + now + ".csv")
    
    print("****Get Backtest Results****")
    perf_stats_all = backtest_stats(account_value=df_account_value, value_col_name = 'total_assets')
    perf_stats_all = pd.DataFrame(perf_stats_all)
    perf_stats_all.to_csv("./" + config.RESULTS_DIR + "/_perf_stats_all_" + now + ".csv")
    """

if __name__ == "__main__":
    main()
