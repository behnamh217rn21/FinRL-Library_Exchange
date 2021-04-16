import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")

#############################################################################
from sklearn import preprocessing
import datetime

#############################################################################
from finrl.config import config
from finrl.marketdata.yahoodownloader import YahooDownloader
from finrl.preprocessing.preprocessors import FeatureEngineer
from finrl.preprocessing.data import data_split, load_dataset
from finrl.env.env_stocktrading_stoploss import StockTradingEnvStopLoss
from finrl.model.models import DRLAgent
from finrl.trade.backtest import backtest_plot, backtest_stats

#############################################################################
import multiprocessing

# Append path for main project folder
import sys
sys.path.append('../../..')
sys.path.append("./FinRL-Library_Exchange")


#############################################################################
#############################################################################
def main():
    """
    train an agent
    """
    print("==============Start Fetching Data===========")
    df = load_dataset(file_name="mt4_dataset_o.csv")
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

    # Training & Trading data split
    train = data_split(processed, config.START_DATE, config.START_TRADE_DATE)
    trade = data_split(processed, config.START_TRADE_DATE, config.END_DATE)
    print(StockTradingEnvStopLoss.__doc__)
    
    information_cols = ["close", "macd", "boll_ub",	"boll_lb", "rsi_30", "cci_30", "dx_30", "close_30_sma", "close_60_sma", "log_volume", "change", "daily_variance"]
    env_train_kwargs = {'initial_amount': 20000,
                        'hmax': 100, 
                        'cache_indicator_data': True,
                        'daily_information_cols': information_cols,
                        'print_verbosity': 500, 
                        'discrete_actions': True}
    e_train_gym = StockTradingEnvStopLoss(df = train, **env_train_kwargs)

    env_trade_kwargs = {'initial_amount': initial_amount,
                        'hmax': 100,
                        'daily_information_cols': information_cols, 
                        'print_verbosity': 500, 
                        'random_start': False,
                        'discrete_actions': True}
    e_trade_gym = StockTradingEnvStopLoss_online(df = trade, **env_trade_kwargs)
    
    # for this example, let's do multiprocessing with n_cores-2
    n_cores = multiprocessing.cpu_count() - 2
    print(f"using {n_cores} cores")   
    # this is our training env. It allows multiprocessing
    env_train, _ = e_train_gym.get_multiproc_env(n = n_cores)
    #env_train, _ = e_train_gym.get_sb_env()
    
    # this is our observation environment. It allows full diagnostics
    env_trade, _ = e_trade_gym.get_sb_env()

    agent = DRLAgent(env=env_train)
    print("==============Model Training===========")
    ddpg_params ={"actor_lr": 5e-06,
                  "critic_lr": 5e-06,
                  "gamma": 0.99,
                  "batch_size": 1024}
    
    model = agent.get_model("ddpg",
                            "eval_env": env_trade,
                            model_kwargs = ddpg_params,
                            verose = 0)

    trained_ddpg = agent.train_model(model=model, 
                                     tb_log_name="ddpg", 
                                     total_timesteps=32600000, 
                                     log_interval=1)

    trained_ddpg.save("./" + config.TRAINED_MODEL_DIR + "/DDPG.model")
    
    print("==============Start Trading===========")
    df_account_value, df_actions = DRLAgent.DRL_prediction(model=trained_ddpg, 
                                                           environment = e_trade_gym)
    
    now = datetime.datetime.now().strftime("%Y%m%d-%Hh%M")
    df_account_value.to_csv("./" + config.RESULTS_DIR + "/_df_account_value" + now + ".csv")
    df_actions.to_csv("./" + config.RESULTS_DIR + "/_df_actions" + now + ".csv")
    
    print("==============Get Backtest Results===========")
    perf_stats_all = backtest_stats(account_value=df_account_value, value_col_name = 'total_assets')
    perf_stats_all = pd.DataFrame(perf_stats_all)
    perf_stats_all.to_csv("./" + config.RESULTS_DIR + "/_perf_stats_all" + now + ".csv")
    

if __name__ == "__main__":
    main()
