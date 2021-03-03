import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import preprocessing

matplotlib.use("Agg")
import datetime

from finrl.config import config
from finrl.marketdata.yahoodownloader import YahooDownloader
from finrl.preprocessing.preprocessors import FeatureEngineer
from finrl.preprocessing.data import data_split
from finrl.env.env_stocktrading_stoploss import StockTradingEnvStopLoss
from finrl.model.models import DRLAgent
from finrl.trade.backtest import backtest_plot, backtest_stats
import multiprocessing


def train_one():
    """
    train an agent
    """
    print("==============Start Fetching Data===========")
    df = YahooDownloader().fetch_data()
    
    print("==============Start Feature Engineering===========")
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=config.TECHNICAL_INDICATORS_LIST,
        use_turbulence=True,
        user_defined_feature=False,
    )

    processed = fe.preprocess_data(df)
    
    processed['log_volume'] = np.log(processed.volume*processed.close)
    processed['change'] = (processed.close-processed.open)/processed.close
    processed['daily_variance'] = (processed.high-processed.low)/processed.close

    # Training & Trading data split
    train = data_split(processed, config.START_DATE, config.START_TRADE_DATE)
    trade = data_split(processed, config.START_TRADE_DATE, config.END_DATE)
    
    print(StockTradingEnvStopLoss.__doc__)

    information_cols = ['daily_variance', 'change', 'log_volume', 'close', 'day', 
                        'macd', 'rsi_30', 'cci_30', 'dx_30']
    env_train_kwargs = {
        initial_amount = 1e6,hmax = 5000, 
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
    e_train_gym = StockTradingEnvStopLoss(df=train, **env_kwargs)
    
    env_trade_kwargs = {
        initial_amount = 1e6,hmax = 5000, 
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
    e_trade_gym = StockTradingEnvStopLoss(df=trade, turbulence_threshold=250, **env_kwargs)
    
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
    now = datetime.datetime.now().strftime("%Y%m%d-%Hh%M")

    ddpg_params ={'actor_lr': 5e-06,
                  'critic_lr': 5e-06,
                  'gamma': 0.99,
                  'batch_size': 1024,
                  'eval_env': env_trade,
                  'nb_eval_steps': 50
                 }
    model_name = "ddpg"
    model = agent.get_model(model_name,  
                            model_kwargs = ddpg_params, 
                            verbose = 0
                           )

    trained_ddpg = agent.train_model(
        model=model, tb_log_name="ddpg", total_timesteps=80000, log_interval=1
    )

    #model.save("trained_models/DDPG.model")
    
    logs_base_dir = f"{config.TENSORBOARD_LOG_DIR}/{model_name}"
    os.makedirs(logs_base_dir, exist_ok=True)
    %load_ext tensorboard
    %tensorboard --logdir {logs_base_dir}
    
    print("==============Start Trading===========")
    df_account_value, df_actions = DRLAgent.DRL_prediction(model=trained_ddpg, 
                                                           environment = e_trade_gym
                                                          )
    
    df_account_value.to_csv("./" + config.RESULTS_DIR + "/df_account_value_" + now + ".csv")
    df_actions.to_csv("./" + config.RESULTS_DIR + "/df_actions_" + now + ".csv")

    print("==============Get Backtest Results===========")
    perf_stats_all = backtest_stats(account_value=df_account_value, value_col_name = 'total_assets')
    perf_stats_all = pd.DataFrame(perf_stats_all)
    perf_stats_all.to_csv("./" + config.RESULTS_DIR + "/perf_stats_all_" + now + ".csv")
    
    
    print("==============Compare to DJIA===========")
    %matplotlib inline
    # S&P 500: ^GSPC
    # Dow Jones Index: ^DJI
    # NASDAQ 100: ^NDX
    backtest_plot(df_account_value, 
                  baseline_ticker = '^DJI', 
                  baseline_start = config.START_TRADE_DATE,
                  baseline_end = config.END_DATE, value_col_name = 'total_assets')
