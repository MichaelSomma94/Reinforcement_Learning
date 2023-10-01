#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 17:21:37 2023

@author: michiundslavki
"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import datetime


from finrl import config
from finrl import config_tickers
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline, convert_daily_return_to_pyfolio_ts
from pyfolio import timeseries
import pyfolio
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
import plotly
from finrl.config import (
    DATA_SAVE_DIR,
    TRAINED_MODEL_DIR,
    TENSORBOARD_LOG_DIR,
    RESULTS_DIR,
    INDICATORS,
    TRAIN_START_DATE,
    TRAIN_END_DATE,
    TEST_START_DATE,
    TEST_END_DATE,
    TRADE_START_DATE,
    TRADE_END_DATE,
)
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.env_util import make_vec_env
from tensorboardX import SummaryWriter
from env.portfolio_allocations import StockPortfolioEnv
import plotly.graph_objs as go


if not os.path.exists("./datasets"):
    os.makedirs("./datasets")
if not os.path.exists("./trained_models"):
    os.makedirs("./trained_models")
if not os.path.exists("./tensorboard_log"):
    os.makedirs("./tensorboard_log")
if not os.path.exists("./results"):
    os.makedirs("./results")

def train_split(df, TRAIN_START, TRAIN_END, lookback=365):
    """
    Filters the dataframe based on the provided date, returning rows 
    from 'lookback' days before the date up to the present data in the DataFrame.
    
    Parameters:
    - df: DataFrame to filter. Assumes the date column is named 'date' and is in datetime format.
    - date_str: Date as a string in format 'YYYY-MM-DD'.
    - lookback: Number of days to look back from the given date.
    
    Returns:
    - Filtered DataFrame or raises an exception if conditions aren't met.
    """
    
    start_date = pd.to_datetime(TRAIN_START)
    end_date = pd.to_datetime(TRAIN_END)
    lookback_date = start_date - pd.Timedelta(days=lookback)
    # Assert that the given date is a weekday
    #assert start_date.weekday() <= 4, "The given start date is not a weekday."
    #assert end_date.weekday() <= 4, "The given end date is not a weekday."
     # Determine start date for the lookback
    assert lookback_date >= df['date'].iloc[0], "The given start date is not in the provided data."
    #print(df['date'].iloc[0])
    try:
        
        # Filter the DataFrame based on the date range
        filtered_df = df[(df['date'] >= lookback_date) & (df['date'] <= end_date)]
        
        # Assert that the resulting DataFrame is not empty
        assert not filtered_df.empty, "No rows available in the specified range."
        
        return filtered_df.reset_index(drop=True)
    except AssertionError as e:
        print(f"Error: {e}")
        return None

def trade_split(df, TEST_START, TEST_END, lookback=365):
    """
    Filters the dataframe based on the provided date, returning rows 
    from 'lookback' days before the date up to the present data in the DataFrame.
    
    Parameters:
    - df: DataFrame to filter. Assumes the date column is named 'date' and is in datetime format.
    - date_str: Date as a string in format 'YYYY-MM-DD'.
    - lookback: Number of days to look back from the given date.
    
    Returns:
    - Filtered DataFrame or raises an exception if conditions aren't met.
    """
    
    start_date = pd.to_datetime(TEST_START)
    end_date = pd.to_datetime(TEST_END)
    lookback_date = start_date - pd.Timedelta(days=lookback)
    # Assert that the given date is a weekday
    #assert start_date.weekday() <= 4, "The given start date is not a weekday."
    #assert end_date.weekday() <= 4, "The given end date is not a weekday."
     # Determine start date for the lookback
    assert lookback_date >= df['date'].iloc[0], "The given start date is not in the provided data."
    
    #assert end_date <= df['date'].iloc[-1], "The given start date is not in the provided data."
    try:
        # Filter the DataFrame based on the date range
        filtered_df = df[(df['date'] >= lookback_date) & (df['date'] <= end_date)]
        
        # Assert that the resulting DataFrame is not empty
        assert not filtered_df.empty, "No rows available in the specified range."
        
        return filtered_df.reset_index(drop=True)
    except AssertionError as e:
        print(f"Error: {e}")
        return None

# add covariance matrix as states
def add_cov(df, look_back=252):
    try: 
        df=df.sort_values(['date','tic'],ignore_index=True)
        df.index = df.date.factorize()[0]
        #print(len(df.index.unique()))
        #given_date = pd.to_datetime('2016-01-01)
        #print(df[df[])
    
        cov_list = []
        return_list = []
        
        # look back is one year
   
        for i in range(look_back,len(df.index.unique())):
          data_lookback = df.loc[i-look_back:i,:]
          #print(data_lookback)
          price_lookback=data_lookback.pivot_table(index = 'date',columns = 'tic', values = 'close')
          return_lookback = price_lookback.pct_change().dropna()
          return_list.append(return_lookback)
        
          covs = return_lookback.cov().values 
          cov_list.append(covs)
        
          
        df_cov = pd.DataFrame({'date':df.date.unique()[look_back:],'cov_list':cov_list,'return_list':return_list})
        df = df.merge(df_cov, on='date')
        df = df.sort_values(['date','tic']).reset_index(drop=True)
        df.index = df.date.factorize()[0]
        #df = df.reset_index(drop=True)
        return df
    except Exception as e:
        print(f"An error occurred: {type(e).__name__} - {str(e)}")

def back_testing(df_daily_reuturn, index):
    try: 
        baseline_df = get_baseline(
                ticker=index, 
                start = df_daily_return.loc[0,'date'],
                end = df_daily_return.loc[len(df_daily_return)-1,'date'])
    
        stats = backtest_stats(baseline_df, value_col_name = 'close')
    
        return stats
    except Exception as e:
        print(f"An error occurred: {type(e).__name__} - {str(e)}")
def minimum_variance(df, unique_tic, unique_trade_date, initial_capital):
    try: 
        #unique_tic = trade_df.tic.unique()
        
        #unique_trade_date = trade_df.date.unique()
        #calculate_portfolio_minimum_variance
        portfolio = pd.DataFrame(index = range(1), columns = unique_trade_date)
        initial_capital = 7000
        portfolio.loc[0,unique_trade_date[0]] = initial_capital
        
        for i in range(len(unique_trade_date)-1):
            df_temp = df[df.date==unique_trade_date[i]].reset_index(drop=True)
            df_temp_next = df[df.date==unique_trade_date[i+1]].reset_index(drop=True)
            #print(df_temp)
            #Sigma = risk_models.sample_cov(df_temp.return_list[0])
            #print(Sigma)
            #calculate covariance matrix
            Sigma = df_temp.return_list[0].cov()
            #portfolio allocation
            ef_min_var = EfficientFrontier(None, Sigma,weight_bounds=(0, 0.1))
            #minimum variance
            raw_weights_min_var = ef_min_var.min_volatility()
            #get weights
            cleaned_weights_min_var = ef_min_var.clean_weights()
            
            #current capital
            cap = portfolio.iloc[0, i]
            #current cash invested for each stock
            current_cash = [element * cap for element in list(cleaned_weights_min_var.values())]
            # current held shares
            current_shares = list(np.array(current_cash)
                                             / np.array(df_temp.close))
            # next time period price
            next_price = np.array(df_temp_next.close)
            #next_price * current share to calculate next total account value 
            portfolio.iloc[0, i+1] = np.dot(current_shares, next_price)
            
        portfolio=portfolio.T
        portfolio.columns = ['account_value']
        return portfolio
            
    
    except Exception as e:
            print(f"An error occurred: {type(e).__name__} - {str(e)}")
# getting the data
DOWNLOAD_START = '2017-01-01'
DOWNLOAD_END ='2023-08-31' 
TRAIN_START = '2018-01-04'
TRAIN_END = '2022-03-30'
TRADE_START = '2022-03-30'
TRADE_END = '2022-04-20'


tic_list = ['AXP',
 'AMGN',
 'AAPL',
 'BA',
 'CAT',
 'CSCO',
 'CVX',
 'GS',
 'HD',
 'HON',
 'IBM',
 'INTC',
 'JNJ',
 'KO',
 'JPM',
 'MCD',
 'MMM',
 'MRK',
 'MSFT',
 'NKE',
 'PG',
 'TRV',
 'UNH',
 'CRM',
 'VZ',
 'V',
 'WBA',
 'WMT', 'DIS', 'DOW']

INDICATORS = ['macd', 'boll_ub', 'boll_lb', 'rsi_30', 'cci_30', 'dx_30', 'close_30_sma', 'close_60_sma']

if __name__ == '__main__' :
    
    writer = SummaryWriter(log_dir="./tensorboard_log")
    
    df = YahooDownloader(start_date = DOWNLOAD_START,
                         end_date = DOWNLOAD_END,
                         ticker_list = tic_list).fetch_data()
    df['date'] = pd.to_datetime(df['date'])
    # add technical indicators 
    fe = FeatureEngineer(
                        use_technical_indicator=True,
                        use_turbulence=False,
                        user_defined_feature = False)
    
    df = fe.preprocess_data(df)
    # spliting in train/trade
    train = train_split(df, TRAIN_START, TRAIN_END) 
    trade = trade_split(df, TRADE_START, TRADE_END)
    #adding Covariance
    df_cov = add_cov(df)
    train_cov = add_cov(train)
    trade_cov = add_cov(trade)
    stock_dimension = len(train.tic.unique())
    state_space = stock_dimension
   
    env_kwargs = {
    "hmax": 100, 
    "initial_amount": 7000, 
    "transaction_cost_pct": 0.001, 
    "state_space": state_space, 
    "stock_dim": stock_dimension, 
    "tech_indicator_list": INDICATORS, 
    "action_space": stock_dimension, 
    "reward_scaling": 1e-4
    
    }
    # Parameter definition
    tb_log_id = "13_stdew_sm_period"
    num_steps_train = 30000
    plot_path ="./results/plots"
    performance_path = "./results/performance"
    heat_map_id = f"heatmap_{tb_log_id}"
    # training Environment + Agent
    

    e_train_gym = StockPortfolioEnv(df = train_cov, **env_kwargs)
    e_train_gym.reset()
    model_a_2c = A2C("MlpPolicy", e_train_gym,learning_rate=0.0001, tensorboard_log="./tensorboard_log", normalize_advantage=True) #"MlpPolicy"
    model_a_2c.learn(total_timesteps=num_steps_train, tb_log_name=tb_log_id)
  
    #e_train_gym.reset()
    #model_a_2ppo = PPO("MlpPolicy", e_train_gym,learning_rate=0.0001, tensorboard_log="./tensorboard_log", normalize_advantage=True, verbose=1) #"MlpPolicy"
   # model_a_2ppo.learn(total_timesteps=num_steps_train, tb_log_name=tb_log_id)

    # Get Information about performance
  
    e_trade_gym = StockPortfolioEnv(df = trade_cov, **env_kwargs)
    e_trade_gym.reset()
    
    df_daily_return, df_actions = DRLAgent.DRL_prediction(model=model_a_2c,
                        environment = e_trade_gym)
    #df_daily_return_ppo, df_actions_ppo = DRLAgent.DRL_prediction(model=model_a_2ppo,
    #                    environment = e_trade_gym)
    #print(df_daily_return.head())
        # get statistics of our agent trading over the test period
    DRL_strat = convert_daily_return_to_pyfolio_ts(df_daily_return)
    #DRL_strat_ppo = convert_daily_return_to_pyfolio_ts(df_daily_return_ppo)
    #plots

    #pyfolio.plot_monthly_returns_heatmap(DRL_strat)
    #plt.savefig(f"{plot_path}/{heat_map_id}.png")
   
    # get Back_test_performance
    stats = back_testing(df_daily_return, "^DJI")
    stats.to_csv(f"{performance_path}/back_test_{tb_log_id}.csv", index=True)
    baseline_index = "^DJI"
    baseline_df = get_baseline(
        ticker=baseline_index, 
        start = df_daily_return.loc[0,'date'],
        end = df_daily_return.loc[len(df_daily_return)-1,'date'])
    baseline_returns = get_daily_return(baseline_df, value_col_name="close")
    #print(baseline_returns.head())
    #get performance of the Agent
    perf_func = timeseries.perf_stats 
    perf_stats_all = perf_func(returns=DRL_strat, 
                              factor_returns=DRL_strat, 
                                positions=None, transactions=None, turnover_denom="AGB")
    print(perf_stats_all)
    #perf_stats_all_ppo = perf_func(returns=DRL_strat_ppo, 
     #                         factor_returns=DRL_strat_ppo, 
     #                           positions=None, transactions=None, turnover_denom="AGB")
    #print(perf_stats_all_ppo)
    
    #perf_stats_all.to_csv(f"{performance_path}/DRL_{tb_log_id}.csv", index=True)
    #print(type(perf_stats_all), perf_stats_all)
    
    #get Minimum Variance Portfolio Performance
    unique_tic = trade_cov.tic.unique()
    unique_trade_date = trade_cov.date.unique()
    df_cov = add_cov(df)
    portfolio_min_var = minimum_variance(df_cov, unique_tic, unique_trade_date, env_kwargs['initial_amount'])
    
    
    # compare the Performance of MinVar, DRL, DJI
    a2c_cumpod =(df_daily_return.daily_return+1).cumprod()-1
    min_var_cumpod =(portfolio_min_var.account_value.pct_change()+1).cumprod()-1
    dji_cumpod =(baseline_returns+1).cumprod()-1
    
    # get final comparison plot
    time_ind = pd.Series(df_daily_return.date)
    trace0_portfolio = go.Scatter(x = time_ind, y = a2c_cumpod, mode = 'lines', name = 'A2C (Portfolio Allocation)')

    trace1_portfolio = go.Scatter(x = time_ind, y = dji_cumpod, mode = 'lines', name = 'DJIA')
    trace2_portfolio = go.Scatter(x = time_ind, y = min_var_cumpod, mode = 'lines', name = 'Min-Variance')
    #trace3_portfolio = go.Scatter(x = time_ind, y = ddpg_cumpod, mode = 'lines', name = 'DDPG')
    #trace4_portfolio = go.Scatter(x = time_ind, y = addpg_cumpod, mode = 'lines', name = 'Adaptive-DDPG')
    #trace5_portfolio = go.Scatter(x = time_ind, y = min_cumpod, mode = 'lines', name = 'Min-Variance')
    
    #trace4 = go.Scatter(x = time_ind, y = addpg_cumpod, mode = 'lines', name = 'Adaptive-DDPG')
    
    #trace2 = go.Scatter(x = time_ind, y = portfolio_cost_minv, mode = 'lines', name = 'Min-Variance')
    #trace3 = go.Scatter(x = time_ind, y = spx_value, mode = 'lines', name = 'SPX')
    
    fig = go.Figure()
    fig.add_trace(trace0_portfolio)
    
    fig.add_trace(trace1_portfolio)
    
    fig.add_trace(trace2_portfolio)
    
    
    
    fig.update_layout(
        legend=dict(
            x=0,
            y=1,
            traceorder="normal",
            font=dict(
                family="sans-serif",
                size=15,
                color="black"
            ),
            bgcolor="White",
            bordercolor="white",
            borderwidth=2
            
        ),
    )
    #fig.update_layout(legend_orientation="h")
    fig.update_layout(title={
            #'text': "Cumulative Return using FinRL",
            'y':0.85,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'})
    #with Transaction cost
    #fig.update_layout(title =  'Quarterly Trade Date')
    fig.update_layout(
    #    margin=dict(l=20, r=20, t=20, b=20),
    
        paper_bgcolor='rgba(1,1,0,0)',
        plot_bgcolor='rgba(1, 1, 0, 0)',
        #xaxis_title="Date",
        yaxis_title="Cumulative Return",
    xaxis={'type': 'date', 
           'tick0': time_ind[0], 
            'tickmode': 'linear', 
           'dtick': 86400000.0 *80}
    
    )
    fig.update_xaxes(showline=True,linecolor='black',showgrid=True, gridwidth=1, gridcolor='LightSteelBlue',mirror=True)
    fig.update_yaxes(showline=True,linecolor='black',showgrid=True, gridwidth=1, gridcolor='LightSteelBlue',mirror=True)
    fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor='LightSteelBlue')
    fig.write_image(f"{plot_path}/pf_performance_{tb_log_id}.png")
    #fig.show()
    # HERE
    # Create a summary writer
    
    #test_scalar = 9
    
    # Inside your training or execution loop, log data
    
    #writer.add_scalar("Portfolio Value", test_scalar)#, iteration)
    
    #Log Images
    #with open(f"{plot_path}/{heat_map_id}.png", 'rb') as image_file:
    #    heat_map = image_file.read()
    #writer.add_image('heat_map', heat_map)
    
    # Log other metrics and values as needed
    
    # Close the writer when done
    #writer.close()

    