#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    rates_subscriptions_v1.py
    
    An example using the Darwinex ZeroMQ Connector for Python 3 and MetaTrader 4  PULL REQUEST
    for v2.0.1 in which a Client modify instrument list configured in the EA to get rate prices.
    
    After receiving ... rates from ... it cancels its feed. At this point it cancels all rate feeds. 
    Then it prints _zmq._Market_Data_DB dictionary and finishes. 
    
    -------------------
    Rates feed:
    -------------------
    Through commmand TRACK_RATES, this client can select multiple INSTRUMENTS (symbol, timeframe).
    For example, to receive rates from instruments INTC(M1) and BAC(H1), this client
    will send this command to the Server, through its PUSH channel:

    "TRACK_RATES;INTC;1;BAC;60"

    Server will answer through the PULL channel with a json response like this:

    {'_action':'TRACK_RATES', '_data': {'instrument_count':2}}

    or if errors, then: 

    {'_action':'TRACK_RATES', '_data': {'_response':'NOT_AVAILABLE'}}

    Once subscribed to this feed, it will receive through the SUB channel, rates in this format:
    "INTC_M1 TIME;OPEN;HIGH;LOW;CLOSE;TICKVOL;SPREAD;REALVOL"
    "BAC_H1 TIME;OPEN;HIGH;LOW;CLOSE;TICKVOL;SPREAD;REALVOL"
        
    --
    
    @author: [raulMrello](https://www.linkedin.com/in/raul-martin-19254530/)
    
"""
#############################################################################
# DWX-ZMQ required imports 
#############################################################################
# Append path for main project folder
import sys
sys.path.append('../../..')

# Import ZMQ-Strategy from relative path
from examples.template.strategies.base.DWX_ZMQ_Strategy import DWX_ZMQ_Strategy

#################################################################################
from pandas import Timestamp
from datetime import timedelta
import datetime

import os
import pandas as pd
import numpy as np

#############################################################################
# Other required imports
#############################################################################
from threading import Lock
from time import sleep
import csv

###############################


#############################################################################
# Class derived from DWZ_ZMQ_Strategy includes data processor for PULL,SUB data
#############################################################################
class rates_subscriptions(DWX_ZMQ_Strategy):
    def __init__(self, 
                 _name="PRICES_SUBSCRIPTIONS",
                 _instruments=[('INTC_M1', 'INTC', 1), ('BAC_M1', 'BAC', 1)],
                 _delay=0.1,
                 _broker_gmt=2,
                 _verbose=False):
        
        # call DWX_ZMQ_Strategy constructor and passes itself as data processor for handling
        # received data on PULL and SUB ports 
        super().__init__(_name,
                         [],          # Empty symbol list (not needed for this example)
                         _broker_gmt,
                         [self],      # Registers itself as handler of pull data via self.onPullData()
                         [self],      # Registers itself as handler of sub data via self.onSubData()
                         _verbose)
        
        # This strategy's variables
        self._instruments = _instruments
        self._delay = _delay
        self._verbose = _verbose
        self._finished = False

        self.finish_time = Timestamp.now('UTC') + timedelta(days=365)
        self.finish_time = datetime.datetime.strftime(self.finish_time, "%Y.%m.%d %H:%M:00")
        self.finish_time = datetime.datetime.strptime(self.finish_time, "%Y.%m.%d %H:%M:00")

        self.p_time= '2000-01-01 00:00'

        cols = ["date", "open", "high", "low", "close", "volume", "spread", "real_volume", "tic",
                "macd", "boll_ub", "boll_lb", "rsi_30", "cci_30", "dx_30", "close_30_sma", "close_60_sma"]
        self.data_df = pd.DataFrame(columns=cols, dtype=float)
        self.cnt = -1

        # lock for acquire/release of ZeroMQ connector
        self._lock = Lock()
        
    ##########################################################################    
    def isFinished(self):        
        """ Check if execution finished"""
        return self._finished
        
    ##########################################################################    
    def onPullData(self, data):        
        """
        Callback to process new data received through the PULL port
        """        
        # print responses to request commands
        print('Response from ExpertAdvisor={}'.format(data))
        
    ##########################################################################    
    def onSubData(self, data):        
        """
        Callback to process new data received through the SUB port
        """
        # split data to get topic and message and balance
        _topic, _, _msg = data.split("&")
        print('Data on Topic={} with Message={} and Balance={}'.format(_topic,
                                                                       self._zmq._Market_Data_DB[_topic][self._zmq._timestamp],
                                                                       self._zmq._Balance
                                                                       ))

        if self._zmq._Market_Data_DB[_topic][self._zmq._timestamp][0] != self.p_time:
            f1 = open("./data_info" + "/balance.txt", 'w')
            f1.write(self._zmq._Balance); f1.close()
            self.p_time = self._zmq._Market_Data_DB[_topic][self._zmq._timestamp][0]

        file = "./data_info" + "/data.csv"
        ohlc, indicator = _msg.split("|")
        _time, _open, _high, _low, _close, _tick_vol, _spread, _real_vol = ohlc.split(",")
        _macd, _boll_ub, _boll_lb, _rsi_30, _cci_30, _adx_30, _close_30_sma, _close_60_sma = indicator.split(";")

        _time = pd.to_datetime(_time, format="%Y.%m.%d %H:%M")
        _time = datetime.datetime.strftime(_time, "%Y-%m-%d %H:%M:00")
        _time = datetime.datetime.strptime(_time, "%Y-%m-%d %H:%M:00")

        self.cnt += 1
        self.data_df.loc[self.cnt, :] = (str(_time), float(_open), float(_high), float(_low), float(_close), int(_tick_vol), int(_spread), int(_real_vol), _topic.split("_")[0], \
                                         float(_macd), float(_boll_ub), float(_boll_lb), float(_rsi_30), float(_cci_30), float(_adx_30), float(_close_30_sma), float(_close_60_sma))

        if ((self.cnt+1) % len(self._instruments)) == 0:
            self.data_df.drop(["spread", "real_volume"], axis=1, inplace=True)
            self.data_df["change"] = (self.data_df.close-self.data_df.open)/self.data_df.close
            self.data_df["log_volume"] = np.log(self.data_df.volume*self.data_df.close)
            self.data_df["daily_variance"] = (self.data_df.high-self.data_df.low)/self.data_df.close
            self.data_df.to_csv(file)

        _timestamp = pd.to_datetime(self._zmq._timestamp, format="%Y-%m-%d %H:%M:%S.%f")
        _timestamp = datetime.datetime.strftime(_timestamp, "%Y-%m-%d %H:%M:%S")
        _timestamp = datetime.datetime.strptime(_timestamp, "%Y-%m-%d %H:%M:%S")

        if _timestamp >= self.finish_time:
            # finishes (removes all subscriptions)  
            self.stop()
        
    ##########################################################################    
    def run(self):        
        """
        Starts price subscriptions
        """        
        self._finished = False

        # Subscribe to all symbols in self._symbols to receive bid,ask prices
        self.__subscribe_to_rate_feeds()

    ##########################################################################    
    def stop(self):
        """
        unsubscribe from all market symbols and exits
        """
        
        # remove subscriptions and stop symbols price feeding
        try:
            # Acquire lock
            self._lock.acquire()
            self._zmq._DWX_MTX_UNSUBSCRIBE_ALL_MARKETDATA_REQUESTS_()
            print('Unsubscribing from all topics')
          
        finally:
            # Release lock
            self._lock.release()
            sleep(self._delay)
      
        try:
            # Acquire lock
            self._lock.acquire()
            self._zmq._DWX_MTX_SEND_TRACKPRICES_REQUEST_([])        
            print('Removing symbols list')
            sleep(self._delay)
            self._zmq._DWX_MTX_SEND_TRACKRATES_REQUEST_([])
            print('Removing instruments list')

        finally:
            # Release lock
            self._lock.release()
            sleep(self._delay)

        self._finished = True


    ##########################################################################
    def __subscribe_to_rate_feeds(self):
        """
        Starts the subscription to the self._instruments list setup during construction.
        1) Setup symbols in Expert Advisor through self._zmq._DWX_MTX_SUBSCRIBE_MARKETDATA_
        2) Starts price feeding through self._zmq._DWX_MTX_SEND_TRACKRATES_REQUEST_
        """
        if len(self._instruments) > 0:
          # subscribe to all instruments' rate feeds
          for _instrument in self._instruments:
            try:
              # Acquire lock
              self._lock.acquire()
              self._zmq._DWX_MTX_SUBSCRIBE_MARKETDATA_(_instrument[0])
              print('Subscribed to {} rate feed'.format(_instrument))
              
            finally:
              # Release lock
              self._lock.release()        
            sleep(self._delay)

          # configure instruments to receive price feeds
          try:
            # Acquire lock
            self._lock.acquire()
            self._zmq._DWX_MTX_SEND_TRACKRATES_REQUEST_(self._instruments)
            print('Configuring rate feed for {} instruments'.format(len(self._instruments)))
            
          finally:
            # Release lock
            self._lock.release()
            sleep(self._delay)     
