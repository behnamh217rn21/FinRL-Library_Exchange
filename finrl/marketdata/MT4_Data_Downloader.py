#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    rates_historic.py
    
    An example using the Darwinex ZeroMQ Connector for Python 3 and MetaTrader 4 PULL REQUEST
    for v2.0.1 in which a Client requests rate history from EURGBP Daily from 2019.01.04 to
    to 2019.01.14.
    -------------------
    Rates history:
    -------------------
    Through commmand HIST, this client can select multiple rates from an INSTRUMENT (symbol, timeframe).
    For example, to receive rates from instruments EURUSD(M1), between two dates, it will send this 
    command to the Server, through its PUSH channel:
    "HIST;EURUSD;1;2019.01.04 00:00:00;2019.01.14 00:00:00"
      
    --
    
    @author: [raulMrello](https://www.linkedin.com/in/raul-martin-19254530/)
    
"""
#############################################################################
# DWX-ZMQ required imports 
#############################################################################
# Import ZMQ-Strategy from relative path
from finrl.env.dwx_zeromq_connector.strategies.base.DWX_ZMQ_Strategy import DWX_ZMQ_Strategy
from finrl.config import config

#############################################################################
# Other required imports
#############################################################################
import os
from pandas import Timedelta, to_datetime, Timestamp
from threading import Thread, Lock
from time import sleep
import random
import pandas as pd
import datetime


#############################################################################
# Class derived from DWZ_ZMQ_Strategy includes data processor for PULL,SUB data
#############################################################################
class rates_historic(DWX_ZMQ_Strategy):
    def __init__(self,
                 _name="PRICES_SUBSCRIPTIONS",
                 _symbols=['EURUSD', 'GDAXI'],
                 _TF=60,
                 _delay=0.1,
                 _broker_gmt=3,
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
        self._symbols_list = _symbols
        self._TF = _TF
        self._verbose = _verbose
        self._finished = False; self._delay = _delay

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
        #print('Historic from ExpertAdvisor={}'.format(data))
        
    ##########################################################################    
    def onSubData(self, data):        
        """
        Callback to process new data received through the SUB port
        """
        # split msg to get topic and message
        _topic, _msg = data.split(" ")
        print('Data on Topic={} with Message={}'.format(_topic, _msg))
   
    ##########################################################################    
    def run(self):        
        """
        Request historic data
        """        
        self._finished = False

        # request rates
        for i in range(0, len(self._symbols_list)):
            print('Requesting {} Rates from {}'.format(self._TF, self._symbols_list[i]))
            self._zmq._DWX_MTX_SEND_HIST_REQUEST_(_symbol=self._symbols_list[i],
                                                  _timeframe=self._TF,
                                                  _start='2020.02.05 18:30:00',
                                                  _end=Timestamp.now().strftime('%Y.%m.%d %H:%M:00'))
            sleep(2)

        print('\nCreating a History Data Dataframe:')
        counter = 0
        _HIST_DATA_DF = pd.DataFrame()
        for symbol in self._symbols_list:
            symbol_H1 = "{}_H1".format(symbol)
            for i in range(0, len(self._zmq._History_DB[symbol_H1])):
                _HIST_DATA_DF=_HIST_DATA_DF.append([self._zmq._History_DB[symbol_H1][i].values()])
                _HIST_DATA_DF = _HIST_DATA_DF.reset_index(drop=True)
                _HIST_DATA_DF.loc[counter, "tic"] = symbol
                counter += 1

        _HIST_DATA_DF.columns = ["date",
                                 "open",
                                 "high",
                                 "low",
                                 "close",
                                 "volume",
                                 "spread",
                                 "real_vol",
                                 "tic",
                                 ]

        _HIST_DATA_DF.drop('spread', axis=1, inplace=True)
        _HIST_DATA_DF.drop('real_vol', axis=1, inplace=True)

        # reset the index, we want to use numbers as index instead of dates
        _HIST_DATA_DF = _HIST_DATA_DF.reset_index(drop=True)

        print("Saving...")
        # convert date to standard string format, easy to filter
        for i in range(0, len(_HIST_DATA_DF)):
            date = datetime.datetime.strptime(str(_HIST_DATA_DF.loc[i, "date"]), "%Y.%m.%d %H:%M")
            _HIST_DATA_DF.loc[i, "date"] = date.strftime("%Y-%m-%d %H:%M:00")

        _HIST_DATA_DF["date"] = pd.to_datetime(_HIST_DATA_DF["date"])
        _HIST_DATA_DF = _HIST_DATA_DF.sort_values(by=['date','tic']).reset_index(drop=True)
        
        _HIST_DATA_DF.to_csv("./" + config.DATASET_DIR + "/mt4_dataset.csv")

        # finishes (removes all subscriptions)  
        self.stop()

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
      
      self._finished = True


""" -----------------------------------------------------------------------------------------------
    -----------------------------------------------------------------------------------------------
    SCRIPT SETUP
    -----------------------------------------------------------------------------------------------
    -----------------------------------------------------------------------------------------------
"""
if __name__ == "__main__":
  
  # creates object with a predefined configuration: historic EURGBP_D1 between 4th adn 14th January 2019
  print('Loading example...')
  example = rates_historic()  

  # Starts example execution
  print('\nrunning example...')  
  example.run()

  # Waits example termination
  print('Waiting example termination...')
  while not example.isFinished():
    sleep(1)
  print('Bye!!!')
