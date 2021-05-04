# -*- coding: utf-8 -*-
"""
    traders.py
    An example trading strategy created using the Darwinex ZeroMQ Connector
    for Python 3 and MetaTrader 4.   
    Source code:
    https://github.com/darwinex/DarwinexLabs/tree/master/tools/dwx_zeromq_connector
    
    @author: Darwinex Labs (www.darwinex.com)    
    Copyright (c) 2019 onwards, Darwinex. All rights reserved.
    
    Licensed under the BSD 3-Clause License, you may not use this file except 
    in compliance with the License.  
    You may obtain a copy of the License at:    
    https://opensource.org/licenses/BSD-3-Clause
"""
#############################################################################
#############################################################################
from finrl.env.dwx_zeromq_connector.strategies.base.DWX_ZMQ_Strategy import DWX_ZMQ_Strategy

from threading import Thread, Lock
from time import sleep
import random
import pandas as pd


class t_class(DWX_ZMQ_Strategy):
    def __init__(self, _name="ONLINE_TRADERS",
                 _symbols=['#INTC', '#AAPL'],
                 _delay=0.1,
                 _broker_gmt=2,
                 _verbose=False
                 ):
        super().__init__(_name, _symbols, _broker_gmt, _verbose)
        
        # This strategy's variables
        self._delay = _delay
        self._traders = []
        self._verbose = _verbose
        
        self._symbols = _symbols
        self._ot = pd.DataFrame()
        
        # lock for acquire/release of ZeroMQ connector
        self._lock = Lock()
        
    ##########################################################################
    def _run_(self, sells, buys):
        """
        Logic:
            For each symbol in self._symbols:
                
                1) Open a new Market Order every 2 seconds
                2) Close any orders that have been running for 10 seconds
                3) Calculate Open P&L every second
                4) Plot Open P&L in real-time
                5) Lot size per trade = 0.01
                6) SL/TP = 10 pips each
        """      
        # Launch traders!
        for index, _symbol in enumerate(self._symbols):
            _t = Thread(name="{}_Trader".format(_symbol),
                        target=self._trader_, 
                        args=(_symbol, 
                              sells[index],
                              buys[index]))
            _t.daemon = True
            _t.start()
            
            print('[{}_Trader] Alright ...'.format(_symbol))
            self._traders.append(_t)
        
    ##########################################################################
    def _trader_(self, _symbol, sell, buy):
        _random_int  = random.randint(1, 999)
        value = "_default_order_{}".format(str(_random_int))
        x_num = 'value'
        # Note: Just for this example, only the Order Type is dynamic.
        globals()[x_num] = self._zmq._generate_default_order_dict()
        globals()[x_num]['_symbol'] = _symbol
        globals()[x_num]['_SL'] = 100
        globals()[x_num]['_TP'] = 100
        globals()[x_num]['_comment'] = '{}_Trader'.format(_symbol)
        """
        Default Order:
        --
        {'_action': 'OPEN',
         '_type': 0,
         '_symbol': EURUSD,
         '_price':0.0,
         '_SL': 100,                     # 10 pips
         '_TP': 100,                     # 10 pips
         '_comment': 'EURUSD_Trader',
         '_lots': 0.01,
         '_magic': 123456}
        """
        print("symbol: {}; sell: {}; buy: {}".format(_symbol, sell, buy))
        try:
            # Acquire lock
            self._lock.acquire()
            
            self._ot = self._reporting._get_open_trades_('{}_Trader'.format(_symbol),
                                                         self._delay,
                                                         10)
            print("trade counter: {}".format(self._ot.shape[0]))
            
            # Reset cycle if nothing received
            if self._zmq._valid_response_(self._ot) == False:
                print("Nothing Received")
                
            ###############################
            # SECTION - SELL TRADES #
            ###############################
            if self._ot.shape[0] > 0:
                if sell != 0:
                    for i in (self._ot.loc[self._ot["_symbol"] == _symbol].index):
                        if sell < self._ot["_lots"].loc[self._ot.index == i]:
                            _ret_cp = self._execution._execute_({'_action': 'CLOSE_PARTIAL',
                                                                 '_ticket': i,
                                                                 'size': sell,
                                                                 '_comment': '{}_Trader'.format(_symbol)},
                                                                self._verbose,
                                                                self._delay,
                                                                10)
                            # Reset cycle if nothing received
                            if self._zmq._valid_response_(_ret_cp) == False:
                                print("Nothing Received")
                                continue   
                            break
                        
                        elif sell == self._ot["_lots"].loc[self._ot.index == i]:
                            _ret_c = self._execution._execute_({'_action': 'CLOSE',
                                                                '_ticket': i,
                                                                'size': sell},
                                                               self._verbose,
                                                               self._delay,
                                                               10)
                            # Reset cycle if nothing received
                            if self._zmq._valid_response_(_ret_c) == False:
                                print("Nothing Received")
                                continue
                            break
                        else:
                            sell = sell - self._ot["_lots"].loc[self._ot["_symbol"] == _symbol]
                            _ret_c = self._execution._execute_({'_action': 'CLOSE',
                                                                '_ticket': i,
                                                                'size': self._ot["_lots"].loc[self._ot.index == i]},
                                                               self._verbose,
                                                               self._delay,
                                                               10)
                            # Reset cycle if nothing received
                            if self._zmq._valid_response_(_ret_c) == False:
                                print("Nothing Received")
                                continue   
                        # Sleep between commands to MetaTrader
                        sleep(self._delay)
                   
            #############################
            # SECTION - buy TRADES #
            #############################
            if buy != 0:
                try:
                    # 1 (OP_BUY) or 0 (OP_SELL)
                    globals()[x_num]['_type'] = 1    
                    globals()[x_num]['_lots'] = buy
                    globals()[x_num]['_magic'] = random.getrandbits(6)
                    
                    # Send instruction to MetaTrader
                    _ret_o = self._execution._execute_(globals()[x_num],
                                                       self._verbose,
                                                       self._delay,
                                                       10)
                    # Reset cycle if nothing received
                    if self._zmq._valid_response_(_ret_o) == False:
                        print("Nothing Received")
                        
                finally:
                    #############################
                    # SECTION - GET OPEN TRADES #
                    #############################
                    self._ot = self._reporting._get_open_trades_('{}_Trader'.format(_symbol),
                                                                 self._delay,
                                                                 10)
                    # Reset cycle if nothing received
                    if self._zmq._valid_response_(self._ot) == False:
                        print("Nothing Received")
        
        finally:
            # Release lock
            self._lock.release()
            
        # Sleep between cycles
        sleep(self._delay)
    
    ##########################################################################
    def _stop_(self):        
        for _t in self._traders:      
            # wait for traders to finish.
            _t.join()
        self._zmq._DWX_ZMQ_SHUTDOWN_()
        print("\ntraders finished.\n")
                                                            
    ##########################################################################
