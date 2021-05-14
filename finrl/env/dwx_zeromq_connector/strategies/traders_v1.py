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
from finrl.env.dwx_zeromq_connector.strategies.base.api.DWX_ZeroMQ_Connector_v2_0_1_RC8 import _DWX_ZMQ_CLEANUP_

from threading import Thread, Lock
from time import sleep
import random
import pandas as pd


class t_class(DWX_ZMQ_Strategy):
    def __init__(self, _name="ONLINE_TRADERS",
                 _symbols=['#INTC', '#AAPL'],
                 _delay=0.2,
                 _broker_gmt=3,
                 _verbose=False
                 ):
        
        # call DWX_ZMQ_Strategy constructor and passes itself as data processor for handling
        # received data on PULL and SUB ports 
        super().__init__(_name,
                         _symbols,       # Empty symbol list (not needed for this example)
                         _broker_gmt,
                         [],             # Registers itself as handler of pull data via self.onPullData()
                         [],             # Registers itself as handler of pull data via self.onPullData()
                         _verbose)
        
        # This strategy's variables
        self._delay = _delay
        self._traders = []
        self._market_open = True
        self._verbose = _verbose
        
        self._symbols = _symbols
        
        # lock for acquire/release of ZeroMQ connector
        self._lock = Lock()
        
    ##########################################################################
    def _run_(self, sells, buys):
        """
        Logic:
            For each symbol in self._symbols:
                1) Calculate Open P&L every second
                2) Plot Open P&L in real-time
                3) Lot size per trade = 0.01
                4) SL/TP = 10 pips each
        """
        # Launch traders!
        for index, _symbol in enumerate(self._symbols):
            _t = Thread(name="{}_Trader".format(_symbol),
                        target=self._trader_, 
                        args=(_symbol, 
                              sells[index], buys[index]))
            _t.daemon = True
            _t.start()
            
            print('[{}_Trader] Alright !'.format(_symbol))
            self._traders.append(_t)
        
    ##########################################################################
    def _trader_(self, _symbol, sell, buy):
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
        print("sell: {}; buy: {}".format(sell, buy))
        
        while self._market_open:
            try:
                # Acquire lock
                self._lock.acquire()
            
                _ot = self._reporting._get_open_trades_(self._delay, 10)
                        
                # Reset cycle if nothing received
                if self._zmq._valid_response_(_ot) == False:
                    print("Nothing Received")
                    continue  
                
                print("trade counter: {}".format(_ot.shape[0]))
                print(_ot)
                
                ###############################
                # SECTION - SELL TRADES #
                ###############################
                if _ot.shape[0] > 0:
                    if sell != 0:
                        for i in (_ot.loc[_ot["_symbol"] == _symbol].index):
                            if sell < _ot["_lots"].loc[_ot.index == i]:
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
                                    break   
                        
                            elif sell == _ot["_lots"].loc[_ot.index == i]:
                                _ret_c = self._execution._execute_({'_action': 'CLOSE',
                                                                    '_ticket': i,
                                                                    'size': sell},
                                                                   self._verbose,
                                                                   self._delay,
                                                                   10)
                                # Reset cycle if nothing received
                                if self._zmq._valid_response_(_ret_c) == False:
                                    print("Nothing Received")
                                    break
                            
                            else:
                                sell = sell - _ot["_lots"].loc[_ot["_symbol"] == _symbol]
                                _ret_c = self._execution._execute_({'_action': 'CLOSE',
                                                                    '_ticket': i,
                                                                    'size': _ot["_lots"].loc[_ot.index == i]},
                                                                   self._verbose,
                                                                   self._delay,
                                                                   10)
                                # Reset cycle if nothing received
                                if self._zmq._valid_response_(_ret_c) == False:
                                    print("Nothing Received")
                                    break 
                                
                            # Sleep between commands to MetaTrader
                            sleep(self._delay)
                   
                #############################
                # SECTION - buy TRADES #
                #############################
                if buy != 0:
                    _random_int  = random.randint(1, 999)
                    value = "_default_order_{}".format(str(_random_int))
                    x_num = 'value'
                    # Note: Just for this example, only the Order Type is dynamic.
                    globals()[x_num] = self._zmq._generate_default_order_dict()
                    globals()[x_num]['_symbol'] = _symbol
                    globals()[x_num]['_SL'] = 100
                    globals()[x_num]['_TP'] = 100
                    globals()[x_num]['_comment'] = '{}_Trader'.format(_symbol)
                    # 0 (OP_BUY) or 1 (OP_SELL)
                    globals()[x_num]['_type'] = 0    
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
                        break
        
            finally:
                # Release lock
                self._lock.release()

            # Sleep between cycles
            sleep(self._delay)
    
    
    ##########################################################################
    def _stop_(self):        
        self._market_open = False 
        
        for _t in self._traders:
            # Setting _market_open to False will stop each "trader" thread
            # from doing anything more. So wait for them to finish.
            _t.join()
        
        #_DWX_ZMQ_CLEANUP_()
        self._zmq._DWX_ZMQ_SHUTDOWN_T_()
        print("\ntraders finished.\n")
                                                            
    ##########################################################################
