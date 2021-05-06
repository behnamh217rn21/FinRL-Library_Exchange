# -*- coding: utf-8 -*-
"""
    DWX_ZMQ_Reporting.py
    --
    @author: Darwinex Labs (www.darwinex.com)
    
    Copyright (c) 2019 onwards, Darwinex. All rights reserved.
    
    Licensed under the BSD 3-Clause License, you may not use this file except 
    in compliance with the License. 
    
    You may obtain a copy of the License at:    
    https://opensource.org/licenses/BSD-3-Clause
"""

from pandas import DataFrame, to_datetime
from time import sleep

class DWX_ZMQ_Reporting():
    
    def __init__(self, _zmq):
        self._zmq = _zmq
        
    ##########################################################################
    def _get_open_trades_(self, _trader='Trader_SYMBOL', 
                          _delay=0.1, _wbreak=10):
        # Reset data output
        self._zmq._set_response_(None)
        print("1111111111111111111111111111")
        # Get open trades from MetaTrader
        self._zmq._DWX_MTX_GET_ALL_OPEN_TRADES_()
        print("2222222222222222222222222222")

        # While loop start time reference            
        _ws = to_datetime('now')
        
        # While data not received, sleep until timeout
        while self._zmq._valid_response_('zmq') == False:
            sleep(_delay)
            print("3333333333333333333333")
            if (to_datetime('now') - _ws).total_seconds() > (_delay * _wbreak):
                break
        print("444444444444444444444444444")
        # If data received, return DataFrame
        if self._zmq._valid_response_('zmq'):
            _response = self._zmq._get_response_()
            print("00000000000000000000000000000")
            print(_response)
            if ('_trades' in _response.keys()
                and len(_response['_trades']) > 0):
                print("ttttttttttttttttttttttttttttttttttt")
                _df = DataFrame(data=_response['_trades'].values(),
                                index=_response['_trades'].keys())
                print("yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy")
                print(_df)
                print(_response['_trades'].values())
                print(_response['_trades'].keys())
                print("lhhhhhhhhhhhhhhhhhhhhhh")
                return _df[_df['_comment'] == _trader]
            
        # Default
        return DataFrame()
    
    ##########################################################################
    
