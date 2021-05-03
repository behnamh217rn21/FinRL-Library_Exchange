# -*- coding: utf-8 -*-
"""
    DWX_ZMQ_Strategy.py
    --
    @author: Darwinex Labs (www.darwinex.com)
    
    Copyright (c) 2019 onwards, Darwinex. All rights reserved.
    
    Licensed under the BSD 3-Clause License, you may not use this file except 
    in compliance with the License. 
    
    You may obtain a copy of the License at:    
    https://opensource.org/licenses/BSD-3-Clause
"""

from finrl.env.dwx_zeromq_connector.strategies.base.api.DWX_ZeroMQ_Connector_v2_0_1_RC8 import DWX_ZeroMQ_Connector
from finrl.env.dwx_zeromq_connector.strategies.base.modules.DWX_ZMQ_Execution import DWX_ZMQ_Execution
from finrl.env.dwx_zeromq_connector.strategies.base.modules.DWX_ZMQ_Reporting import DWX_ZMQ_Reporting

class DWX_ZMQ_Strategy(object):
    
    def __init__(self, 
                 _name="DEFAULT_STRATEGY",      # Name 
                 _symbols=[('EURUSD',0.01),     # List of (Symbol,Lotsize) tuples
                           ('AUDNZD',0.01),
                           ('NDX',0.10),
                           ('UK100',0.1),
                           ('GDAXI',0.01),
                           ('XTIUSD',0.01),
                           ('SPX500',1.0),
                           ('STOXX50E',0.10),
                           ('XAUUSD',0.01)],
                 _broker_gmt=0,                 # Darwinex GMT offset
                 _pulldata_handlers = [],       # Handlers to process data received through PULL port.
                 _subdata_handlers = [],        # Handlers to process data received through SUB port.
                 _verbose=False):               # Print ZeroMQ messages
                 
        self._name = _name
        self._symbols = _symbols
        self._broker_gmt = _broker_gmt
        
        # Not entirely necessary here.
        import random
        n = random.randint(1, 999)
        value = "self._zmq_{}".format(str(n))
        x = 'value'
        print("1111111111111111111111111")
        print(value)
        file = open("f.txt","w")
        file.writelines(value)
        file.close()
        globals()[x] = DWX_ZeroMQ_Connector(_pulldata_handlers=_pulldata_handlers,
                                            _subdata_handlers=_subdata_handlers,
                                            _verbose=_verbose)
        
        # Modules
        self._execution = DWX_ZMQ_Execution(globals()[x])
        self._reporting = DWX_ZMQ_Reporting(globals()[x])
        
    ##########################################################################
    
    def _run_(self):
        
        """
        Enter strategy logic here
        """
         
    ##########################################################################
