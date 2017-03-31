# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 19:38:58 2017
configuration file
@author: yiran.zhou
"""

import pandas as pd
import numpy as np
from pandas.tseries.offsets import Day, BDay, DateOffset, MonthEnd

# 常数数据结构，需要读入或者设定，但是在程序运行过程中不再改变
m_cash = 0
m_px = dict()
m_spec = pd.DataFrame()
m_dateRng = np.nan
m_blockPeriod = list()
m_uxExpDateLst = np.nan

def initialize():
    # 数据结构初始化
    global m_cash
    global m_px
    global m_spec
    global m_dateRng
    global m_blockPeriod
    global m_uxExpDateLst
    
    # m_cash 设定初始账户金额
    m_cash = 1000000 
    
    # m_px 读入交易产品价格的时间序列
    m_px['Close'] = pd.read_excel('price.xls', sheetname = 'Sheet1 (2)')  
    m_px['Open'] = pd.read_excel('price.xls', sheetname = 'Sheet1 (3)')
    m_px['Low'] = pd.read_excel('price.xls', sheetname = 'Sheet1 (4)')
    m_px['High'] = pd.read_excel('price.xls', sheetname = 'Sheet1 (5)')
    m_px['Close'].set_index('PX_LAST', inplace = True)
    m_px['Open'].set_index('PX_OPEN', inplace = True)
    m_px['Low'].set_index('PX_LOW', inplace = True)
    m_px['High'].set_index('PX_HIGH', inplace = True)
      
    # m_spec 读入产品的规格数据
    m_spec = pd.read_excel('spec.xls', sheetname = 'Sheet1')  
    m_spec.set_index('Ticker', inplace = True)
    
    # m_dateRng 设定交易模拟的时间段
    m_dateRng = pd.date_range('6/1/2004', '3/20/2008', freq = 'B') 
        
    # m_blockPeriod 读入禁止交易的时间段
    bp = pd.read_excel('blockPeriod.xls', sheetname = 'Sheet1')
    i = 0
    while i < len(bp.index):
        startDate = bp.ix[i, 'Date'] + Day(bp.ix[i, 'prev'])
        endDate = bp.ix[i, 'Date'] + Day(bp.ix[i, 'after'])
        bpRng = pd.date_range(startDate, endDate, freq = 'B')
        m_blockPeriod += list(bpRng)
        i += 1
    m_blockPeriod.sort()
    
    # m_uxExpDateLst 生成UX的结算日期
    spxExpDateLst = pd.date_range('1/1/2004', '10/1/2017',freq='WOM-3FRI') # SPX每月第三周的周五到期,比测试时间多3个月
    m_uxExpDateLst = spxExpDateLst - Day(30) # UX在spx option到期前30天到期
    uxExpDateLst2 = list(m_uxExpDateLst)
            
    return 0
        
        
if __name__ == '__main__':
    initialize()
    
    
    
    
    
    
    
    
    
    
    