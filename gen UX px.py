# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 15:10:13 2017
根据VIX的数据生成UX的推算价格
@author: yiran.zhou
"""

import pandas as pd
import numpy as np
from pandas.tseries.offsets import Day, BDay, DateOffset, MonthEnd



if __name__ == '__main__':
    # 读入VIX数据
    vixPx = pd.read_excel('vix.xls', sheetname = 'Sheet1')
    vixPx.set_index('VIX Index', inplace = True)
    
    # UX合约理论上的到期日
    spxExpDateLst = pd.date_range('1/1/1992', '10/1/2017',freq='WOM-3FRI') # SPX每月第三周的周五到期,比测试时间多3个月
    uxExpDateLst = spxExpDateLst - Day(30) # UX在spx option到期前30天到期
    
    dateRng = pd.date_range('1/1/1993', '3/20/2017', freq = 'B') 
    
    # UX的数据结构
    cols = vixPx.columns
    ux1Px = pd.DataFrame(index = vixPx.index, columns = cols)
    ux2Px = pd.DataFrame(index = vixPx.index, columns = cols)    
    ux3Px = pd.DataFrame(index = vixPx.index, columns = cols)    
    uxPxLst = [ux1Px, ux2Px, ux3Px]
    
    # UX-VIX的spread和VIX的beta，依次是UX1, UX2, UX3
    beta = [-0.109, -0.239, -0.321]
    alpha = [2.766, 6.037, 8.115]
    
    i = 0
    while i < len(vixPx.index):
        date = vixPx.index[i]

        nextExpDate = uxExpDateLst[uxExpDateLst >= date][0]
        prevExpDate = uxExpDateLst[uxExpDateLst < date][-1]
        days = np.busday_count(prevExpDate, nextExpDate)
        passDays = np.busday_count(prevExpDate, date)
        intercept = 0.5
        
        timedecay = intercept - 1 * passDays / days
        
        j = 0
        while j < 3:
            uxPxLst[j].ix[i, 'PX_LAST'] = vixPx.ix[i, 'PX_LAST'] * (1 + beta[j]) + alpha[j] + timedecay
            uxPxLst[j].ix[i, 'PX_OPEN'] = vixPx.ix[i, 'PX_OPEN'] * (1 + beta[j]) + alpha[j] + timedecay
            uxPxLst[j].ix[i, 'PX_LOW'] = vixPx.ix[i, 'PX_LOW'] * (1 + beta[j]) + alpha[j] + timedecay
            uxPxLst[j].ix[i, 'PX_HIGH'] = vixPx.ix[i, 'PX_HIGH'] * (1 + beta[j]) + alpha[j] + timedecay
            j += 1
        
        i += 1
    
    