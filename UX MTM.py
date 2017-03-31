# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 11:16:19 2017
计算UX的每日MTM

类似于一个交易模拟器
******************************************************************************************
包含的内容：
交易品种的价格数据 (OCLH各一个df)【dict，dataframe】
交易品种的参数 【dataframe】
每日的Portfolio（cash，头寸大小，成本价，OCLH）【dict，dataframe】
每日的trade（头寸大小，成交价）【dict，dataframe】
交易规则（生成各种交易）
******************************************************************************************
Portfolio的df结构
Ticker(index) | Position /  Cost / Open / Close / Low / High / Category / MktVal
Trade的df结构
number(index) | Ticker / Size / Price / RealizedPnL / customField
Spec的df结构
Ticker(index) | Multiplier / Category
******************************************************************************************
workflow
初始化
根据交易规则生成trade【applyTradeRule】
根据trade生产当天的portfolio【updatePortfolio】
--函数根据trade计算仓位和成本
    --调用【calRealizedPnL】计算实现盈亏
    --调用【updateCash】计算现金变动
--去掉平仓的头寸
--调用【updatePortfolioPrice】填充头寸当天的价格
--调用【calPortfolioMarketValue】计算portfolio每个品种的market value
******************************************************************************************
@author: yiran.zhou
"""


import pandas as pd
import numpy as np
import sys
import config as cf
sys.path.append('..')
import taifook.taifook as tf
import taifook.zigzag_c as zz
import pylab as pl
import matplotlib.pyplot as plt
from pandas.tseries.offsets import Day, BDay, DateOffset, MonthEnd
from scipy import stats


#********************************************specific trading rules*************************************
#  UXA的基础交易规则
# 只有在expire day发生交易, 买入3个月之后的UX，平掉到期的UX
# 每次交易10张
def applyTradeRuleUXA(idx, tdTrades, date, prevPortfolio):
    # 只有在expire day发生交易, 买入3个月之后的UX，平掉到期的UX
    if date in cf.m_uxExpDateLst:
        m = date.month
        y = date.year
        m2 = (date + DateOffset(months = 3)).month
        y2 = (date + DateOffset(months = 3)).year
        expireContract = 'UX' + tf.month2ticker(m) + str(y)
        newContract = 'UX' + tf.month2ticker(m2) + str(y2)
        # 平掉到期的合约
        if expireContract in prevPortfolio.index:
            px = findUXPx(date, expireContract, 'Close')
            expireTd = createTrade(idx, expireContract, 10, px, 0, 'Close')
            idx += 1
            tdTrades = pd.concat([tdTrades, expireTd])
        # 卖出3个月后的合约
        px = findUXPx(date, newContract, 'Close')
        newTd = createTrade(idx, newContract, -10, px, 0, 'Open')
        idx += 1
        tdTrades = pd.concat([tdTrades, newTd])      
    return idx, tdTrades

#  UXA的基础交易规则 2
# 只有在expire day发生交易, 买入3个月之后的UX，平掉到期的UX
# 每次交易张数按照notional AUM/15来交易
def applyTradeRuleUXA2(idx, tdTrades, date, prevPortfolio):
    # 只有在expire day发生交易, 买入3个月之后的UX，平掉到期的UX
    if date in cf.m_uxExpDateLst:
        m = date.month
        y = date.year
        m2 = (date + DateOffset(months = 3)).month
        y2 = (date + DateOffset(months = 3)).year
        expireContract = 'UX' + tf.month2ticker(m) + str(y)
        newContract = 'UX' + tf.month2ticker(m2) + str(y2)
        # 平掉到期的合约
        if expireContract in prevPortfolio.index:
            px = findUXPx(date, expireContract, 'Close')
            numOfct = prevPortfolio.ix[expireContract, 'Position']
            expireTd = createTrade(idx, expireContract, -numOfct, px, 0, 'Close')
            idx += 1
            tdTrades = pd.concat([tdTrades, expireTd])
        # 卖出3个月后的合约
        px = findUXPx(date, newContract, 'Close')
        aum = np.sum(prevPortfolio.ix[:, 'MktVal'])
        numOfct = aum/15/1000/px
        newTd = createTrade(idx, newContract, -numOfct, px, 0, 'Open')
        idx += 1
        tdTrades = pd.concat([tdTrades, newTd])       
    return idx, tdTrades
    
#  UXA的基础交易规则 3
# 只有在expire day发生交易, 买入3个月之后的UX，平掉到期的UX
# 每次交易张数按照notional AUM/15来交易
# 只有在UX3 > 17时卖出
def applyTradeRuleUXA3(idx, tdTrades, date, prevPortfolio):
    # 只有在expire day发生交易, 买入3个月之后的UX，平掉到期的UX
    if date in cf.m_uxExpDateLst:
        m = date.month
        y = date.year
        m2 = (date + DateOffset(months = 3)).month
        y2 = (date + DateOffset(months = 3)).year
        expireContract = 'UX' + tf.month2ticker(m) + str(y)
        newContract = 'UX' + tf.month2ticker(m2) + str(y2)
        # 平掉到期的合约
        if expireContract in prevPortfolio.index:
            px = findUXPx(date, expireContract, 'Close')
            numOfct = prevPortfolio.ix[expireContract, 'Position']
            expireTd = createTrade(idx, expireContract, -numOfct, px, 0, 'Close')
            idx += 1
            tdTrades = pd.concat([tdTrades, expireTd])
        # 卖出3个月后的合约
        px = findUXPx(date, newContract, 'Close')
        if px > 17:
            aum = np.sum(prevPortfolio.ix[:, 'MktVal'])
            numOfct = aum/15/1000/px
            newTd = createTrade(idx, newContract, -numOfct, px, 0, 'Open')
            idx += 1
            tdTrades = pd.concat([tdTrades, newTd])       
    return idx, tdTrades
    
#  UXA的基础交易规则 4
# 买入3个月之后的UX，平掉到期的UX
# 每次交易张数按照notional AUM/15来交易
# 只有在UX3 > 20时卖出，但是一个月的任意一天都可以卖
def applyTradeRuleUXA4(idx, tdTrades, date, prevPortfolio): 
    m = date.month
    y = date.year
    m2 = (date + DateOffset(months = 3)).month
    y2 = (date + DateOffset(months = 3)).year
    expireContract = 'UX' + tf.month2ticker(m) + str(y)
    newContract = 'UX' + tf.month2ticker(m2) + str(y2)
    # 到合约日平掉到期的合约
    if date in cf.m_uxExpDateLst:
        if expireContract in prevPortfolio.index:
            px = findUXPx(date, expireContract, 'Close')
            numOfct = prevPortfolio.ix[expireContract, 'Position']
            expireTd = createTrade(idx, expireContract, -numOfct, px, 0, 'Close')
            idx += 1
            tdTrades = pd.concat([tdTrades, expireTd])
    # 卖出3个月后的合约
    if newContract not in prevPortfolio.index:
        px = findUXPx(date, newContract, 'Close')
        if px > 20:
            aum = np.sum(prevPortfolio.ix[:, 'MktVal'])
            numOfct = aum/15/1000/px
            newTd = createTrade(idx, newContract, -numOfct, px, 0, 'Open')
            idx += 1
            tdTrades = pd.concat([tdTrades, newTd])       
    return idx, tdTrades
    
#  UXA的基础交易规则 3
# 只有在expire day发生交易, 买入3个月之后的UX，平掉到期的UX
# 每次交易张数按照notional AUM/15来交易
# 在block period里面，平掉所有仓位，并且不交易
def applyTradeRuleUXA5(idx, tdTrades, date, prevPortfolio):
    # 只有在expire day发生交易, 买入3个月之后的UX，平掉到期的UX
    noTrade = False
    if date in cf.m_blockPeriod:
        noTrade = True
        idx, tdTrades = closeAllPos(idx, tdTrades, date, prevPortfolio)
 
    m = date.month
    y = date.year
    m2 = (date + DateOffset(months = 3)).month
    y2 = (date + DateOffset(months = 3)).year
    expireContract = 'UX' + tf.month2ticker(m) + str(y)
    newContract = 'UX' + tf.month2ticker(m2) + str(y2)
    if date in cf.m_uxExpDateLst:
        # 平掉到期的合约
        if expireContract in prevPortfolio.index:
            px = findUXPx(date, expireContract, 'Close')
            numOfct = prevPortfolio.ix[expireContract, 'Position']
            expireTd = createTrade(idx, expireContract, -numOfct, px, 0, 'Close')
            idx += 1
            tdTrades = pd.concat([tdTrades, expireTd])
        # 卖出3个月后的合约
        px = findUXPx(date, newContract, 'Close')
        if noTrade == False:
            aum = np.sum(prevPortfolio.ix[:, 'MktVal'])
            numOfct = aum/15/1000/px
            newTd = createTrade(idx, newContract, -numOfct, px, 0, 'Open')
            idx += 1
            tdTrades = pd.concat([tdTrades, newTd])       
    return idx, tdTrades   
    
#  UXA的基础交易规则 6
# 只有在expire day发生交易, 买入3个月之后的UX，平掉到期的UX
# 每次交易张数按照notional AUM/15来交易
# 在block period里面，平掉所有仓位，并且不交易
# 只有在UX3 > 17时卖出
def applyTradeRuleUXA6(idx, tdTrades, date, prevPortfolio):
    # 只有在expire day发生交易, 买入3个月之后的UX，平掉到期的UX
    noTrade = False
    if date in cf.m_blockPeriod:
        noTrade = True
        idx, tdTrades = closeAllPos(idx, tdTrades, date, prevPortfolio)
     
    m = date.month
    y = date.year
    m2 = (date + DateOffset(months = 3)).month
    y2 = (date + DateOffset(months = 3)).year
    expireContract = 'UX' + tf.month2ticker(m) + str(y)
    newContract = 'UX' + tf.month2ticker(m2) + str(y2)
    if date in cf.m_uxExpDateLst:
        # 平掉到期的合约
        if expireContract in prevPortfolio.index:
            px = findUXPx(date, expireContract, 'Close')
            numOfct = prevPortfolio.ix[expireContract, 'Position']
            expireTd = createTrade(idx, expireContract, -numOfct, px, 0, 'Close')
            idx += 1
            tdTrades = pd.concat([tdTrades, expireTd])
            
    # 卖出3个月后的合约
    if newContract not in prevPortfolio.index:
        px = findUXPx(date, newContract, 'Close')
        if noTrade == False and px > 17:
            aum = np.sum(prevPortfolio.ix[:, 'MktVal'])
            numOfct = aum/15/1000/px
            newTd = createTrade(idx, newContract, -numOfct, px, 0, 'Open')
            idx += 1
            tdTrades = pd.concat([tdTrades, newTd]) 
      
    return idx, tdTrades 
#********************************************specific trading rules*************************************

# 应用自定义的交易规则
def applyTradeRule(date, prevPortfolio):
    idx = 0 # 每增加一笔trade，index + 1
    tdTrades = pd.DataFrame()
    idx, tdTrades = applyTradeRuleUXA(idx, tdTrades, date, prevPortfolio)
    return tdTrades

# 平掉所有头寸
def closeAllPos(idx, tdTrades, date, prevPortfolio):
    i = 0    
    while i < len(prevPortfolio.index):
        if prevPortfolio.index[i] != 'Cash':
            px = findPx(date, prevPortfolio.index[i], 'Close')
            numOfct = prevPortfolio.ix[i, 'Position']
            newTd = createTrade(idx, prevPortfolio.index[i], -numOfct, px, 0, 'Close')
            idx += 1
            tdTrades = pd.concat([tdTrades, newTd])    
        i += 1
    return idx, tdTrades
    
# 找UX合约的价格, 合约格式为UXF2017，价格数据是UX3,UX2...
def findUXPx(date, uxCtr, OCLH):
    expireMonth = str(tf.ticker2month(uxCtr[2]))
    expireYear = uxCtr[3:]
    d1 = pd.Timestamp(expireYear + '-' + expireMonth + '-' + '1')
    d2 = d1 + MonthEnd(1)
    expireDay = cf.m_uxExpDateLst[(cf.m_uxExpDateLst <= d2) & (cf.m_uxExpDateLst >= d1)][0]
    numOfCtr = len(cf.m_uxExpDateLst[(cf.m_uxExpDateLst <= expireDay) & (cf.m_uxExpDateLst >= date)]) #判断是UX是第几个合约
    pxTicker = 'UX' + str(numOfCtr) + ' Index'
    return cf.m_px[OCLH].ix[date, pxTicker]

# 根据给入的ticker名称找spec里面的ticker名，例如通过UXJ7 Index找UXA Index
def findSpecTicker(tic):
    if tic[:2] == 'UX':
        specTic = 'UXA Index'
    else: 
        specTic = tic
    return specTic
    
# 根据给入的ticker名称找spec里面的Category名， 例如通过UXJ7 Index找future
def findSpecCategory(tic):
    specTic = findSpecTicker(tic)
    return cf.m_spec.ix[specTic, 'Category']

# 根据品种计算market value
def calPositionMktVal(ticker, category, pos, price, cost):
    mv = 0
    if category == 'Cash':
        mv = pos
    elif category == 'Future':
        mul = cf.m_spec.ix[findSpecTicker(ticker), 'Multiplier']
        mv = (price - cost)* pos * mul
    elif category == 'Equity':
        mv = pos * price
    elif category == 'FX':
        mv = (price - cost) * pos
    elif category == 'Option':
        mul = cf.m_spec.ix[findSpecTicker(ticker), 'Multiplier']
        mv = (pos * price) * mul
    else:
        mv = pos
    return mv

# 计算portfolio里面每个品种收盘时的market value
def calPortfolioMarketValue(portfolio):
    i = 0
    while i < len(portfolio.index):
        ticker = portfolio.index[i]
        category = portfolio.ix[i, 'Category']
        price = portfolio.ix[i, 'Close']
        cost = portfolio.ix[i, 'Cost']
        pos = portfolio.ix[i, 'Position']
        portfolio.ix[i, 'MktVal'] = calPositionMktVal(ticker, category, pos, price, cost)
             
        i += 1
    return 0

# 生产trade的dataframe
def createTrade(idx, tic, size, px, realizedPnL, customField):
    cols = ['Ticker' , 'Size', 'Price', 'RealizedPnL', 'customField']    
    data = {'Ticker': tic, 'Size': size, 'Price':px, 'RealizedPnL':realizedPnL, 'customField':customField}
    return pd.DataFrame(data, index = [idx], columns = cols)

# 找品种那天的价格
def findPx(date, tic, OCLH):
    px = np.nan
    if tic[:2] == 'UX':    
        px = findUXPx(date, tic, OCLH)
    else:
        px = cf.m_px[OCLH].ix[date, tic]
    return px
    
# 根据交易更新portfolio, 同时更新trade的realized PnL
def updatePortfolio(tdTrade, prevPortfolio, date): 
    tdPortfolio = prevPortfolio.copy()
    i = 0    
    while i < len(tdTrade.index): # 对每笔交易进行处理
        tic = tdTrade.ix[i, 'Ticker']
        size = tdTrade.ix[i, 'Size']
        price = tdTrade.ix[i, 'Price']
        # portfolio里面没有这个品种，就一种情况
        if tic not in tdPortfolio.index: 
            data = {'Position':size, 'Cost':price, 'Category':findSpecCategory(tic)}
            tmp = pd.DataFrame(data, index = [tic])
            cols = tdPortfolio.columns
            tdPortfolio = pd.concat([tdPortfolio, tmp])
            tdPortfolio = tdPortfolio[cols]
            updateCash(tdTrade, i, tdPortfolio)
        # portfolio里面有这个品种，分两种情况
        else: 
            # 加仓，就一种情况
            if size * tdPortfolio.ix[tic, 'Position'] > 0: 
                tdPortfolio.ix[tic, 'Cost'] = (tdPortfolio.ix[tic, 'Cost'] * tdPortfolio.ix[tic, 'Position']\
                + size * price) / (tdPortfolio.ix[tic, 'Position'] + size)
                tdPortfolio.ix[tic, 'Position'] += size
                updateCash(tdTrade, i, tdPortfolio)
            # 减仓，分两种情况
            else: 
                # 减仓或者平仓
                if abs(size) <= abs(tdPortfolio.ix[tic, 'Position']):
                    calRealizedPnL(tdTrade, i, tdPortfolio) # 先计算realized PnL，再计算cash
                    updateCash(tdTrade, i, tdPortfolio)
                    tdPortfolio.ix[tic, 'Position'] += size
                # 反向开仓
                else:
                    calRealizedPnL(tdTrade, i, tdPortfolio) # 先计算realized PnL，再计算cash
                    updateCash(tdTrade, i, tdPortfolio)                   
                    tdPortfolio.ix[tic, 'Position'] += size
                    tdPortfolio.ix[tic, 'Cost'] = price
        i += 1          
    # 去掉平仓的头寸
    j = 0
    while j < len(tdPortfolio.index):
        if tdPortfolio.ix[j, 'Position'] == 0: # 如果被平仓了，去掉这个品种
            tdPortfolio.drop(tdPortfolio.index[j], inplace = True)
        else:
            j += 1
    # 更新头寸的当天价格，计算其收盘的martket value
    updatePortfolioPrice(tdPortfolio, date)                                    
    calPortfolioMarketValue(tdPortfolio)
    return tdPortfolio

   
# 计算交易的realized PnL， 每笔交易计算一次
def calRealizedPnL(tdTrade, tIdx, tdPortfolio):
    tic = tdTrade.ix[tIdx, 'Ticker']
    size = tdTrade.ix[tIdx, 'Size']
    price = tdTrade.ix[tIdx, 'Price']  
    category = findSpecCategory(tic)
    cost = tdPortfolio.ix[tic, 'Cost']
    pos = tdPortfolio.ix[tic, 'Position']
    # 如果是反手做多或做空, 只有原来position的量需要用来计算realized PnL
    if abs(size) > abs(pos) :
        size = -pos
    # 分交易品种讨论
    if category == 'Future' or category == 'Option':
        mul = cf.m_spec.ix[findSpecTicker(tic), 'Multiplier']
        pnl = (price - cost) * -size * mul
    elif category == 'FX' or category == 'Equity':
        pnl = (price - cost) * -size
    else:
        pnl = (price - cost) * -size
    tdTrade.ix[tIdx, 'RealizedPnL'] = pnl
    return 0
    
# 根据交易更新cash, 每笔交易更新一次
def updateCash(tdTrade, tIdx, tdPortfolio):   
    tic = tdTrade.ix[tIdx, 'Ticker']
    size = tdTrade.ix[tIdx, 'Size']
    price = tdTrade.ix[tIdx, 'Price'] 
    pnl = tdTrade.ix[tIdx, 'RealizedPnL'] 
    category = findSpecCategory(tic)
    # 分交易品种讨论
    if category == 'Future' or category == 'FX':
        tdPortfolio.ix['Cash', 'Position'] += pnl
    elif category == 'Option':
        mul = cf.m_spec.ix[findSpecTicker(tic), 'Multiplier']
        tdPortfolio.ix['Cash', 'Position'] += -size * price * mul
    elif category == 'Equity':
        tdPortfolio.ix['Cash', 'Position'] += -size * price
    else:
        tdPortfolio.ix['Cash', 'Position'] += -size * price
    return 0 
    
# 根新portfolio每个品种的价格数据 
def updatePortfolioPrice(tdPortfolio, date):
    i = 0
    while i < len(tdPortfolio.index):
        tic = tdPortfolio.index[i]
        if tic != 'Cash':
            tdPortfolio.ix[i, 'Open'] = findPx(date, tic, 'Open')
            tdPortfolio.ix[i, 'Close'] = findPx(date, tic, 'Close')
            tdPortfolio.ix[i, 'Low'] = findPx(date, tic, 'Low')
            tdPortfolio.ix[i, 'High'] = findPx(date, tic, 'High')      
        i += 1
    return 0

# 主程序
if __name__ == '__main__':
    
    # 配置程序，设定初始金额，模拟时间段，读入交易价格，产品信息等常量
    cf.initialize()
    
    # 基础数据结构（变量）
    m_Portfolio = dict()
    m_trades = dict()
    m_PortfolioMTM = pd.DataFrame()
    
    # 初始化基础数据结构
    data = {'Ticker':'Cash', 'Position':cf.m_cash, 'Cost':1, 'Open':1, 'Close':1, \
    'Low':1, 'High':1, 'Category':'Cash', 'MktVal':cf.m_cash}
    startPortfolio = pd.DataFrame(data, index = [0], columns = ['Ticker', 'Position', 'Cost', \
    'Open', 'Close', 'Low', 'High', 'Category', 'MktVal'])
    startPortfolio.set_index('Ticker', inplace = True)
    m_Portfolio[cf.m_dateRng[0] - BDay(1)] = startPortfolio # 建立最开始的资产组合（只有现金）
    m_PortfolioMTM = pd.DataFrame(index = cf.m_dateRng, columns = ['Open', 'Close','Low','High'])
     
    # 循环计算每日头寸和交易
    i = 0
    while i < len(cf.m_dateRng):
        date = cf.m_dateRng[i]
        prevPortfolio = m_Portfolio[date - BDay(1)] #前一天的Portfolio
        # 计算当天的trade和portfolio
        tdTrades =  applyTradeRule(date, prevPortfolio) 
        tdPortfolio = updatePortfolio(tdTrades, prevPortfolio, date)
        # 把当天的结果加入数据字典中
        m_Portfolio[date] = tdPortfolio
        if len(tdTrades.index) != 0:
            m_trades[date] = tdTrades
        i += 1
    
    # 画出portfoli的PnL
    i = 0
    while i < len(cf.m_dateRng):
        date = cf.m_dateRng[i]
        tdPortfolio = m_Portfolio[date]
        
        k = 0
        m_PortfolioMTM.ix[i, 'Open'] = 0
        m_PortfolioMTM.ix[i, 'Close'] = 0
        m_PortfolioMTM.ix[i, 'Low'] = 0
        m_PortfolioMTM.ix[i, 'High'] = 0
        while k < len(tdPortfolio.index):
            category = tdPortfolio.ix[k, 'Category']
            cost = tdPortfolio.ix[k, 'Cost']
            pos = tdPortfolio.ix[k, 'Position']
            ticker = tdPortfolio.index[k]
            m_PortfolioMTM.ix[i, 'Open'] += calPositionMktVal(ticker, category, \
            pos, tdPortfolio.ix[k, 'Open'], cost)
            m_PortfolioMTM.ix[i, 'Close'] += calPositionMktVal(ticker, category, \
            pos, tdPortfolio.ix[k, 'Close'], cost)
            m_PortfolioMTM.ix[i, 'Low'] += calPositionMktVal(ticker, category, \
            pos, tdPortfolio.ix[k, 'Low'], cost)
            m_PortfolioMTM.ix[i, 'High'] += calPositionMktVal(ticker, category, \
            pos, tdPortfolio.ix[k, 'High'], cost)
            
            k += 1
        i += 1
    
    lst = tf.df2OHLC(m_PortfolioMTM)
    tf.darwCandleChart(lst)
    
    