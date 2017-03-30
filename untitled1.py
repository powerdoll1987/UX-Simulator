# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 15:33:09 2017

@author: yiran.zhou
"""

print(m_cash)

df = pd.DataFrame()
for (k,v) in m_trades.items():
    df = pd.concat([df, v])

df[df['Size'] < 0]
a = df[df['Size'] < 0]
b = df[df['Size'] > 0]

i = 0
l = len(b.index)
b.index = np.arange(0,l)
while i < len(b.index):
    tic = b.ix[i, 'Ticker']
    s = tic[2:3]
    m = tf.ticker2month(s)
    y = int(tic[3:])
    b.ix[i, 'Date'] = y*100+m
    i += 1