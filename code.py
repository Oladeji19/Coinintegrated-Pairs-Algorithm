import matplotlib.pyplot
import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
import scipy.optimize as spop
import matplotlib.pyplot as plt

stocks = ['HD', 'LOW']
start ='2019-01-01'
end ='2021-11-15'
window = 252
t_threshold = -2.5
fee = 0.001
data = pd.DataFrame()
returns = pd.DataFrame()
for stock in stocks :
    prices = yf.download(stock, start, end)
    data[stock] = prices['Close']
    returns[stock] = np.append(data[stock][1:].reset_index(drop=True)/data[stock][:-1].reset_index(drop=True)-1, 0)
print(data)


gross_returns = np.array([])
net_returns = np.array ([])
t_s = np.array([])
stock1 = stocks[0]
stock2 = stocks[1]

for t in range(window, len(data)):
    def unit_root(b):
            a = np.average(data[stock2][t-window:t]-b*data[stock1][t-window:t])
            fair_value = a +b*data[stock1][t-window:t]
            diff = np.array(fair_value-data[stock2][t-window:t])
            diff_diff = diff[1:]-diff[:-1]
            reg = sm.OLS(diff_diff, diff[:-1])
            res = reg.fit()
            return res.params[0]/res.bse[0]

    res1 = spop.minimize(unit_root, data[stock2][t]/data[stock1][t], method = 'Nelder-Mead')
    t_opt = res1.fun
    b_opt = float(res1.x)
    a_opt = np.average(data[stock2][t-window:t] - b_opt*data[stock1][t-window:t])
    fair_value = a_opt +b_opt*data[stock1][t]

    if t== window:
            old_signal = 0
    if t_opt > t_threshold:
        signal = 0
        gross_return = 0
    else :
        signal = np.sign(fair_value - data[stock2][t])
        gross_return = signal * returns [stock2][t] - signal*returns[stock1][t]

    fees = fee*abs(signal - old_signal)
    net_return = gross_return - fees
    gross_returns = np.append (gross_returns, gross_return)
    net_returns = np.append(net_returns, net_return)
    t_s = np.append(t_s, t_opt)

    print('day '+str(data.index[t]))
    print('')
    if signal ==0 :
        print('no trading')
    elif signal ==1:
        print('long position on '+stock2+' and short position on'+stock1)
    else :
        print('long position on '+stock1+' and short position on '+stock2)

    print ('gross daily return: '+str(round(gross_return*100,2))+'%')
    print ('net daily return: '+str(round(net_return*100,2))+'%')
    print('cumulative net return so far :'+str(round(np.prod(1+net_returns)*100-100,2))+'%')
    print('')

plt.plot(np.append(1,np.cumprod(1+gross_returns)))
plt.plot(np.append(1,np.cumprod(1+net_returns)))
print("BETA Value for "+stock2+" is: ")
print(b_opt)
print("ALPHA Value for "+stock2+" is: ")
print(a_opt)
ting = gross_returns.std()
ting = ting*np.sqrt(252)
sharpe = (0.104- 0.0161)/ting
print("The SHARPE Ratio for the portfolio is : ")
print(sharpe)

matplotlib.pyplot.show()
