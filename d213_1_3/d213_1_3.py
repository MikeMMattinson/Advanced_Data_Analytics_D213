#!/usr/bin/env python
# coding: utf-8

# # D213 Task 1 Rev 3 - Mattinson

# ## Update & install
# pip install pmdarima
!pip install pmdarima
# ## import packages & read data

# ### import packages

# In[1]:


#import basic libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy import signal


# In[2]:


# import and configure matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300


# In[3]:


# import required model libraries
from statsmodels.tsa.stattools import acf, pacf
#from statsmodels.tsa.arima.model import ARIMA
import statsmodels.tsa.stattools as ts
#from statsmodels.tsa.arima_model import ARIMA2
from statsmodels.tsa.arima.model import ARIMA


# In[4]:


# Where to save figures and model diagrams
# adapted code (Geron, 2019)
import os
IMAGES_PATH = os.path.join(".", "figures")
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print('Saving figure: {}'.format(fig_id))
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, 
        dpi=resolution, bbox_inches = "tight")
    
MODEL_PATH = os.path.join(".", "models")
os.makedirs(MODEL_PATH, exist_ok=True)   

TABLE_PATH = os.path.join(".", "tables")
os.makedirs(TABLE_PATH, exist_ok=True) 

DATA_PATH = os.path.join(".", "data")
os.makedirs(DATA_PATH, exist_ok=True)


# ### read time data

# In[5]:


def read_time_series(file: str, index: str, start_date=None, freq='d') -> pd.DataFrame():
    """create dataframe of time series data
    Author: Mike Mattinson
    Date: June 22, 2022
    
    Parameters
    ----------
    file: str
       filename of time series data
    index: str
       column name of date index
    start_date: datetime
       (optional) if using specific start date
    freq: str
       (default) '24H' 24-hour increments
    
    Returns
    -------
    tsdf: pd.DataFrame()
       time series dataframe
    
    """
    
    # read and initialize index
    tsdf =  pd.read_csv(file)
    tsdf.set_index(index, inplace=True)
    
    # re-index on specific optional start_date
    index_label = 'Date'
    if(start_date is not None):
        tsdf[index_label] = (pd.date_range(
            start=start_date,
            periods=tsdf.shape[0],
            freq=freq))
        tsdf.set_index(index_label, inplace=True)
        tsdf['Year'] = tsdf.index.year
        tsdf['Month'] = tsdf.index.month
        #tsdf['Weekday Name'] = tsdf.index.weekday_name
                    
    # print out summary
    print(tsdf.info())
    print(tsdf.shape)
    print(tsdf.sample(5, random_state=0))
    
    return tsdf # time series dataframe


# In[6]:


# read time series data from CSV file
from datetime import datetime
df =  read_time_series(
    file='data/teleco_time_series.csv', 
    index='Day', freq='d', 
    start_date=datetime(2020,1,1)
)


# ## clean & explore data

# In[7]:


# show sample from dataframe
n_rows=10
df.sample(n_rows, random_state=0)


# In[8]:


# drop zero values
df= df[df['Revenue'] != 0]


# In[9]:


# descripe numerical data
df.describe()


# In[10]:


#find rolling mean of previous n periods
n_days = 30
df['rolling_mean'] = df['Revenue'].rolling(window=n_days).mean()
df['rolling_std'] = df['Revenue'].rolling(window=n_days).std()


# In[11]:


#check missing data
df.isnull().any()


# ### export cleaned data

# In[12]:


# export cleaned data to file
df.to_csv('tables\cleaned.csv', index=True, header=True)
print(df.info())
print(df.shape)


# ### revenue plot with polyfit regression

# https://stackoverflow.com/questions/39801403/how-to-derive-equation-from-numpys-polyfit
!pip install sympy
# In[13]:


# equation of poly fit 
from sympy import S, symbols, printing
x = pd.Series(range(df.shape[0]))
y = df['Revenue'].values
n_deg = 3
p = np.polyfit(x, y, deg=n_deg)
f = np.poly1d(p)
e = symbols("x")
poly = sum(S("{:6.7f}".format(v))*e**i for i, v in enumerate(p[::-1]))
eq_latex = printing.latex(poly)
print(p)
print(poly) # won't include zero terms


# In[14]:


# visualize raw revenue data
x = pd.Series(df.index.values) # if using date
x2 = pd.Series(range(df.shape[0])) # if using date index
fig, ax = plt.subplots(2,1, figsize = (9, 8), sharex=True, sharey=True)
ax[0].plot(x, df.Revenue, 'lightblue', label='Revenue')
ax[1].plot(x, df.Revenue, 'lightblue', label='Revenue')
ax[0].plot(x,f(x2),"b", label='Poly fit (deg=' + str(n_deg) + ')')
ax[0].legend()
ax[0].set_title('Revenue ($M)\nwith Poly Fit')
ax[1].plot(x,df['rolling_mean'], "red",
         label=str(n_days) + '-d Rolling Mean')
ax[1].plot(x,df['rolling_std'], "black", 
         label=str(n_days) + '-d Rolling Std')
ax[1].set_title('Revenue ($M)\n30-d Rolling Mean\n30-d Rolling Std Dev')
ax[1].legend()
import matplotlib.dates as mdates
ax[1].xaxis.set_major_locator(mdates.YearLocator())
ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax[1].xaxis.set_minor_locator(mdates.MonthLocator())
ax[1].xaxis.set_minor_formatter(mdates.DateFormatter('\n%b'))
fig.supxlabel('Date') # common x label
fig.supylabel('Revenue ($M)') # common y label
#plt.gcf().text(0, -.1, "${}$".format(eq_latex), fontsize=14)
title = 'Revenue ($M)'
save_fig(title) 


# Generally, trending up and not stationary. Also, does not appear to have seasonality.

# ## diff data - make stationary

# ### dickey-fuller - on raw data, non-stationary data

# https://www.statsmodels.org/dev/generated/statsmodels.tsa.stattools.adfuller.html
# 
# https://machinelearningmastery.com/time-series-data-stationary-python/ 
# 
# https://www.machinelearningplus.com/time-series/augmented-dickey-fuller-test/
# 
# https://www.quora.com/What-is-an-Augmented-Dickey-Fuller-test

# In[15]:


import statsmodels.tsa.stattools as ts
def dickey_fuller(
    array: np.array, 
    critical=0.05,
    stats=False) -> float:
    """return p-value of augmented dickey-fullter test
    Author: Mike Mattinson
    Date: June 29, 2022
    
    Parameters
    ----------
    array: np.array # array-like
       array of values to be evaluated
    critical: float (default=0.05)
       critical value
    stats: bool (default=False)
        include stats is output or not
    
    Returns
    -------
    pvalue: float
        p-value

    """
    result = ts.adfuller(array, autolag='AIC')
    pvalue = result[1]
    
    if(stats):
        print('ADF Statistic: %f' % result[0])
        print('p-value: %f' % pvalue)
        print('Critical Values:')
        for key, value in result[4].items():
            print('\t%s: %.3f' % (key, value))
    
    if pvalue <= critical:
        print('Reject H0, data is stationary.')
    else:
        print('Accept H0, data is non-stationary.')
    
    return pvalue


# In[16]:


# augmented dickey-fuller
dickey_fuller(df['Revenue'].values, stats=True)


# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.diff.html

# In[17]:


"""Calculates difference of Dataframe element compared with another 
element in the Dataframe (default is element in previous row)."""
df_stationary = df.diff(periods=1,axis=0).dropna()
print(df_stationary.info())
print(df_stationary.shape)
#print(df_stationary.describe())


# ### dickey-fuller - on differenced data

# In[18]:


# augmented dickey-fuller
dickey_fuller(df_stationary['Revenue'].values,
        stats=True)


# ### export stationary data

# In[19]:


# export stationary data to file
df_stationary.to_csv('tables\stationary.csv', index=True, header=True)
print(df_stationary.info())
print(df_stationary.shape)


# ## train test split

# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
#setup training and test data 80/20
test_size = int(.20 * df_stationary.shape[0])  # last 20%
train, test = train_test_split(df_stationary, 
            test_size=test_size, shuffle=False)
print('training: {}'.format(train.shape))
print('testing: {}'.format(test.shape))
# In[20]:


# use last 30 days for testing
train = df.iloc[:-30]
test = df.iloc[-30:]
print('training: {}'.format(train.shape))
print('testing: {}'.format(test.shape))


# In[21]:


test.info()


# In[22]:


test.describe()


# In[23]:


train.info()


# In[24]:


train.describe()


# ## spectral density

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.periodogram.html
# 
# https://www.geeksforgeeks.org/plot-the-power-spectral-density-using-matplotlib-python/
# 
# https://online.stat.psu.edu/stat510/lesson/12/12.1
# 
# https://web.stanford.edu/class/earthsys214/notes/series.html
# 

# In[25]:


from scipy import signal
def sd_plot(data, target, ax, i: int, title: str) -> None:
    f, Pxx = signal.periodogram(data[target])
    ax[i].semilogy(f, Pxx, label='data')
    ax[i].set_title(title)
    ax[i].hlines(y=10e-1, xmin=0, xmax=0.5, lw=1, 
                 linestyles='--', color='r', label='10e-1')
    ax[i].set_ylim([1e-6, 1e2])
    ax[i].legend()
    return None


# In[26]:


# plot spectral density
fig, ax = plt.subplots(3,1, figsize = (9, 8), sharex=True, sharey=True)
sd_plot(data=df, target='Revenue', ax=ax, i=0,
         title='Spectral Density\nRaw data')
sd_plot(data=df_stationary, target='Revenue', ax=ax, i=1,
         title='Spectral Density\nStationary data')
sd_plot(data=train, target='Revenue', ax=ax, i=2,
         title='Spectral Density\nTraining data')
title = 'Spectral Density'
fig.supxlabel('Frequency') # common x label
fig.supylabel('Spectral Density') # common y label
save_fig(title) 


# ## acf & pacf plots

# In[27]:


from statsmodels.tsa.stattools import acf
def acf_plot(data, target, ax, i: int, conf: bool, title: str) -> None:
    
    acf_values = acf((data[target].values))
    acf_df = pd.DataFrame([acf_values]).T
    acf_df.columns = ['ACF']
    ax[i].plot(acf_df.ACF, 'b-', label='data')
    if(conf):
        ax[i].hlines(y=0.05, xmin=0, xmax=len(acf_values), lw=1, 
                 linestyles='--', color='r', label='Conf lvl +/- 0.05')
        ax[i].hlines(y=-0.05, xmin=0, xmax=len(acf_values), lw=1, 
                 linestyles='--', color='r')    
    ax[i].set_title(title)
    ax[i].legend()

    return None


# In[28]:


from statsmodels.tsa.stattools import pacf
def pacf_plot(data, target, ax, i: int, conf: bool, title: str) -> None:

    pacf_values = pacf((data[target].values))
    pacf_df = pd.DataFrame([pacf_values]).T
    pacf_df.columns = ['PACF']
    ax[i].plot(pacf_df.PACF, 'b-', label='data')
    if(conf):
        ax[i].hlines(y=0.05, xmin=0, xmax=len(pacf_values), lw=1, 
                 linestyles='--', color='r', label='Conf lvl +/- 0.05')
        ax[i].hlines(y=-0.05, xmin=0, xmax=len(pacf_values), lw=1, 
                 linestyles='--', color='r')  
    ax[i].set_title(title)
    ax[i].legend()

    return None 


# In[29]:


# autocorrelation/partial autocorrleation
fig, ax = plt.subplots(3,1, figsize = (9, 8), sharex=True, sharey=False)
acf_plot(data=df, target='Revenue', ax=ax, i=0, conf=False,
         title='Autocorrelation (ACF)\nRaw data')
acf_plot(data=train, target='Revenue', ax=ax, i=1, conf=True,
         title='Autocorrelation (ACF)\nTraining data')
pacf_plot(data=train, target='Revenue', ax=ax, i=2, conf=True,
          title='Partial Autocorrelation (PACF)\nTraining data')
fig.supxlabel('Lag') # common x label
fig.supylabel('Correlation') # common y label
title = 'Autocorrelation - Partial Autocorrelation Plots'
save_fig(title)


# ## decompose cleaned data

# https://machinelearningmastery.com/decompose-time-series-data-trend-seasonality/

# In[30]:


# decompose cleaned data - additive
from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(df['Revenue'].values, 
            model='additive', period=1)
result.plot()
title = 'Decomposition on cleaned data'
save_fig(title)

# decompose log data
result = seasonal_decompose(lnrevenue, model='additive', period=1)
result.plot()
pyplot.show()# decompose revenue data - multiplicative
# adapted from https://machinelearningmastery.com/decompose-time-series-data-trend-seasonality/
from matplotlib import pyplot
from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(revenue, model='multiplicative', period=1)
result.plot()
pyplot.show()
# ## auto find p,d,q values

# In[31]:


# use auto arima to find best p,d,q
from pmdarima import auto_arima
import warnings
warnings.filterwarnings('ignore')
pdq = auto_arima(train['Revenue'], 
        trace=True, supress_warings=True)
#pdq.summary()


# ## final model

# ### model (1,1,0) on original data

# In[35]:


# create ARIMA model (1,1,0) on training data
model = ARIMA(df['Revenue'], order=(1,1,0))
results = model.fit()
results.summary()


# ### make a forecast outside of sample data

# In[46]:


# make forecast outside of sample
results.forecast(30)


# ## plot forecast of final model (30-day) compared to the test data

# In[36]:


df.tail(30)


# In[37]:


# prediction for last 30-days
predictions = results.predict(start=700, end=730, type='levels')
print(predictions)


# In[43]:


fig, ax = plt.subplots(1,1, figsize = (9, 8))
pred = plt.plot(predictions, "b", label='Predictions')
plt.plot(test['Revenue'], "r", label='Test data')
plt.xlabel("Date Index")
plt.ylabel("Revenue")
title = 'Final Model Predictions vs Test Data'
plt.legend()
plt.grid()
import matplotlib.dates as mdates
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.xaxis.set_minor_locator(mdates.MonthLocator())
ax.xaxis.set_minor_formatter(mdates.DateFormatter('\n%b'))
plt.title(title)
save_fig(title) 


# In[ ]:




