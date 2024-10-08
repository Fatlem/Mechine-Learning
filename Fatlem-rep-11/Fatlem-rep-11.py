#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[2]:


import pandas as pd

DATA = '../input/nvidia-corporation-nvda-stock-2015-2024/nvidia_stock_2015_to_2024.csv'

df = pd.read_csv(filepath_or_buffer=DATA, parse_dates=['date'])
df['year'] = df['date'].dt.year
df.head()


# "Let's look first at the price history; because this is a successful company and we have a long series we probably need to look at the data on a logarithmic scale."

# In[3]:


from plotly import express

express.line(data_frame=df, x='date', y=['open', 'high', 'low', 'close'], log_y=True).show()
express.line(data_frame=df, x='date', y=['close', 'adjclose'], log_y=True).show()
express.line(data_frame=df, x='date', y='volume', log_y=True).show()


# #Lowess (or loess)
# 
# Locally weighted scatterplot smoothing
# 
# "Loess stands for locally estimated scatterplot smoothing (lowess stands for locally weighted scatterplot smoothing) and is one of many non-parametric regression techniques, but arguably the most flexible."
# https://www.epa.gov/sites/default/files/2016-07/documents/loess-lowess.pdf

# In[4]:


express.scatter(data_frame=df, x='date', y='volume', trendline='lowess', log_y=True, color='year')


# #Above: 
# 
# "Our volume data looks pretty solid but we have a few days with anomalously low volume."

# "Our price trendline is very smooth even when we use daily adjusted close values."

# In[5]:


express.scatter(data_frame=df, x='date', y='adjclose', trendline='lowess', color='year', log_y=True)


# #Our price trendline is very smooth even when we use daily adjusted close values.

# In[6]:


express.scatter(data_frame=df[['date', 'adjclose']].set_index(keys=['date']).resample('ME').mean().reset_index(), x='date', y='adjclose', trendline='lowess', log_y=True)


# "Here we have filtered out the volume outliers and we get almost an annual layer cake of close prices by year as Zynex's stock price has marched ever higher even as trading volume has declined."

# In[7]:


express.scatter(data_frame=df[df['volume'] > 20000], x='volume', y='adjclose', color='year', log_x=True, log_y=True)

