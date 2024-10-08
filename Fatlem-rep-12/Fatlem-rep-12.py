#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesRegressor


# In[3]:


df = pd.read_csv("../input/playstore-analysis/googleplaystore.csv")


# In[4]:


df.head()


# In[5]:


df.shape


# In[6]:


df.describe()


# In[7]:


df.info()


# ### **Data Cleaning**

# In[8]:


#dropping rows with null values
df.dropna(how='any',inplace=True)


# In[9]:


df['Reviews'] = df['Reviews'].astype(int)


# In[10]:


df.head()


# In[11]:


df['Size'].unique()


# In[12]:


def Kb_to_Mb(size):
    if size.endswith('M'):
        return float(size[:-1])
    elif size.endswith('k'):
        return float(size[:-1])/1000
    else:
        return size


# In[13]:


df['Size'] = df['Size'].apply(lambda x: Kb_to_Mb(x))


# In[14]:


df['Size'].value_counts()


# In[15]:


df['Size'].fillna(method="bfill",inplace=True)


# In[16]:


df['Size'].replace({'Varies with device':11.0},inplace=True)


# In[17]:


df.head()


# In[18]:


df.rename(columns={'Size_MB':'Size_MB'},inplace=True)


# In[19]:


df.head()


# In[20]:


df["Installs"] = df['Installs'].str[:-1]


# In[21]:


df['Installs'] = df['Installs'].apply(lambda x: x.replace(",",""))


# In[22]:


df['Installs'] = df['Installs'].astype(int)


# In[23]:


df.head()


# In[24]:


df['Type'].value_counts()


# In[25]:


df['Price'] = df['Price'].apply(lambda x: x if x == '0' else x[1:])


# In[26]:


df.Price= df.Price.astype(float)


# In[27]:


df.rename(columns={'Price':'Price_in_$'},inplace=True)


# In[28]:


df.head()


# In[29]:


df['Content Rating'].unique()


# In[30]:


df['Category'].unique()


# In[31]:


df['Current Ver'].value_counts()


# In[32]:


df['Android Ver'].value_counts()


# In[33]:


def update_version(version):
    if version.endswith('up'):
        ver = version.split(' ')
        ver = ver[0]
        return ver
    elif version == "Varies with device":
        ver = '4.1.0'
        return str(ver)
    else:
        ver = version.split('-')
        ver = ver[0]
        return str(ver)
        


# In[34]:


df['Android Ver'] = df['Android Ver'].apply(lambda x: update_version(x))


# In[35]:


df.head()


# In[36]:


df['Genres'].unique()


# * ### **Exploratory Data Analysis and Visualization**

# In[37]:


plt.rcParams['figure.figsize'] = (11,9)
df.hist()
plt.show()


# In[38]:


#correlation between variables
plt.rcParams['figure.figsize'] = (12,9)
sns.heatmap(df.corr(),annot=True,cmap="Reds")
plt.show()


# In[39]:


df.head()


# **App with highest Reviews and Installs**

# In[40]:


plt.rcParams['figure.figsize'] = (12,9)
sns.barplot(x='App',y='Installs',hue='Reviews',data = df.sort_values('Installs',ascending=False)[:10])
plt.legend(loc='center')
plt.xticks(rotation=90)
plt.show()


# **App with Highest Ratings**

# In[41]:


plt.rcParams['figure.figsize'] = (12,9)
sns.barplot(x='Rating',y='App',data = df.sort_values('Rating',ascending=False)[:10])
plt.show()


# **Category of App with Highest ratings**

# In[42]:


plt.rcParams['figure.figsize'] = (12,9)
sns.barplot(x='Rating',y='Category',data = df.sort_values('Rating',ascending=False)[:10])
plt.show()


# We can see that the Rating does not depend solely upon Category and Name of the App

# In[43]:


plt.rcParams['figure.figsize'] = (12,9)
sns.countplot(x=df['Type'],data = df)
plt.show()


# **Which Category Apps are free and which one are Paid**

# In[44]:


plt.rcParams['figure.figsize'] = (50,20)
sns.countplot(x=df['Category'],hue='Type',data = df)
plt.show()


# **App with Most number of installs**

# In[45]:


plt.rcParams['figure.figsize'] = (20,9)
sns.countplot(x='App',hue='Installs',data = df.sort_values('Installs',ascending=False)[:5])
plt.show()


# Highest Paid apps

# In[46]:


plt.rcParams['figure.figsize'] = (25,9)
sns.barplot(x='App',y='Price_in_$',data = df.sort_values('Price_in_$',ascending=False)[:10])
plt.show()


# **Which category Apps are most installed by People**

# In[47]:


plt.rcParams['figure.figsize'] = (20,12)
sns.barplot(x='Installs',y='Category',data = df.sort_values('Installs',ascending=False))
plt.show()


# ### **Feature Selection**

# In[48]:


df.columns


# In[49]:


x = df[['Reviews','Size','Installs','Price_in_$']]
y = df[['Rating']]


# In[ ]:





# **Feature Importance**

# In[50]:


bstfeatures = ExtraTreesRegressor()
fit = bstfeatures.fit(x,y)
print(fit.feature_importances_)
feat_imp = pd.Series(bstfeatures.feature_importances_,index=x.columns)
feat_imp.plot(kind='barh')
plt.title("Most important features")
plt.show()


# Hence from above we can see that Reviews contribute  more to the Ratings of the App

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




