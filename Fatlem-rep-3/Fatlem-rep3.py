#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import pandas as pd


# In[7]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("Students data.csv")

print(data.columns)

plt.figure(figsize=(6, 4))
sns.countplot(x="gender", data=data)
plt.title("Gender Distribution")
plt.show()


# In[8]:


plt.figure(figsize=(8, 6))
sns.histplot(data=data, x="Functional_analysis", bins=30, kde=True)
plt.title("Math Score Distribution")
plt.show()


# In[9]:


plt.figure(figsize=(8, 6))
sns.boxplot(x="gender", y="Functional_analysis", data=data)
plt.title("Math Score by Gender")
plt.show()

