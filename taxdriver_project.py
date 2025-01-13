#!/usr/bin/env python
# coding: utf-8

# # Maximizing Revenue for Taxi Cab Drivers through Payment Type Analysis

# # Problem Statement
# 
# In the fast-paced taxi booking sector, making the most of revenue is essential for long-term success and driver happiness. Our goal is to use data-driven insights to maximise revenue streams for taxi drivers in order to meet this need. Our research aims to determine whether payment methods have an impact on fare pricing by focusing on the relationship between payment type and fare amount.

# # Objective
# 
# This project's main goal is to run an A/B test to examine the relationship between the total fare and the method of payment. We use Python hypothesis testing and descriptive statistics to extract useful information that can help taxi drivers generate more cash. In particular, we want to find out if there is a big difference in the fares for those who pay with credit cards versus those who pay with cash.

# # Research Question
# 
# Is there a relationship between total fare amount and payment type and can we nudge customers towards payment methods that generate higher revenue for drivers, without negatively impacting customer experience?

# # Coding
# 

# In[1]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import scipy.stats as st
import warnings 

warnings.filterwarnings('ignore')


# In[2]:


data = pd.read_csv(r"yellow_tripdata_2020-01.csv")


# In[3]:


data.head(10)


# In[4]:


data.shape


# In[5]:


data["tpep_pickup_datetime"] = pd.to_datetime(data["tpep_pickup_datetime"])
data["tpep_dropoff_datetime"] = pd.to_datetime(data["tpep_dropoff_datetime"])


# In[6]:


data.dtypes


# In[7]:


data["duration"] =  data["tpep_dropoff_datetime"] - data["tpep_pickup_datetime"] 


# In[8]:


data.head()


# In[9]:


data["duration"] = data["duration"].dt.total_seconds()/60


# In[10]:


data.head()


# In[11]:


df = data[['passenger_count','trip_distance','payment_type','fare_amount','total_amount']]


# In[12]:


df.head()


# In[13]:


df.shape


# In[14]:


df.dtypes


# In[15]:


df.describe()


# In[16]:


df.isnull().sum()


# In[17]:


df.dropna(inplace = True)


# In[18]:


df['passenger_count'] = df['passenger_count'].astype(int)


# In[19]:


df['payment_type'] = df['payment_type'].astype(int)


# In[20]:


df.duplicated()


# In[21]:


df.drop_duplicates(inplace = True)


# In[22]:


df.shape


# In[23]:


df['passenger_count'].value_counts(normalize = True)


# In[24]:


df = df[(df['passenger_count'] > 0) & (df['passenger_count'] < 7)] 


# In[25]:


df['payment_type'].value_counts(normalize = True)


# In[26]:


df = df[df['payment_type'] < 3]


# In[27]:


df['payment_type'] = df['payment_type'].astype('int64')
df['passenger_count'] = df['passenger_count'].astype('int64')


# In[28]:


df.dtypes


# In[29]:


df['payment_type'].replace([1,2],["card","cash"], inplace= True)


# In[30]:


df.head()


# In[31]:


df.describe()


# In[32]:


df = df[df['trip_distance'] > 0]
df = df[df['fare_amount'] > 0]
df = df[df['total_amount'] > 0]


# In[33]:


df = df.join(data['duration'])


# In[34]:


df.head()


# In[35]:


df.isnull().sum()


# In[36]:


df.duplicated().sum()


# In[37]:


df = df[df['duration'] > 0]


# In[38]:


plt.boxplot(df['fare_amount'])
plt.show()


# In[39]:


for col in ['trip_distance','duration','total_amount','trip_distance']:
    q1 = df[col].quantile(0.25)
    q2 = df[col].quantile(0.75)
    
    IQR = q2 -q1
    
    lb = q1 - 1.5*IQR
    ub = q2 + 1.5*IQR
    
    df = df[(df[col] >= lb) & (df[col] <=ub)]


# In[40]:


df.shape


# In[41]:


df.head()


# In[42]:


plt.figure(figsize=(12, 5))
plt.subplot(1,2,1)

plt.hist(df[df['payment_type'] == 'card']["fare_amount"], color='sandybrown', alpha=0.7,edgecolor="black", label='Card Payments')
plt.hist(df[df['payment_type'] == 'cash']["fare_amount"], color='lightgreen', alpha=0.7,edgecolor="black", label='Cash Payments')
plt.title('Distribution of Fare Amount by Payment Type')
plt.legend()

plt.subplot(1,2,2)
plt.hist(df[df['payment_type'] == 'card']["trip_distance"], color='sandybrown', alpha=0.7,edgecolor="black", label='Card Payments')
plt.hist(df[df['payment_type'] == 'cash']["trip_distance"], color='lightgreen', alpha=0.7,edgecolor="black", label='Cash Payments')
plt.title('Distribution of trip distance by Payment Type')
plt.legend()

plt.show()


# In[43]:


sizes = df['payment_type'].value_counts(normalize=True)
labels = sizes.index
colors = ['sandybrown', 'lightgreen']

plt.figure(figsize=(8, 5))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140, shadow=True)


plt.title('Payment Types Distribution')
plt.legend(loc='best')
plt.axis('equal')  

plt.show()


# In[44]:


passenger_count = df.groupby(['payment_type','passenger_count'])["passenger_count"].count().unstack()


# In[45]:


passenger_count


# In[46]:


# Plot
passenger_count.plot(kind='bar', stacked=False, figsize=(8, 6) ,color = ["darkorange" ,"orange","gold","peachpuff","moccasin","sandybrown"] )

# Add title and labels
plt.title('Passenger Count Distribution by Payment Type')
plt.xlabel('Payment Type')
plt.ylabel('Count of Passengers')

# Show plot
plt.legend(title='Passenger Count')
plt.show()


# In[47]:


df


# # hypthesis testing
# 
# null hypthesis - there is no difference in average fair between customers who pay through credit card and customers who pay                      through cash
# 
# alternative hypothesis - there is difference in average fair between customers who pay through credit card and customers who pay                          through cash

# In[48]:


card_sample = df[df['payment_type']=="card"]['fare_amount']
cash_sample = df[df['payment_type']=="cash"]['fare_amount']



# In[49]:


t_stats,p_value = st.ttest_ind(a = card_sample,b = cash_sample, equal_var = False)
print('T statistic',t_stats,'P value',p_value)


# solution 
# 
# With such a low p-value, we reject the null hypothesis. This means that there is a significant difference in the average fare between customers who pay through credit card and customers who pay through cash.

# In[ ]:




