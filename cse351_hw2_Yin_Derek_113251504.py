#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import datetime
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sn


energy = pd.read_csv(r'C:\Users\derek\Documents\CSE351\HW2\energy_data.csv')
weather = pd.read_csv(r'C:\Users\derek\Documents\CSE351\HW2\weather_data.csv')

weather['time'] = weather['time'].apply(lambda x: pd.to_datetime(x, unit='s'))
# datetime.datetime.fromtimestamp(x)
weather.head()
# weather['time'] = datetime.datetime.fromtimestamp(weather['time'])


# Imported all necessary packages, began by parsing all time fields of the weather column. All Unix time values (seconds after 01/01/1970) were converted to YYYY-MM-DD HH:MM:SS format. 

# In[2]:


energy.tail()


# In[3]:


energy['Date & Time'] = energy['Date & Time'].apply(lambda x: pd.to_datetime(x))
en = energy.set_index('Date & Time').resample('D').sum()
we = weather.set_index('time').resample('D').mean()
#en.resample('D').sum()
# display(we)


# Next, I did the same with the energy dataset's 'Date & Time' Column, converting all cells in the column to a datetime object. Now that the dates in both datasets are represented as datetime objects, we can now set the indices of each data set as the date/time column, resampling the data from half-hourly/hourly intervals to daily intervals. I took the sum when resampling the energy dataset because energy usage is certainly cumulative, but took the mean for the resampling of the weather dataset, because variables like cloud cover and pressure are not cumulative.

# In[4]:


list(en)
#en.rename(columns = {'Date & Time':'time'}, inplace = False)
en.head()
# merged = pd.merge(we, en, on='time', how='outer')


# In[5]:


merged = pd.merge(en, we, left_index = True, right_index = True, how='outer')
merged.head()
# I merged the data on the index of Date and Time after making both datasets compatible for merging.


# In[6]:


merged = merged.dropna()
# Droppign null rows (none to begin with, but used for safety)


# In[7]:


# merged.plot(x='precipIntensity', y = 'use [kW]', style = 'o')

train = merged[:'2014-11-30'] 
#Data up until December 
test = merged['2014-12-01':]
# December Data

# Removing outliers
q_low = train["temperature"].quantile(0.01)
q_hi  = train["temperature"].quantile(0.99)
train = train[(train["temperature"] < q_hi) & (train["temperature"] > q_low)]

X_train = train[['temperature']].values
y_train = train['use [kW]'].values



X_test = test[['temperature']].values
y_test = test['use [kW]'].values

# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# For this model, I split the train test into all data before December, and test set became all data after December. Then, I removed outliers from ONLY the train data. Then, I used temperature to predict use [kW] with the Linear Regression package. 

# In[8]:


merged['use [kW]'].describe()
# Viewing the mean and std of the use [kW] column will be useful in gauging the success of our model.


# In[9]:


y_pred = regressor.predict(X_test)
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df

# df2 = pd.DataFrame({'Date': energy['Date & Time'], 'Predicted': y_pred})


# This dataframe shows the actual use values vs the estimated use in kilowatts values for the month of December. Some predicted values are closer to the actual value than others. 

# In[10]:


print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# My linear regression model takes the data from the first 11 months of 2014 to train, and the month of December as the test data. The root mean squared error of this model is ~7, which is around 25% of the mean of use [kW] across the entire year. Therefore, the model is somewhat adequate at predicting December's use [kW]. 

# Included in the homework ZIP is the CSV dump for this data, as well as included below.

# In[11]:


df = pd.DataFrame({'Date': test.index, 'Predicted': y_pred})
df


# This dataframe shows the predicted energy use [kW] based on temperature for the month of December. This was achieved by training the linear regression model with data from the first 11 months of the year. 

# In[12]:


# compression_opts = dict(method='zip',
#                        archive_name='part3.csv')  
#df.to_csv('part3.zip', index=False,
#          compression=compression_opts)  


# This kernel creates the csv dump but is commented out because it will be included in submission ZIP regardless.

# In[13]:


weather.head()


# In[14]:


weather2 =  weather.set_index('time').resample('D').mean()
weather2['temperature']= weather2['temperature'].apply(lambda x: 1 if x >= 35 else 0)


# In[15]:


logtrain = weather2[:'2014-11-30'] 
#Data up until December 
logtest = weather2['2014-12-01':]
# December Data

X_train2 = logtrain[['dewPoint', 'pressure']].values
y_train2 = logtrain['temperature'].values

X_test2 = logtest[['dewPoint', 'pressure']].values
y_test2 = logtest['temperature'].values

# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
log = LogisticRegression()
log.fit(X_train2, y_train2)


# I split the original weather dataframe into new train and test sets, and trained the model using two columns, dewPoint and pressure, instead of just one column. I did trial and error to see which datapoints produced better F1 scores, which would indicate a better logistic regression model. 

# In[16]:


#y_pred = regressor.predict(X_test)
#df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
#df
y_pred2 = log.predict(X_test2)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(metrics.f1_score(y_test2, y_pred2)))


# The accuracy of our logistic regression is 0.70, which indicates it is better than randomly guessing, but still shows room for improvement.

# In[17]:


df = pd.DataFrame({'Date': logtest.index, 'Predicted': y_pred2})
df


# This dataframe shows the predicted binary classifier for temperature for the month of December based on our Logistic regression.

# In[18]:


# df.to_csv('part4.csv', index=False)


# In[19]:


energy2 = energy
energy2['Date & Time'] = pd.to_datetime(energy2['Date & Time'], errors='coerce', format='%Y-%m-%d %H:%M:%S')
for index, cell in enumerate(energy2['Date & Time']):
        #energy2['Date & Time'][index] = "Day"
        energy2.loc[index, 'Date & Time'] = "Day" if cell.hour >= 19 else "Night"
        
        
energy2.rename(columns= {'Date & Time' : "Time of Day"}, inplace = True)
energy2.head()
#display(energy2["Date & Time"][17518].hour)


# First, I classified every row as either at Night or Day by checking whether or not the hour of the day is >= 19 (7:00PM) and setting the value accordingly. Then, I changed the title of the column as well

# In[20]:


energy2=energy2.groupby('Time of Day').sum()
energy2.head()


# In[21]:


# Utilizing a simple bar graph, we can plot the differences in energy usage between day and night.
energy2.plot.bar( y ="Washer [kW]", ylabel= "use [kW]", title = 'Total Washer Use (Day vs Night)')


# As shown in the bar graph above, the washer uses significantly more energy at night as opposed to day. This can be attributed to people working jobs and being out the house in the day time, then coming home and utilizing the washer at night. 

# In[22]:


# Utilizing a simple bar graph, we can plot the differences in energy usage between day and night.
energy2.plot.bar( y ="Fridge (R) [kW]", ylabel = "use [kW]", title = "Total Fridge Use (Day vs Night)")


# Similarly to the energy usage of the Washer, the fridge uses much more energy at night as people are more likely to be at home at nighttime. Opening and closing the fridge frequently will increase energy usage, and this occurs more often at nightime. 
