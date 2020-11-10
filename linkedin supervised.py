#!/usr/bin/env python
# coding: utf-8

# In[58]:


import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  

# Reading data from remote link
url = "http://bit.ly/w-data"
s_data= pd.read_csv(url)
print("Data imported successfully")

s_data.head(10)

# Plotting the distribution of scores
s_data.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# In[70]:



X = s_data.iloc[:, -2].values
X =X.reshape(-1,1)
y = s_data.iloc[:, 1].values
y = y.reshape (-1,1)



# In[71]:


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.2, random_state=0) 


# In[72]:


from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 

print("Training complete.")


# In[73]:


# Plotting the regression line
line = regressor.coef_*X+regressor.intercept_

# Plotting for the test data
plt.scatter(X, y)
plt.plot(X, line);
plt.show()


# In[74]:


print(X_test) # Testing data - In Hours
y_pred = regressor.predict(X_test) # Predicting the scores


# In[75]:


# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': [y_test], 'Predicted': [y_pred]})  
df 


# In[76]:


# You can also test with your own data
#regression.predict([[9.25], [52], [31]])
hours = ([[9.25]])
own_pred = regressor.predict(hours)
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))


# In[77]:


from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred)) 


# In[ ]:




