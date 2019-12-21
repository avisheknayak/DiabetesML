#!/usr/bin/env python
# coding: utf-8

# In[134]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix,r2_score
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[135]:


dataset=pd.read_csv('diabtesupdated.csv')


# In[136]:


dataset.shape


# In[137]:


dataset.describe()


# In[138]:


dataset.head()


# In[139]:


dataset.isnull().any()


# In[160]:


#sns.heatmap(dataset)
sns.distplot(dataset['Glucose'],kde=True, bins=10);


# In[173]:


sns.distplot(dataset['Insulin'],kde=True, bins=10,color='green');


# In[157]:


sns.distplot(dataset['BMI'],kde=True, bins=10);


# In[172]:


sns.distplot(dataset['Pregnancies'],kde=True, bins=10,color='orange');


# In[156]:


X = dataset.drop('Outcome', axis=1)
Y = dataset['Outcome']


# In[121]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)


# In[122]:


classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)


# In[123]:


y_pred = classifier.predict(X_test)


# In[124]:


print(confusion_matrix(y_test, y_pred))


# In[125]:


print(classification_report(y_test, y_pred))


# In[126]:


df = pd.DataFrame({'Actual Type of Diabetes ': y_test, 'Predicted Type of Diabetes': y_pred})
df1 = df.head(50)
print(df1)


# In[169]:


df1.plot(kind='bar',figsize=(10,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


# In[128]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[129]:


r2_score(y_test,y_pred)


# In[132]:


print("The accuracy of the model is :" ,accuracy_score(y_test, y_pred)*100)


# In[ ]:





# In[ ]:




