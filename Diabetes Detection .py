#!/usr/bin/env python
# coding: utf-8

# # DIABETES PREDICTION

# In[1]:


import numpy as np
import pandas as pd


# # IMPORTING DATASET

# In[2]:


dataset = pd.read_csv("Diabetes_Dataset.csv")
dataset.head(10)


# # ANALYZING THE DATASET 

# In[3]:


dataset.info()


# In[4]:


dataset.isnull().sum()


# In[5]:


x = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]


# # SPLITTING THE DATASET

# In[6]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x , y , test_size = 25 , random_state = 0)


# # APPLYING CLASSIFIER AND EVALUATION

# ## RANDOM FOREST 

# In[7]:


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 6, criterion = 'entropy', random_state = 0)
classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)


# In[8]:


from sklearn.metrics  import accuracy_score
acc_random_forest = round(accuracy_score(y_pred, y_test), 2)*100
print("Accuracy of Random Forest Model is:",acc_random_forest)


# ## LOGISTIC REGRESSION

# In[9]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(solver = 'lbfgs' , max_iter=1000)
logreg.fit(x_train,y_train)

y_pred = logreg.predict(x_test)

acc_logreg = round(accuracy_score(y_pred, y_test), 2)*100
print("Accuracy of Logistic Regression Model is:",acc_logreg)


# ## KNN

# In[10]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)

acc_knn = round(accuracy_score(y_pred, y_test), 2)*100
print("Accuracy of KNN Model is:",acc_knn)


# In[ ]:




